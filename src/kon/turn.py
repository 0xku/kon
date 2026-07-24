"""
Single turn execution - one LLM request/response cycle with streaming.

Streams chunks from the LLM and yields typed events as they arrive:
- ThinkingStartEvent/DeltaEvent/EndEvent - model's reasoning
- TextStartEvent/DeltaEvent/EndEvent - response text
- ToolStartEvent/ArgsDeltaEvent/EndEvent - tool calls being built
- ToolApprovalEvent - when a tool requires user approval
- ToolResultEvent - after each tool execution
- TurnEndEvent - final event with complete AssistantMessage

Tool execution strategy:
- All tool calls are collected during streaming
- After streaming completes, all ToolEndEvents are yielded first (UI shows pending state)
- Each tool is permission-checked; safe read-only tools auto-approve while
  mutating tools yield ToolApprovalEvent and await user approval before executing
- Then ToolResultEvent is yielded with the result (or denial reason)

Cancellation handling:
- Races each stream chunk against cancel_event using asyncio.wait(FIRST_COMPLETED)
- ESC takes effect immediately, not just when the next chunk arrives
- Finalizes any partial content (thinking/text/tool call in progress)
- Skips remaining tool executions with "Interrupted by user" placeholder
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, StrEnum, auto

from pydantic import ValidationError

from . import config as kon_config
from .async_utils import OperationCancelledError, await_or_cancel, sleep_or_cancel
from .core.errors import format_error
from .core.types import (
    AssistantMessage,
    FileChanges,
    ImageContent,
    Message,
    StopReason,
    StreamDone,
    StreamError,
    TextContent,
    TextPart,
    ThinkingContent,
    ThinkPart,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
    ToolResultMessage,
)
from .events import (
    ErrorEvent,
    InterruptedEvent,
    RetryEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolApprovalEvent,
    ToolArgsDeltaEvent,
    ToolArgsTokenUpdateEvent,
    ToolEndEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    WarningEvent,
)
from .llm import BaseProvider
from .llm.base import LLMStream
from .permissions import ApprovalResponse, PermissionDecision, check_permission
from .tools import BaseTool, get_tool, get_tool_definitions

_STREAM_EXHAUSTED = object()
_TOOL_ARGS_TOKEN_DISPLAY_THRESHOLD = 20
_TOOL_ARGS_TOKEN_CHUNK_UPDATE_INTERVAL = 4


def _count_tokens(text: str) -> int:
    return len(text) // 4


class StreamState(StrEnum):
    THINK = "think"
    TEXT = "text"
    TOOL_CALL = "tool_call"


@dataclass
class PendingToolCall:
    tool_call: ToolCall
    tool: BaseTool | None
    display: str
    approval_preview: str = ""
    preflight_error: str | None = None


async def _safe_anext(aiter):
    """
    Get next item, returning _STREAM_EXHAUSTED on StopAsyncIteration.

    StopAsyncIteration cannot propagate out of an asyncio task,
    so we catch it and return a sentinel instead.
    """
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return _STREAM_EXHAUSTED


async def _cancel_and_reap(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _close_stream(stream: LLMStream) -> None:
    with contextlib.suppress(Exception):
        await stream.aclose()


def tool_call_idle_timeout_seconds() -> float | None:
    timeout = kon_config.llm.tool_call_idle_timeout_seconds
    return None if timeout <= 0 else timeout


def _create_skipped_tool_result(
    tool_call: ToolCall, reason: str = "Interrupted by user"
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=[TextContent(text=reason)],
        is_error=True,
    )


def _finalize_tool_call_data(tool_call_data: dict, tools: list[BaseTool]) -> PendingToolCall:
    arguments_raw = tool_call_data["arguments"]
    initial_arguments = tool_call_data.get("initial_arguments")
    initial_arguments_dict = initial_arguments if isinstance(initial_arguments, dict) else {}
    stalled = tool_call_data.get("stalled", False)
    preflight_error: str | None = None

    stripped_args = arguments_raw.strip()
    if stripped_args:
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            if stalled:
                # The stream timed out mid-arguments, so whatever we collected is
                # truncated and initial_arguments is a stale snapshot from the
                # start of the call. Refuse to execute rather than guess.
                arguments = {}
                preflight_error = (
                    "Tool call arguments were cut off when the stream stalled; "
                    "skipping execution instead of running with truncated arguments."
                )
            elif initial_arguments_dict:
                arguments = initial_arguments_dict
            else:
                arguments = {}
                preflight_error = (
                    "Tool call arguments were incomplete or invalid JSON; "
                    "skipping execution instead of running with empty arguments."
                )
    else:
        arguments = initial_arguments_dict

    tool_call = ToolCall(id=tool_call_data["id"], name=tool_call_data["name"], arguments=arguments)

    tool = get_tool(tool_call.name)
    display = ""
    approval_preview = ""
    if tool and preflight_error is None:
        try:
            params = tool.params(**arguments)
            display = tool.format_call(params)
            approval_preview = tool.format_preview(params) or ""
        except (TypeError, KeyError, ValueError, ValidationError):
            if stalled:
                preflight_error = (
                    "Tool call arguments failed validation after the stream stalled "
                    "mid-call, so they are likely incomplete; skipping execution."
                )
            else:
                preflight_error = (
                    "Tool call arguments failed validation before execution; skipping execution."
                )

    return PendingToolCall(
        tool_call=tool_call,
        tool=tool,
        display=display,
        approval_preview=approval_preview,
        preflight_error=preflight_error,
    )


async def _execute_tool(
    tool_call: ToolCall, tool: BaseTool | None, cancel_event: asyncio.Event | None = None
) -> tuple[ToolResultMessage, FileChanges | None]:
    if not tool:
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=[TextContent(text=f"Unknown tool: {tool_call.name}")],
            is_error=True,
        ), None

    try:
        params = tool.params(**tool_call.arguments)
        result: ToolResult = await tool.execute(params, cancel_event=cancel_event)

        content: list[TextContent | ImageContent] = []
        if result.result:
            content.append(TextContent(text=result.result))
        if result.images:
            content.extend(result.images)
        if not content:
            content.append(TextContent(text="(no output)"))

        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=content,
            ui_summary=result.ui_summary,
            ui_details=result.ui_details,
            ui_details_full=result.ui_details_full,
            is_error=not result.success,
            file_changes=result.file_changes,
        ), result.file_changes
    except Exception as e:
        return ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=[TextContent(text=f"Error executing tool: {e}")],
            is_error=True,
        ), None


async def _await_approval(
    future: asyncio.Future[ApprovalResponse], cancel_event: asyncio.Event | None
) -> ApprovalResponse | None:
    try:
        return await await_or_cancel(future, cancel_event)
    except OperationCancelledError:
        return None


class _ChunkOutcome(Enum):
    CHUNK = auto()
    EXHAUSTED = auto()
    CANCELLED = auto()
    STALLED = auto()


class _TurnRunner:
    """
    State for one streaming turn, split into phases:

    1. _open_stream() - request the stream, retrying transient failures
    2. _consume_stream() - drain chunks, buffering content and tool calls
    3. _run_pending_tools() - permission-check and execute collected tool calls

    run() orchestrates the phases and emits the final TurnEndEvent.
    """

    def __init__(
        self,
        provider: BaseProvider,
        messages: list[Message],
        tools: list[BaseTool],
        system_prompt: str | None,
        turn: int,
        cancel_event: asyncio.Event | None,
        retry_delays: list[int] | None,
    ):
        self._provider = provider
        self._messages = messages
        self._tools = tools
        self._system_prompt = system_prompt
        self._turn = turn
        self._cancel_event = cancel_event
        self._retry_delays = retry_delays if retry_delays is not None else [2, 4, 8]

        self._stream: LLMStream | None = None

        self._content: list[TextContent | ThinkingContent | ToolCall] = []
        self._tool_results: list[ToolResultMessage] = []
        self._tool_call_count = 0

        self._think_buffer: list[str] = []
        self._think_signature: str | None = None
        self._text_buffer: list[str] = []

        # Collect tool calls during streaming, execute after stream completes
        self._pending_tool_calls: list[dict] = []
        self._active_tool_calls: dict[int, dict] = {}

        # Token counting for tool argument streaming
        self._tool_arg_counters: dict[int, tuple[int, int]] = {}

        self._current_state: StreamState | None = None
        self._stop_reason: StopReason = StopReason.STOP
        self._interrupted = False

    async def run(self) -> AsyncIterator[StreamEvent]:
        if self._is_cancelled():
            for event in self._interrupted_turn_end():
                yield event
            return

        async for event in self._open_stream():
            yield event
        if self._stream is None:
            return

        async for event in self._consume_stream():
            yield event

        async for event in self._run_pending_tools():
            yield event

        if self._interrupted:
            yield InterruptedEvent(message="Interrupted by user")

        assistant_message = AssistantMessage(
            content=self._content, usage=self._stream.usage, stop_reason=self._stop_reason
        )
        yield TurnEndEvent(
            turn=self._turn,
            assistant_message=assistant_message,
            tool_results=self._tool_results,
            stop_reason=self._stop_reason,
            tool_call_count=self._tool_call_count,
        )

    # -- Phase 1: open the stream, retrying transient failures ----------------

    async def _open_stream(self) -> AsyncIterator[StreamEvent]:
        """Request the LLM stream. Leaves self._stream as None on terminal failure."""
        tool_defs = get_tool_definitions(self._tools) if self._tools else None

        for attempt_num, delay in enumerate([*self._retry_delays, None]):
            if self._is_cancelled():
                for event in self._interrupted_turn_end():
                    yield event
                return

            try:
                self._stream = await self._provider.stream(
                    self._messages, system_prompt=self._system_prompt, tools=tool_defs
                )
                return
            except Exception as e:
                if self._provider.should_retry_for_error(e) and delay is not None:
                    yield RetryEvent(
                        attempt=attempt_num + 1,
                        total_attempts=len(self._retry_delays),
                        delay=delay,
                        error=format_error(e),
                    )
                    if await sleep_or_cancel(delay, self._cancel_event):
                        for event in self._interrupted_turn_end():
                            yield event
                        return
                    continue
                yield ErrorEvent(error=format_error(e))  # Not retryable or retries exhausted
                yield TurnEndEvent(
                    turn=self._turn,
                    assistant_message=None,
                    tool_results=[],
                    stop_reason=StopReason.ERROR,
                )
                return

    # -- Phase 2: drain the stream, buffering content ---------------------------

    async def _consume_stream(self) -> AsyncIterator[StreamEvent]:
        assert self._stream is not None
        stream_iter = self._stream.__aiter__()
        # Race stream chunks against cancel_event so ESC takes effect immediately,
        # not just when the next chunk happens to arrive from the API.
        cancel_task = (
            asyncio.create_task(self._cancel_event.wait()) if self._cancel_event else None
        )
        tool_call_timeout = tool_call_idle_timeout_seconds()

        try:
            while True:
                if self._is_cancelled():
                    self._mark_interrupted()
                    break

                chunk_timeout = (
                    tool_call_timeout
                    if (
                        tool_call_timeout is not None
                        and (
                            self._current_state == StreamState.TOOL_CALL
                            or self._pending_tool_calls
                        )
                    )
                    else None
                )
                outcome, chunk = await self._next_chunk(stream_iter, cancel_task, chunk_timeout)

                if outcome is _ChunkOutcome.STALLED:
                    await _close_stream(self._stream)
                    # Calls still streaming arguments may be truncated; mark them
                    # so finalization can skip execution instead of running with
                    # partial arguments.
                    for tool_call_data in self._active_tool_calls.values():
                        tool_call_data["stalled"] = True
                    timeout_secs = chunk_timeout or 0
                    yield WarningEvent(
                        warning=(
                            f"Tool-call stream stalled for {timeout_secs:g}s; "
                            "continuing with collected arguments."
                        )
                    )
                    # Some local providers intermittently miss terminal stream events
                    # after a tool call is fully emitted. If we're already in a tool
                    # call path, finalize what we have and continue execution.
                    for event in self._finalize_current_state(include_empty=False):
                        yield event
                    self._promote_tool_use_stop_reason()
                    break

                if outcome is _ChunkOutcome.CANCELLED:
                    await _close_stream(self._stream)
                    self._mark_interrupted()
                    break

                if outcome is _ChunkOutcome.EXHAUSTED:
                    for event in self._finalize_current_state():
                        yield event
                    self._promote_tool_use_stop_reason()
                    break

                for event in self._handle_chunk(chunk):
                    yield event
        finally:
            # Clean up the cancel waiter task
            if cancel_task and not cancel_task.done():
                await _cancel_and_reap(cancel_task)

        # Handle interruption - finalize partial content
        if self._interrupted:
            for event in self._finalize_current_state(include_empty=False):
                yield event

    async def _next_chunk(
        self, stream_iter, cancel_task: asyncio.Task | None, chunk_timeout: float | None
    ) -> tuple[_ChunkOutcome, object]:
        next_task = asyncio.create_task(_safe_anext(stream_iter))

        if cancel_task and not cancel_task.done():
            done, _ = await asyncio.wait(
                {next_task, cancel_task},
                timeout=chunk_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await _cancel_and_reap(next_task)
                return _ChunkOutcome.STALLED, None
            if cancel_task in done:
                await _cancel_and_reap(next_task)
                return _ChunkOutcome.CANCELLED, None
            chunk = next_task.result()
        elif chunk_timeout is not None:
            try:
                chunk = await asyncio.wait_for(next_task, timeout=chunk_timeout)
            except TimeoutError:
                await _cancel_and_reap(next_task)
                return _ChunkOutcome.STALLED, None
        else:
            chunk = await next_task

        if chunk is _STREAM_EXHAUSTED:
            return _ChunkOutcome.EXHAUSTED, None
        return _ChunkOutcome.CHUNK, chunk

    def _handle_chunk(self, chunk: object) -> list[StreamEvent]:
        match chunk:
            case ThinkPart(think=t, signature=sig):
                return self._on_think_part(t, sig)
            case TextPart(text=t):
                return self._on_text_part(t)
            case ToolCallStart(id=id, name=name, index=index, arguments=initial_arguments):
                return self._on_tool_call_start(id, name, index, initial_arguments)
            case ToolCallDelta(index=index, arguments_delta=delta, replace=replace):
                return self._on_tool_call_delta(index, delta, replace)
            case StreamDone(stop_reason=reason):
                self._stop_reason = reason
                return self._finalize_current_state()
            case StreamError(error=err):
                self._stop_reason = StopReason.ERROR
                return [ErrorEvent(error=err)]
        return []

    def _on_think_part(self, think: str, signature: str | None) -> list[StreamEvent]:
        # Anthropic can emit signature-only ThinkParts (redacted/
        # encrypted reasoning with no plain-text). Capture the
        # signature but don't open a thinking UI block, otherwise
        # the renderer shows an empty bordered stub. Also trim any
        # leading whitespace from the first visible thinking delta;
        # Anthropic may emit an initial empty/space delta.
        if self._current_state != StreamState.THINK:
            think = think.lstrip()
            if not think:
                if signature:
                    self._think_signature = signature
                return []

        events: list[StreamEvent] = []
        if self._current_state and self._current_state != StreamState.THINK:
            events.extend(self._finalize_current_state())

        if self._current_state != StreamState.THINK:
            events.append(ThinkingStartEvent())

        self._current_state = StreamState.THINK
        self._think_buffer.append(think)
        if signature:
            self._think_signature = signature

        events.append(ThinkingDeltaEvent(delta=think))
        return events

    def _on_text_part(self, text: str) -> list[StreamEvent]:
        # Skip whitespace-only text that would start a new (empty)
        # content block — prevents phantom gaps between thinking
        # and tool-call blocks.
        if not text.strip() and self._current_state != StreamState.TEXT:
            return []

        events: list[StreamEvent] = []
        if self._current_state and self._current_state != StreamState.TEXT:
            events.extend(self._finalize_current_state())

        if self._current_state != StreamState.TEXT:
            events.append(TextStartEvent())

        self._current_state = StreamState.TEXT
        self._text_buffer.append(text)

        events.append(TextDeltaEvent(delta=text))
        return events

    def _on_tool_call_start(
        self, tool_call_id: str, name: str, index: int, initial_arguments: dict | None
    ) -> list[StreamEvent]:
        self._tool_call_count += 1
        events: list[StreamEvent] = []
        if self._current_state and self._current_state != StreamState.TOOL_CALL:
            events.extend(self._finalize_current_state())

        initial_arguments_json = ""
        if initial_arguments:
            try:
                initial_arguments_json = json.dumps(initial_arguments)
            except (TypeError, ValueError):
                initial_arguments_json = ""

        self._current_state = StreamState.TOOL_CALL
        self._active_tool_calls[index] = {
            "id": tool_call_id,
            "name": name,
            "arguments": initial_arguments_json,
            "initial_arguments": initial_arguments or {},
        }

        events.append(ToolStartEvent(tool_call_id=tool_call_id, tool_name=name))
        return events

    def _on_tool_call_delta(self, index: int, delta: str, replace: bool) -> list[StreamEvent]:
        tool_call = self._active_tool_calls.get(index)
        if not tool_call:
            return []

        if replace:
            tool_call["arguments"] = delta
            chunk_count, token_count = 0, 0
        else:
            tool_call["arguments"] += delta
            chunk_count, token_count = self._tool_arg_counters.get(index, (0, 0))

        events: list[StreamEvent] = [ToolArgsDeltaEvent(tool_call_id=tool_call["id"], delta=delta)]

        # Count tokens and fire update event every Nth chunk after threshold tokens
        chunk_count += 1
        token_count += _count_tokens(delta)
        self._tool_arg_counters[index] = (chunk_count, token_count)

        if (
            token_count > _TOOL_ARGS_TOKEN_DISPLAY_THRESHOLD
            and chunk_count % _TOOL_ARGS_TOKEN_CHUNK_UPDATE_INTERVAL == 0
        ):
            events.append(
                ToolArgsTokenUpdateEvent(
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    token_count=token_count,
                )
            )
        return events

    def _finalize_current_state(self, include_empty: bool = True) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if self._current_state == StreamState.THINK:
            full_thinking = "".join(self._think_buffer)
            if include_empty or full_thinking:
                self._content.append(
                    ThinkingContent(thinking=full_thinking, signature=self._think_signature)
                )
                events.append(
                    ThinkingEndEvent(thinking=full_thinking, signature=self._think_signature)
                )
            self._think_buffer = []
            self._think_signature = None
        elif self._current_state == StreamState.TEXT:
            full_text = "".join(self._text_buffer)
            if include_empty or full_text:
                self._content.append(TextContent(text=full_text))
                events.append(TextEndEvent(text=full_text))
            self._text_buffer = []
        elif self._current_state == StreamState.TOOL_CALL and self._active_tool_calls:
            self._pending_tool_calls.extend(self._active_tool_calls.values())
            self._active_tool_calls.clear()
            self._tool_arg_counters.clear()

        self._current_state = None
        return events

    # -- Phase 3: execute collected tool calls ---------------------------------

    async def _run_pending_tools(self) -> AsyncIterator[StreamEvent]:
        # 1. First, yield all ToolEndEvents (UI shows all tools in pending state)
        # 2. Then execute each tool and yield ToolResultEvent
        finalized_tools: list[PendingToolCall] = []
        for tool_data in self._pending_tool_calls:
            pending = _finalize_tool_call_data(tool_data, self._tools)
            finalized_tools.append(pending)
            self._content.append(pending.tool_call)

            yield ToolEndEvent(
                tool_call_id=pending.tool_call.id,
                tool_name=pending.tool_call.name,
                arguments=pending.tool_call.arguments,
                display=pending.display,
            )

        for pending in finalized_tools:
            async for event in self._run_one_tool(pending):
                yield event

    async def _run_one_tool(self, pending: PendingToolCall) -> AsyncIterator[StreamEvent]:
        file_changes = None
        if self._is_cancelled():
            result = _create_skipped_tool_result(pending.tool_call)
        elif pending.preflight_error is not None:
            result = _create_skipped_tool_result(pending.tool_call, reason=pending.preflight_error)
        else:
            approved = True
            if self._needs_approval(pending):
                loop = asyncio.get_running_loop()
                future: asyncio.Future[ApprovalResponse] = loop.create_future()
                yield ToolApprovalEvent(
                    tool_call_id=pending.tool_call.id,
                    tool_name=pending.tool_call.name,
                    display=pending.approval_preview,
                    future=future,
                )
                approved = (
                    await _await_approval(future, self._cancel_event) == ApprovalResponse.APPROVE
                )

            if approved:
                result, file_changes = await _execute_tool(
                    pending.tool_call, pending.tool, self._cancel_event
                )
            else:
                result = _create_skipped_tool_result(
                    pending.tool_call,
                    reason=(
                        "Tool call denied by user. Ask them what they'd like you to do instead."
                    ),
                )

        self._tool_results.append(result)
        yield ToolResultEvent(
            tool_call_id=pending.tool_call.id,
            tool_name=pending.tool_call.name,
            result=result,
            file_changes=file_changes,
        )

    @staticmethod
    def _needs_approval(pending: PendingToolCall) -> bool:
        # Unknown tools get ALLOW; they'll error in _execute_tool anyway
        if not pending.tool:
            return False
        decision = check_permission(pending.tool, pending.tool_call.arguments)
        return decision == PermissionDecision.PROMPT

    # -- Shared state helpers ---------------------------------------------------

    def _is_cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _mark_interrupted(self) -> None:
        self._interrupted = True
        self._stop_reason = StopReason.INTERRUPTED

    def _promote_tool_use_stop_reason(self) -> None:
        if self._pending_tool_calls and self._stop_reason == StopReason.STOP:
            self._stop_reason = StopReason.TOOL_USE

    def _interrupted_turn_end(self) -> list[StreamEvent]:
        return [
            InterruptedEvent(message="Interrupted by user"),
            TurnEndEvent(
                turn=self._turn,
                assistant_message=None,
                tool_results=[],
                stop_reason=StopReason.INTERRUPTED,
            ),
        ]


async def run_single_turn(
    provider: BaseProvider,
    messages: list[Message],
    tools: list[BaseTool],
    system_prompt: str | None = None,
    turn: int = 0,
    cancel_event: asyncio.Event | None = None,
    retry_delays: list[int] | None = None,
) -> AsyncIterator[StreamEvent]:
    runner = _TurnRunner(
        provider=provider,
        messages=messages,
        tools=tools,
        system_prompt=system_prompt,
        turn=turn,
        cancel_event=cancel_event,
        retry_delays=retry_delays,
    )
    async for event in runner.run():
        yield event
