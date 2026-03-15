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
- Each tool is permission-checked; safe read-only tools auto-approve, mutating tools
  yield ToolApprovalEvent and await user approval before executing
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
from enum import StrEnum

from pydantic import ValidationError

from . import config as kon_config
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
_DEFAULT_TOOL_CALL_IDLE_TIMEOUT_SECONDS = 60.0
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


def _get_tool_call_idle_timeout_seconds() -> float | None:
    timeout = kon_config.llm.tool_call_idle_timeout_seconds
    if timeout <= 0:
        return None
    return timeout or _DEFAULT_TOOL_CALL_IDLE_TIMEOUT_SECONDS


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
    try:
        arguments = json.loads(tool_call_data["arguments"])
    except json.JSONDecodeError:
        arguments = {}

    tool_call = ToolCall(id=tool_call_data["id"], name=tool_call_data["name"], arguments=arguments)

    tool = get_tool(tool_call.name)
    display = ""
    if tool:
        try:
            params = tool.params(**arguments)
            display = tool.format_call(params)
        except (TypeError, KeyError, ValueError, ValidationError):
            pass

    return PendingToolCall(tool_call=tool_call, tool=tool, display=display)


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
            display=result.display,
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
    if cancel_event is None:
        return await future
    if cancel_event.is_set():
        return None
    cancel_task = asyncio.create_task(cancel_event.wait())
    done, pending = await asyncio.wait({future, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if future in done:
        return future.result()
    return None


async def run_single_turn(
    provider: BaseProvider,
    messages: list[Message],
    tools: list[BaseTool],
    system_prompt: str | None = None,
    turn: int = 0,
    cancel_event: asyncio.Event | None = None,
    retry_delays: list[int] | None = None,
) -> AsyncIterator[StreamEvent]:
    tool_defs = get_tool_definitions(tools) if tools else None

    if cancel_event and cancel_event.is_set():
        yield InterruptedEvent(message="Interrupted by user")
        yield TurnEndEvent(
            turn=turn, assistant_message=None, tool_results=[], stop_reason=StopReason.INTERRUPTED
        )
        return

    delays = retry_delays if retry_delays is not None else [2, 4, 8]
    stream: LLMStream | None = None

    for attempt_num, delay in enumerate([*delays, None]):
        try:
            stream = await provider.stream(messages, system_prompt=system_prompt, tools=tool_defs)
            break  # Success, exit retry loop
        except Exception as e:
            if provider.should_retry_for_error(e) and delay is not None:
                yield RetryEvent(
                    attempt=attempt_num + 1, total_attempts=len(delays), delay=delay, error=str(e)
                )
                await asyncio.sleep(delay)
                continue
            yield ErrorEvent(error=str(e))  # Not retryable or retries exhausted
            yield TurnEndEvent(
                turn=turn, assistant_message=None, tool_results=[], stop_reason=StopReason.ERROR
            )
            return

    # Stream should be set at this point
    assert stream is not None

    content: list[TextContent | ThinkingContent | ToolCall] = []
    tool_results: list[ToolResultMessage] = []

    think_buffer: list[str] = []
    think_signature: str | None = None
    text_buffer: list[str] = []

    # Collect tool calls during streaming, execute after stream completes
    pending_tool_calls: list[dict] = []
    current_tool_call: dict | None = None

    # Token counting for tool argument streaming
    _tool_arg_chunk_counter = 0
    _tool_arg_token_count = 0

    current_state: StreamState | None = None
    stop_reason: StopReason = StopReason.STOP
    interrupted = False
    has_meaningful_output = False

    def _finalize_current_state(include_empty: bool = True) -> list[StreamEvent]:
        nonlocal current_state, current_tool_call, think_buffer, think_signature, text_buffer

        events: list[StreamEvent] = []

        if current_state == StreamState.THINK:
            full_thinking = "".join(think_buffer)
            if include_empty or full_thinking:
                content.append(ThinkingContent(thinking=full_thinking, signature=think_signature))
                events.append(ThinkingEndEvent(thinking=full_thinking, signature=think_signature))
            think_buffer = []
            think_signature = None
        elif current_state == StreamState.TEXT:
            full_text = "".join(text_buffer)
            if include_empty or full_text:
                content.append(TextContent(text=full_text))
                events.append(TextEndEvent(text=full_text))
            text_buffer = []
        elif current_state == StreamState.TOOL_CALL and current_tool_call:
            pending_tool_calls.append(current_tool_call)
            current_tool_call = None

        current_state = None
        return events

    # Race stream chunks against cancel_event so ESC takes effect immediately,
    # not just when the next chunk happens to arrive from the API.
    stream_iter = stream.__aiter__()
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None
    tool_call_idle_timeout_seconds = _get_tool_call_idle_timeout_seconds()

    while True:
        if cancel_event and cancel_event.is_set():
            interrupted = True
            stop_reason = StopReason.INTERRUPTED
            break

        next_task = asyncio.create_task(_safe_anext(stream_iter))
        chunk_timeout = (
            tool_call_idle_timeout_seconds
            if (
                tool_call_idle_timeout_seconds is not None
                and (current_state == StreamState.TOOL_CALL or pending_tool_calls)
            )
            else None
        )

        if cancel_task and not cancel_task.done():
            done, _ = await asyncio.wait(
                {next_task, cancel_task},
                timeout=chunk_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                next_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_task
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
                for finalize_event in _finalize_current_state(include_empty=False):
                    yield finalize_event
                if pending_tool_calls and stop_reason == StopReason.STOP:
                    stop_reason = StopReason.TOOL_USE
                break

            if cancel_task in done:
                next_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_task
                interrupted = True
                stop_reason = StopReason.INTERRUPTED
                break

            chunk = next_task.result()
        elif chunk_timeout is not None:
            try:
                chunk = await asyncio.wait_for(next_task, timeout=chunk_timeout)
            except TimeoutError:
                next_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_task
                timeout_secs = chunk_timeout or 0
                yield WarningEvent(
                    warning=(
                        f"Tool-call stream stalled for {timeout_secs:g}s; "
                        "continuing with collected arguments."
                    )
                )
                for finalize_event in _finalize_current_state(include_empty=False):
                    yield finalize_event
                if pending_tool_calls and stop_reason == StopReason.STOP:
                    stop_reason = StopReason.TOOL_USE
                break
        else:
            chunk = await next_task

        if chunk is _STREAM_EXHAUSTED:
            for finalize_event in _finalize_current_state():
                yield finalize_event
            if pending_tool_calls and stop_reason == StopReason.STOP:
                stop_reason = StopReason.TOOL_USE
            break

        match chunk:
            case ThinkPart(think=t, signature=sig):
                if current_state and current_state != StreamState.THINK:
                    for finalize_event in _finalize_current_state():
                        yield finalize_event

                if current_state != StreamState.THINK:
                    yield ThinkingStartEvent()

                current_state = StreamState.THINK
                think_buffer.append(t)
                has_meaningful_output = True
                if sig:
                    think_signature = sig

                yield ThinkingDeltaEvent(delta=t)

            case TextPart(text=t):
                if not has_meaningful_output and not t.strip():
                    continue

                if current_state and current_state != StreamState.TEXT:
                    for finalize_event in _finalize_current_state():
                        yield finalize_event

                if current_state != StreamState.TEXT:
                    yield TextStartEvent()

                current_state = StreamState.TEXT
                text_buffer.append(t)
                if t.strip():
                    has_meaningful_output = True

                yield TextDeltaEvent(delta=t)

            case ToolCallStart(id=id, name=name):
                has_meaningful_output = True
                if current_state and current_state != StreamState.TOOL_CALL:
                    for finalize_event in _finalize_current_state():
                        yield finalize_event
                elif current_state == StreamState.TOOL_CALL and current_tool_call:
                    pending_tool_calls.append(current_tool_call)
                    current_tool_call = None

                # Reset token counters when starting a new tool call
                _tool_arg_chunk_counter = 0
                _tool_arg_token_count = 0

                current_state = StreamState.TOOL_CALL
                current_tool_call = {"id": id, "name": name, "arguments": ""}

                yield ToolStartEvent(tool_call_id=id, tool_name=name)

            case ToolCallDelta(arguments_delta=delta):
                if current_tool_call:
                    current_tool_call["arguments"] += delta
                    yield ToolArgsDeltaEvent(tool_call_id=current_tool_call["id"], delta=delta)

                    # Count tokens and fire update event every Nth chunk after threshold tokens
                    _tool_arg_chunk_counter += 1
                    _tool_arg_token_count += _count_tokens(delta)

                    if (
                        _tool_arg_token_count > _TOOL_ARGS_TOKEN_DISPLAY_THRESHOLD
                        and _tool_arg_chunk_counter % _TOOL_ARGS_TOKEN_CHUNK_UPDATE_INTERVAL == 0
                    ):
                        yield ToolArgsTokenUpdateEvent(
                            tool_call_id=current_tool_call["id"],
                            tool_name=current_tool_call["name"],
                            token_count=_tool_arg_token_count,
                        )

            case StreamDone(stop_reason=reason):
                stop_reason = reason

                for finalize_event in _finalize_current_state():
                    yield finalize_event

            case StreamError(error=err):
                yield ErrorEvent(error=err)
                stop_reason = StopReason.ERROR

    # Clean up the cancel waiter task
    if cancel_task and not cancel_task.done():
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_task

    # Handle interruption - finalize partial content
    if interrupted:
        for finalize_event in _finalize_current_state(include_empty=False):
            yield finalize_event

    # Process all pending tool calls:
    # 1. First, yield all ToolEndEvents (UI shows all tools in pending state)
    # 2. Then execute each tool and yield ToolResultEvent
    finalized_tools: list[PendingToolCall] = []
    for tool_data in pending_tool_calls:
        pending = _finalize_tool_call_data(tool_data, tools)
        finalized_tools.append(pending)
        content.append(pending.tool_call)

        yield ToolEndEvent(
            tool_call_id=pending.tool_call.id,
            tool_name=pending.tool_call.name,
            arguments=pending.tool_call.arguments,
            display=pending.display,
        )

    # Now execute tools one by one
    for pending in finalized_tools:
        file_changes = None
        if cancel_event and cancel_event.is_set():
            result = _create_skipped_tool_result(pending.tool_call)
        else:
            # Unknown tools get ALLOW; they'll error in _execute_tool anyway
            decision = (
                check_permission(pending.tool, pending.tool_call.arguments)
                if pending.tool
                else PermissionDecision.ALLOW
            )

            approved = True
            if decision == PermissionDecision.PROMPT:
                loop = asyncio.get_running_loop()
                future: asyncio.Future[ApprovalResponse] = loop.create_future()
                yield ToolApprovalEvent(
                    tool_call_id=pending.tool_call.id,
                    tool_name=pending.tool_call.name,
                    future=future,
                )
                approved = await _await_approval(future, cancel_event) == ApprovalResponse.APPROVE

            if approved:
                result, file_changes = await _execute_tool(
                    pending.tool_call, pending.tool, cancel_event
                )
            else:
                result = _create_skipped_tool_result(
                    pending.tool_call,
                    reason="Tool call denied by user. Ask the user what they'd like you to do instead.",
                )

        tool_results.append(result)
        yield ToolResultEvent(
            tool_call_id=pending.tool_call.id,
            tool_name=pending.tool_call.name,
            result=result,
            file_changes=file_changes,
        )

    if interrupted:
        yield InterruptedEvent(message="Interrupted by user")

    usage = stream.usage
    assistant_message = AssistantMessage(content=content, usage=usage, stop_reason=stop_reason)

    yield TurnEndEvent(
        turn=turn,
        assistant_message=assistant_message,
        tool_results=tool_results,
        stop_reason=stop_reason,
    )
