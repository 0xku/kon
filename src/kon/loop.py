"""
Run the agent loop and stream events for the UI.

Each turn runs `run_single_turn()`, forwards turn/tool events immediately, persists assistant/tool
messages to the session, and decides whether to continue. After every turn, overflow compaction
may run and emit its own start/end events so the UI can reflect that state in real time.

The loop ends on stop/error/interruption, compaction pause mode, or max turns.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from . import config as kon_config
from .context import Context, formatted_agent_mds, formatted_git_context, formatted_skills
from .core.compaction import generate_summary, is_overflow
from .core.errors import format_error
from .core.types import (
    AssistantMessage,
    ImageContent,
    Message,
    StopReason,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    Event,
    InterruptedEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .llm import BaseProvider
from .session import MessageEntry, Session
from .tools import BaseTool
from .turn import run_single_turn


def build_system_prompt(
    cwd: str, context: Context | None = None, tools: list[BaseTool] | None = None
) -> str:
    now = datetime.now()
    date_time = now.strftime("%A, %B %d, %Y at %I:%M %p %Z").strip()

    if context is None:
        context = Context.load(cwd)

    prompt = kon_config.llm.system_prompt.content

    if tools:
        guidelines: list[str] = []
        seen_guidelines: set[str] = set()
        for tool in tools:
            for guideline in tool.prompt_guidelines:
                if guideline in seen_guidelines:
                    continue
                guidelines.append(guideline)
                seen_guidelines.add(guideline)
        if guidelines:
            prompt += "\n\n# Tool usage\n\n- " + "\n- ".join(guidelines)

    if context.agents_files:
        prompt += "\n\n" + formatted_agent_mds(context.agents_files)

    if context.skills:
        prompt += "\n\n" + formatted_skills(context.skills)

    if kon_config.llm.system_prompt.git_context:
        git_context = formatted_git_context(cwd)
        if git_context:
            prompt += "\n\n" + git_context

    prompt += f"\n\n# Env\n\nCurrent date and time: {date_time}"
    prompt += f"\nCurrent working directory: {cwd}"

    return prompt


@dataclass
class AgentConfig:
    context_window: int | None = None
    max_output_tokens: int | None = None


class Agent:
    def __init__(
        self,
        provider: BaseProvider,
        tools: list[BaseTool],
        session: Session,
        cwd: str | None = None,
        context: Context | None = None,
        system_prompt: str | None = None,
        config: AgentConfig | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.session = session
        self.config = config or AgentConfig()
        self._cwd = cwd or os.getcwd()
        self._context = context or Context.load(self._cwd)
        self._system_prompt = system_prompt or build_system_prompt(
            self._cwd, self._context, tools=tools
        )
        self._run_usage = Usage()

    @property
    def context(self) -> Context:
        return self._context

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def reload_context(self) -> None:
        self._context = Context.load(self._cwd)
        self._system_prompt = build_system_prompt(self._cwd, self._context, tools=self.tools)

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

    def _add_usage(self, usage: Usage | None) -> None:
        if usage:
            self._run_usage.input_tokens += usage.input_tokens
            self._run_usage.output_tokens += usage.output_tokens
            self._run_usage.cache_read_tokens += usage.cache_read_tokens
            self._run_usage.cache_write_tokens += usage.cache_write_tokens

    async def run(
        self,
        query: str,
        images: list[ImageContent] | None = None,
        cancel_event: asyncio.Event | None = None,
        steer_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        self._run_usage = Usage()

        if images:
            user_content: list[TextContent | ImageContent] = [TextContent(text=query), *images]
            user_message = UserMessage(content=user_content)
        else:
            user_message = UserMessage(content=query)

        self.session.append_message(user_message)

        yield AgentStartEvent()

        turn = 0
        stop_reason = StopReason.STOP
        was_interrupted = False

        system_prompt = self._system_prompt

        try:
            max_turns = kon_config.agent.max_turns
            while turn < max_turns:
                if cancel_event and cancel_event.is_set():
                    was_interrupted = True
                    stop_reason = StopReason.INTERRUPTED
                    yield InterruptedEvent(message="Interrupted by user")
                    break

                if steer_event and steer_event.is_set():
                    stop_reason = StopReason.STEER
                    break

                turn += 1
                yield TurnStartEvent(turn=turn)

                messages = self.session.messages
                tool_results: list[ToolResultMessage] = []
                async for event in run_single_turn(
                    provider=self.provider,
                    messages=messages,
                    tools=self.tools,
                    system_prompt=system_prompt,
                    turn=turn,
                    cancel_event=cancel_event,
                ):
                    yield event

                    if isinstance(event, TurnEndEvent):
                        if event.assistant_message:
                            self._add_usage(event.assistant_message.usage)
                            self.session.append_message(event.assistant_message)
                        tool_results = event.tool_results
                        stop_reason = event.stop_reason
                        for result in tool_results:
                            self.session.append_message(result)
                    elif isinstance(event, InterruptedEvent):
                        was_interrupted = True

                if was_interrupted or stop_reason == StopReason.INTERRUPTED:
                    stop_reason = StopReason.INTERRUPTED
                    break

                if steer_event and steer_event.is_set():
                    stop_reason = StopReason.STEER
                    break

                # Check for context overflow after each turn.
                # We iterate events instead of awaiting a single compaction result so
                # CompactionStartEvent can be forwarded immediately and the UI can
                # render a "compacting" state while summary generation is running.
                did_compact = False
                async for compaction_event in self._check_compaction(
                    stop_reason, system_prompt, cancel_event
                ):
                    yield compaction_event
                    if isinstance(compaction_event, CompactionEndEvent):
                        did_compact = True
                if did_compact:
                    if kon_config.compaction.on_overflow == "pause":
                        break
                    # Continue mode: synthetic user message was injected, continue loop
                    continue

                if stop_reason != StopReason.TOOL_USE:
                    break

            if turn >= max_turns and not was_interrupted and stop_reason == StopReason.TOOL_USE:
                stop_reason = StopReason.LENGTH

        except Exception as e:  # intentionally broad — top-level boundary; crash = broken TUI
            yield ErrorEvent(error=format_error(e))
            stop_reason = StopReason.ERROR

        yield AgentEndEvent(stop_reason=stop_reason, total_turns=turn, total_usage=self._run_usage)

    async def _check_compaction(
        self, stop_reason: StopReason, system_prompt: str, cancel_event: asyncio.Event | None
    ) -> AsyncIterator[CompactionStartEvent | CompactionEndEvent]:
        if stop_reason == StopReason.ERROR:
            return

        # Get the latest assistant message that has usage.
        # The most recent assistant entry can be interrupted/error and have no usage.
        last_usage: Usage | None = None
        for entry in reversed(self.session.active_entries):
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage is None:
                    continue
                last_usage = usage
                break

        if last_usage is None:
            return

        context_window = self.config.context_window or kon_config.agent.default_context_window
        max_output = self.config.max_output_tokens or self.provider.config.max_tokens or 0
        buffer_tokens = kon_config.compaction.buffer_tokens

        if not is_overflow(last_usage, context_window, max_output, buffer_tokens):
            return

        if cancel_event and cancel_event.is_set():
            return

        tokens_before = (
            last_usage.input_tokens
            + last_usage.output_tokens
            + last_usage.cache_read_tokens
            + last_usage.cache_write_tokens
        )

        # Yield start event immediately so UI can show status
        yield CompactionStartEvent()

        try:
            # Use all_messages (uncompacted) for summarization so LLM sees full history
            summary = await generate_summary(
                self.session.all_messages, self.provider, system_prompt
            )

            # Everything before is summarized, nothing "kept"
            first_kept_id = self.session.leaf_id or ""

            self.session.append_compaction(
                summary=summary, first_kept_entry_id=first_kept_id, tokens_before=tokens_before
            )

            # In continue mode, inject synthetic continue message
            if kon_config.compaction.on_overflow == "continue":
                continue_msg = UserMessage(
                    content=(
                        "Continue if you have next steps, or stop and ask for clarification if you"
                        " are unsure how to proceed. If there is nothing to do don't add a large"
                        " preamble, just summarise everything so far in 2-3 lines and be done."
                    )
                )
                self.session.append_message(continue_msg)

            yield CompactionEndEvent(tokens_before=tokens_before)

        except Exception as e:
            yield CompactionEndEvent(
                tokens_before=tokens_before, aborted=True, reason=format_error(e)
            )
