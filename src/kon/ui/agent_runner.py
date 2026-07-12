"""Drives agent runs: forwards agent events to the chat UI and handles ! / !! shell
commands typed at the prompt."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from kon import config

from ..core.types import ImageContent, StopReason, ToolResultMessage
from ..events import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    RetryEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolApprovalEvent,
    ToolArgsTokenUpdateEvent,
    ToolEndEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    WarningEvent,
)
from ..notify import NotificationEvent, notify
from ..permissions import ApprovalResponse
from ..runtime import ConversationRuntime
from ..tools import get_tool
from ..tools.bash import BashParams, BashTool
from .chat import ChatLog
from .queue_ui import QueuedPrompt, _unpack_queue_item
from .widgets import InfoBar, StatusLine

_NOTIFY_EVENTS = (AgentEndEvent, ToolApprovalEvent)


class AgentRunnerMixin:
    _is_running: bool
    _cancel_event: asyncio.Event | None
    _steer_event: asyncio.Event | None
    _interrupt_requested: bool
    _abort_shown: bool
    _current_block_type: str | None
    _hide_thinking: bool
    _approval_future: asyncio.Future[ApprovalResponse] | None
    _approval_tool_id: str | None
    _approval_selection: ApprovalResponse
    _pending_session_switch_id: str | None
    _shell_tool_counter: int
    _pending_queue: deque[QueuedPrompt]
    _steer_queue: deque[QueuedPrompt]
    _runtime: ConversationRuntime

    if TYPE_CHECKING:
        app: Any
        query_one: Any
        run_worker: Any

        def _update_queue_display(self) -> None: ...
        def _clear_approval_state(self) -> None: ...
        def _show_pending_update_notice_if_idle(self) -> None: ...
        def _format_tool_result_text(
            self, message: ToolResultMessage
        ) -> tuple[str, str | None]: ...
        async def _load_session_by_id(self, session_id: str) -> None: ...

    def _should_notify_for_event(self, event: object) -> bool:
        return self._notification_event_type(event) is not None

    def _notification_event_type(self, event: object) -> NotificationEvent | None:
        if not config.notifications.enabled:
            return None
        if not isinstance(event, _NOTIFY_EVENTS):
            return None
        if isinstance(event, AgentEndEvent):
            if event.stop_reason == StopReason.INTERRUPTED:
                return None
            if event.stop_reason == StopReason.ERROR:
                return "error"
            return "completion"
        if isinstance(event, ToolApprovalEvent):
            return "permission"
        return None

    async def _run_agent(self, prompt: str, images: list[ImageContent] | None = None) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-line", StatusLine)
        info_bar = self.query_one("#info-bar", InfoBar)

        agent = self._runtime.prepare_for_run()
        if agent is None:
            chat.add_info_message("Agent not initialized")
            self._is_running = False
            return
        current_prompt = prompt
        current_images = images

        while True:
            was_interrupted = False

            self._cancel_event = asyncio.Event()
            self._steer_event = asyncio.Event()
            self._abort_shown = False
            self._current_block_type = None
            if self._interrupt_requested:
                self._cancel_event.set()

            status.set_status("working")

            try:
                async for event in agent.run(
                    current_prompt,
                    images=current_images,
                    cancel_event=self._cancel_event,
                    steer_event=self._steer_event,
                ):
                    notification_event = self._notification_event_type(event)
                    if notification_event:
                        notify(notification_event)

                    if await self._render_agent_event(event, chat, status, info_bar):
                        was_interrupted = True

            except Exception as e:
                chat.add_info_message(str(e), error=True)

            if was_interrupted and not self._abort_shown:
                chat.add_aborted_message("Interrupted by user")
                self._abort_shown = True

            self._interrupt_requested = False
            self._cancel_event = None
            self._steer_event = None
            self._clear_approval_state()
            status.set_status("idle")

            if was_interrupted:
                self._pending_queue.clear()
                self._steer_queue.clear()
                self._update_queue_display()
                break

            queued = self._dequeue_next_prompt()
            if queued is None:
                break
            next_display, next_query, next_images = queued
            chat.add_user_message(next_display)
            current_prompt = next_query
            current_images = next_images

        self._is_running = False

        if self._pending_session_switch_id:
            session_id = self._pending_session_switch_id
            self._pending_session_switch_id = None
            self.run_worker(self._load_session_by_id(session_id), exclusive=True)

        self._show_pending_update_notice_if_idle()

    def _dequeue_next_prompt(self) -> tuple[str, str, list[ImageContent]] | None:
        # Steer messages take priority — drain steer queue first
        if self._steer_queue:
            queued = self._steer_queue.popleft()
        elif self._pending_queue:
            queued = self._pending_queue.popleft()
        else:
            return None
        self._update_queue_display()
        return _unpack_queue_item(queued)

    async def _render_agent_event(
        self, event: object, chat: ChatLog, status: StatusLine, info_bar: InfoBar
    ) -> bool:
        """Render one agent event into the UI. Returns True if it signals interruption."""
        was_interrupted = False

        match event:
            case AgentStartEvent():
                pass

            case TurnStartEvent():
                pass

            case ThinkingStartEvent():
                if self._current_block_type != "thinking":
                    if self._current_block_type:
                        chat.end_block()
                    block = chat.start_thinking()
                    if self._hide_thinking:
                        block.add_class("-hidden")
                    self._current_block_type = "thinking"

            case ThinkingDeltaEvent(delta=d):
                await chat.append_to_current(d)

            case ThinkingEndEvent():
                pass

            case TextStartEvent():
                if self._current_block_type != "content":
                    if self._current_block_type:
                        chat.end_block()
                    chat.start_content()
                    self._current_block_type = "content"

            case TextDeltaEvent(delta=d):
                await chat.append_to_current(d)

            case TextEndEvent():
                pass

            case ToolStartEvent(tool_call_id=id, tool_name=name):
                if self._current_block_type:
                    chat.end_block()
                tool = get_tool(name)
                icon = tool.tool_icon if tool else "→"
                chat.start_tool(name, id, "", icon=icon)
                self._current_block_type = "tool_call"
                status.increment_tool_calls()
                status.set_streaming_tokens(0)  # Reset token count for new tool

            case ToolArgsTokenUpdateEvent(token_count=tc):
                status.set_streaming_tokens(tc)

            case ToolEndEvent(tool_call_id=id, display=display):
                chat.update_tool_call_msg(id, display)

            case ToolApprovalEvent(tool_call_id=id, tool_name=name, display=disp, future=f):
                self.app.bell()
                self._approval_selection = ApprovalResponse.APPROVE
                chat.show_tool_approval(
                    id, preview=disp or None, selected=self._approval_selection
                )
                self._approval_future = f
                self._approval_tool_id = id

            case ToolResultEvent(tool_call_id=id, result=r, file_changes=fc):
                self._approval_future = None
                self._approval_tool_id = None
                if r:
                    markup = True
                    ui_summary = r.ui_summary
                    ui_details = r.ui_details
                    ui_details_full = r.ui_details_full
                    if ui_summary is None and ui_details is None and r.content:
                        ui_details, ui_details_full = self._format_tool_result_text(r)
                    success = not r.is_error
                    chat.set_tool_result(
                        id,
                        ui_summary,
                        ui_details,
                        success,
                        markup=markup,
                        ui_details_full=ui_details_full,
                    )
                if fc:
                    info_bar.update_file_changes(fc.path, fc.added, fc.removed)

            case TurnEndEvent():
                if event.assistant_message and event.assistant_message.usage:
                    usage = event.assistant_message.usage
                    info_bar.update_tokens(
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.cache_read_tokens,
                        usage.cache_write_tokens,
                    )

            case InterruptedEvent():
                was_interrupted = True
                if self._current_block_type:
                    chat.end_block()
                    self._current_block_type = None

            case CompactionStartEvent():
                if self._current_block_type:
                    chat.end_block()
                    self._current_block_type = None
                chat.show_spinner_status("Auto-compacting...")

            case CompactionEndEvent(tokens_before=tb, aborted=ab, reason=why):
                if ab:
                    msg = "Compaction failed"
                    if why:
                        msg += f": {why}"
                    chat.show_status(msg)
                else:
                    chat.add_compaction_message(tb)

            case RetryEvent(attempt=a, total_attempts=t, delay=d, error=e):
                msg = f"Request failed (attempt {a}/{t}), retrying in {d}s; Error: {e}"
                chat.add_info_message(msg, error=True)

            case ErrorEvent(error=e):
                chat.add_info_message(str(e), error=True)

            case WarningEvent(warning=w):
                chat.add_info_message(str(w), warning=True)

            case AgentEndEvent(stop_reason=reason):
                if reason == StopReason.INTERRUPTED:
                    was_interrupted = True
                if self._current_block_type:
                    chat.end_block()
                self._current_block_type = None

        return was_interrupted

    def _handle_shell_command(self, display_text: str, original_text: str) -> None:
        """Handle shell commands prefixed with ! or !!"""
        if self._is_running:
            return

        chat = self.query_one("#chat-log", ChatLog)

        # Determine if we should send output to LLM
        send_to_llm = display_text.startswith("!!")

        command_text = display_text[2:] if send_to_llm else display_text[1:]
        command_text = command_text.strip()

        if not command_text:
            return

        # Add user message showing the command
        chat.add_user_message(display_text)

        # Execute the command
        self._is_running = True
        self.run_worker(self._execute_shell_command(command_text, send_to_llm), exclusive=True)

    async def _execute_shell_command(self, command: str, send_to_llm: bool) -> None:
        """Execute a shell command and display the result"""
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-line", StatusLine)

        try:
            # Create bash tool instance
            bash_tool = BashTool()

            # Create cancellation event for this command
            cancel_event = asyncio.Event()
            self._cancel_event = cancel_event

            # Execute the command
            status.set_status("running")
            # Manual shell output should render like regular bash tool output:
            # collapsed preview with ctrl+o expansion when details are available.
            result = await bash_tool.execute(
                BashParams(command=command), cancel_event=cancel_event, inline_output=False
            )

            # Persist the command and its output so session resume and /export
            # include manual shell commands, not just agent tool calls.
            session = self._runtime.session
            if session is not None:
                prefix = "!!" if send_to_llm else "!"
                session.append_custom_message(
                    "shell_command",
                    f"{prefix}{command}",
                    details={
                        "command": command,
                        "output": result.result or "",
                        "success": result.success,
                    },
                )

            # Start tool block and route the result through ChatLog so manual
            # shell commands use the same rendering/expansion path as agent tools.
            self._shell_tool_counter += 1
            tool_id = f"shell-{self._shell_tool_counter}"
            chat.start_tool("bash", tool_id, f"$ {command}", icon="$")

            # Display the result
            if result.success:
                ui_summary = result.ui_summary
                ui_details = result.ui_details
                markup = True
                if ui_summary is None and ui_details is None:
                    ui_summary = result.result or "(no output)"
                    markup = False
            else:
                ui_summary = result.ui_summary or "Command failed"
                ui_details = result.ui_details or result.result
                markup = True

            chat.set_tool_result(
                tool_id,
                ui_summary,
                ui_details,
                result.success,
                markup=markup,
                ui_details_full=result.ui_details_full,
            )

            # If using !!, send output to LLM for follow-up unless the command was interrupted.
            if send_to_llm and result.result and not cancel_event.is_set():
                prompt = (
                    "Shell command output:\n\n```\n"
                    f"{result.result}\n```\n\nWhat would you like me to do with this?"
                )
                self._is_running = True
                await self._run_agent(prompt)
                return

        except Exception as e:
            chat.add_info_message(f"Error executing command: {e}", error=True)
        finally:
            self._is_running = False
            self._interrupt_requested = False
            self._cancel_event = None
            status.set_status("idle")
