from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ..runtime import ConversationRuntime
from ..session import CompactionEntry, CustomMessageEntry, MessageEntry, Session
from ..tools import BaseTool, get_tool, tools_by_name
from .chat import ChatLog
from .commands import CommandsMixin
from .input import InputBox
from .tool_output import escape_tool_output_text, truncate_tool_output_text
from .widgets import InfoBar, StatusLine, format_path


class SessionUIMixin:
    _cwd: str
    _hide_thinking: bool
    _current_block_type: str | None
    _api_key: str | None
    _tools: list[BaseTool]
    _openai_compat_auth_mode: Any
    _anthropic_compat_auth_mode: Any
    _runtime: ConversationRuntime

    # Methods from App - declared for type checking
    if TYPE_CHECKING:
        query_one: Any

    # Methods from other mixins/main class
    def _sync_runtime_state(self) -> None: ...

    def _resolve_system_prompt(self, session: Session | None = None) -> str:
        return self._runtime.resolve_system_prompt(session)

    def _extract_text_content(self, content: str | list[TextContent | ImageContent]) -> str:
        if isinstance(content, str):
            return content

        parts: list[str] = []
        image_number = 0
        for part in content:
            if isinstance(part, TextContent):
                parts.append(part.text)
            elif isinstance(part, ImageContent):
                image_number += 1
                marker = (
                    f"[Image #{image_number} {part.display_name}]"
                    if part.display_name
                    else "[image]"
                )
                parts.append(marker)

        return "".join(parts).strip() or "(no content)"

    def _format_tool_call(self, tool_call: ToolCall) -> str:
        tool = tools_by_name.get(tool_call.name)
        if not tool:
            return json.dumps(tool_call.arguments) if tool_call.arguments else ""

        try:
            params = tool.params(**tool_call.arguments)
            return tool.format_call(params)
        except Exception:
            return json.dumps(tool_call.arguments) if tool_call.arguments else ""

    def _format_tool_result_text(self, message: ToolResultMessage) -> tuple[str, str | None]:
        if message.content:
            parts = [part.text for part in message.content if isinstance(part, TextContent)]
            full_text = "".join(parts)
            collapsed_text, truncated = truncate_tool_output_text(full_text)
            return collapsed_text, escape_tool_output_text(full_text) if truncated else None

        return "", None

    def _render_session_entries(self, session: Session) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        started_tools: set[str] = set()

        for entry in session.active_entries:
            if isinstance(entry, MessageEntry):
                message = entry.message
                if isinstance(message, UserMessage):
                    chat.add_user_message(self._extract_text_content(message.content))
                elif isinstance(message, AssistantMessage):
                    for part in message.content:
                        if isinstance(part, TextContent) and part.text:
                            chat.add_content(part.text)
                        elif isinstance(part, ThinkingContent) and part.thinking:
                            block = chat.add_thinking(part.thinking)
                            if self._hide_thinking:
                                block.add_class("-hidden")
                        elif isinstance(part, ToolCall):
                            call_msg = self._format_tool_call(part)
                            tool = get_tool(part.name)
                            icon = tool.tool_icon if tool else "→"
                            chat.start_tool(part.name, part.id, call_msg, icon=icon)
                            started_tools.add(part.id)
                elif isinstance(message, ToolResultMessage):
                    tool_id = message.tool_call_id
                    if tool_id not in started_tools:
                        tool = get_tool(message.tool_name)
                        icon = tool.tool_icon if tool else "→"
                        chat.start_tool(message.tool_name, tool_id, "", icon=icon)
                        started_tools.add(tool_id)

                    markup = True
                    ui_summary = message.ui_summary
                    ui_details = message.ui_details
                    ui_details_full = message.ui_details_full
                    if ui_summary is None and ui_details is None:
                        ui_details, ui_details_full = self._format_tool_result_text(message)

                    images = [part for part in message.content if isinstance(part, ImageContent)]
                    chat.set_tool_result(
                        tool_id,
                        ui_summary,
                        ui_details,
                        not message.is_error,
                        markup=markup,
                        ui_details_full=ui_details_full,
                        images=images or None,
                    )
            elif isinstance(entry, CompactionEntry):
                chat.add_compaction_message(entry.tokens_before)
            elif isinstance(entry, CustomMessageEntry):
                target_session_id = str(
                    (entry.details or {}).get("target_session_id") or ""
                ).strip()
                query = str((entry.details or {}).get("query") or "").strip()
                if entry.custom_type == CommandsMixin.HANDOFF_BACKLINK_TYPE and target_session_id:
                    chat.add_handoff_link_message(
                        label="Origin session",
                        target_session_id=target_session_id,
                        query=query,
                        direction="back",
                    )
                elif (
                    entry.custom_type == CommandsMixin.HANDOFF_FORWARD_LINK_TYPE
                    and target_session_id
                ):
                    chat.add_handoff_link_message(
                        label="Handoff session",
                        target_session_id=target_session_id,
                        query=query,
                        direction="forward",
                    )
                elif entry.custom_type == "shell_command":
                    details = entry.details or {}
                    command = str(details.get("command") or "")
                    output = str(details.get("output") or "")
                    success = bool(details.get("success", True))
                    tool_id = f"shell-replay-{entry.id}"
                    chat.start_tool("bash", tool_id, f"$ {command}", icon="$")
                    chat.set_tool_result(
                        tool_id, None, output or "(no output)", success, markup=False
                    )
                elif entry.display:
                    chat.add_info_message(entry.content)

    async def _load_session_by_id(self, session_id: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        input_box = self.query_one("#input-box", InputBox)

        try:
            session = Session.continue_by_id(self._cwd, session_id)
        except Exception as exc:
            chat.add_info_message(f"Failed to load linked session: {exc}", error=True)
            input_box.focus()
            return

        if session.session_file is None:
            chat.add_info_message(
                "Failed to load linked session: missing session file", error=True
            )
            input_box.focus()
            return

        await self._load_session(session.session_file)

    async def _load_session(self, session_path: str | Path) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)
        status = self.query_one("#status-line", StatusLine)
        input_box = self.query_one("#input-box", InputBox)

        try:
            session = self._runtime.load_session(session_path)
        except Exception as exc:
            chat.add_info_message(f"Failed to load session: {exc}", error=True)
            input_box.focus()
            return

        self._sync_runtime_state()
        self._current_block_type = None

        status.reset()
        token_totals = session.token_totals()
        info_bar.set_tokens(
            token_totals.input_tokens,
            token_totals.output_tokens,
            token_totals.context_tokens,
            token_totals.cache_read_tokens,
            token_totals.cache_write_tokens,
        )
        info_bar.set_file_changes(session.file_changes_summary())

        model_info = session.model
        if model_info:
            provider, model_id, _ = model_info
            info_bar.set_model(model_id, provider)

        info_bar.set_thinking_level(self._runtime.thinking_level)

        await chat.remove_all_children()

        chat.add_session_info(getattr(self, "VERSION", ""))

        if self._runtime.agent:
            chat.add_loaded_resources(
                context_paths=[
                    format_path(f.path) for f in self._runtime.agent.context.agents_files
                ],
                skills=self._runtime.agent.context.skills,
                tools=self._runtime.tools,
            )

        self._render_session_entries(session)
        chat.add_info_message("Resumed session")
        input_box.focus()
