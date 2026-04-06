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
from ..llm import ApiType, BaseProvider, ProviderConfig, get_max_tokens, get_model
from ..loop import Agent, build_system_prompt
from ..session import CompactionEntry, CustomMessageEntry, MessageEntry, Session
from ..tools import BaseTool, get_tool, tools_by_name
from .chat import ChatLog
from .commands import CommandsMixin
from .input import InputBox
from .widgets import InfoBar, StatusLine, format_path


class SessionUIMixin:
    # Attributes provided by the App subclass
    _cwd: str
    _agent: Any
    _hide_thinking: bool
    _session: Session | None
    _current_block_type: str | None
    _model: str
    _model_provider: str | None
    _thinking_level: str
    _api_key: str | None
    _provider: BaseProvider | None
    _tools: list[BaseTool]

    # Methods from App - declared for type checking
    if TYPE_CHECKING:
        query_one: Any

    # Methods from other mixins/main class
    def _get_provider_api_type(self, provider: BaseProvider) -> ApiType: ...
    def _create_provider(self, api_type: ApiType, config: ProviderConfig) -> BaseProvider: ...
    def _apply_thinking_level_style(self, level: str) -> None: ...

    def _extract_text_content(self, content: str | list[TextContent | ImageContent]) -> str:
        if isinstance(content, str):
            return content

        parts: list[str] = []
        for part in content:
            if isinstance(part, TextContent):
                parts.append(part.text)
            elif isinstance(part, ImageContent):
                parts.append("[image]")

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

    def _truncate_tool_output(self, text: str, max_lines: int = 5) -> str:
        if not text:
            return text

        lines = text.split("\n")
        if len(lines) > max_lines:
            hidden = len(lines) - max_lines
            lines = lines[:max_lines]
            lines.append(f"... ({hidden} more lines)")

        return "\n".join(lines)

    def _format_tool_result_text(self, message: ToolResultMessage) -> str:
        if message.content:
            parts = [part.text for part in message.content if isinstance(part, TextContent)]
            return self._truncate_tool_output("".join(parts))

        return ""

    def _render_session_entries(self, session: Session) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        started_tools: set[str] = set()

        for entry in session.entries:
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
                    if ui_summary is None and ui_details is None:
                        ui_details = self._format_tool_result_text(message)
                        markup = False

                    chat.set_tool_result(
                        tool_id, ui_summary, ui_details, not message.is_error, markup=markup
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
                elif entry.display:
                    chat.add_info_message(entry.content)

    @staticmethod
    def _calculate_session_tokens(session: Session) -> tuple[int, int, int, int, int]:
        """
        Calculate cumulative token usage from session entries.

        Returns
        -------
        (input_tokens, output_tokens, context_tokens, cache_read_tokens, cache_write_tokens)
        """
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        context_tokens = 0

        for entry in session.entries:
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage:
                    input_tokens += usage.input_tokens
                    output_tokens += usage.output_tokens
                    cache_read_tokens += usage.cache_read_tokens
                    cache_write_tokens += usage.cache_write_tokens
                    context_tokens = (
                        usage.input_tokens
                        + usage.output_tokens
                        + usage.cache_read_tokens
                        + usage.cache_write_tokens
                    )

        return input_tokens, output_tokens, context_tokens, cache_read_tokens, cache_write_tokens

    @staticmethod
    def _calculate_session_file_changes(session: Session) -> dict[str, tuple[int, int]]:
        file_changes: dict[str, tuple[int, int]] = {}
        for entry in session.entries:
            if isinstance(entry, MessageEntry) and isinstance(entry.message, ToolResultMessage):
                fc = entry.message.file_changes
                if fc:
                    prev_added, prev_removed = file_changes.get(fc.path, (0, 0))
                    file_changes[fc.path] = (prev_added + fc.added, prev_removed + fc.removed)
        return file_changes

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
            session = Session.load(session_path)
        except Exception as exc:
            chat.add_info_message(f"Failed to load session: {exc}", error=True)
            input_box.focus()
            return

        self._session = session
        self._current_block_type = None

        status.reset()
        input_t, output_t, context_t, cache_read_t, cache_write_t = self._calculate_session_tokens(
            session
        )
        info_bar.set_tokens(input_t, output_t, context_t, cache_read_t, cache_write_t)
        info_bar.set_file_changes(self._calculate_session_file_changes(session))
        info_bar.set_session_id(session.id[:8])

        model_info = session.model
        if model_info:
            provider, model_id, session_base_url = model_info
            self._model = model_id

            self._model_provider = provider
            restored_model = get_model(model_id, provider)
            restored_base_url = session_base_url or (
                restored_model.base_url if restored_model else None
            )
            if restored_model and self._provider:
                current_api_type = self._get_provider_api_type(self._provider)
                if restored_model.api != current_api_type:
                    provider_config = ProviderConfig(
                        api_key=self._api_key,
                        base_url=restored_base_url,
                        model=model_id,
                        max_tokens=get_max_tokens(model_id),
                        thinking_level=self._thinking_level,
                        provider=provider,
                        session_id=session.id,
                        openai_compat_auth_mode=self._openai_compat_auth_mode,
                        anthropic_compat_auth_mode=self._anthropic_compat_auth_mode,
                    )
                    try:
                        self._provider = self._create_provider(restored_model.api, provider_config)
                    except ValueError as e:
                        chat.add_info_message(str(e), error=True)
                else:
                    self._provider.config.model = model_id
                    self._provider.config.base_url = restored_base_url
                    self._provider.config.session_id = session.id
            elif self._provider:
                self._provider.config.model = model_id
                if restored_base_url:
                    self._provider.config.base_url = restored_base_url
                self._provider.config.session_id = session.id

            info_bar.set_model(model_id, provider)

        thinking_level = session.thinking_level
        if self._provider:
            valid_levels = self._provider.thinking_levels
            if valid_levels and thinking_level not in valid_levels:
                thinking_level = valid_levels[0]
            self._provider.set_thinking_level(thinking_level)
        self._thinking_level = thinking_level
        info_bar.set_thinking_level(thinking_level)
        self._apply_thinking_level_style(thinking_level)

        if self._provider is not None:
            system_prompt = session.system_prompt or build_system_prompt(
                self._cwd, tools=self._tools
            )
            self._agent = Agent(
                provider=self._provider,
                tools=self._tools,
                session=session,
                cwd=self._cwd,
                system_prompt=system_prompt,
            )
        elif self._agent is not None:
            self._agent.session = session

        await chat.remove_all_children()

        chat.add_session_info(getattr(self, "VERSION", ""))

        if self._agent:
            chat.add_loaded_resources(
                context_paths=[format_path(f.path) for f in self._agent.context.agents_files],
                skill_paths=[format_path(s.path) for s in self._agent.context.skills],
            )

        self._render_session_entries(session)
        chat.add_info_message("Resumed session")
        input_box.focus()
