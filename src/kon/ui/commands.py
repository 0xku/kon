from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kon import config, set_theme

from ..core.compaction import generate_summary
from ..core.handoff import generate_handoff_prompt
from ..core.types import AssistantMessage, ToolCall, ToolResultMessage
from ..llm import (
    ApiType,
    BaseProvider,
    ProviderConfig,
    clear_openai_credentials,
    get_all_models,
    get_max_tokens,
    get_model,
    is_copilot_logged_in,
    is_openai_logged_in,
    openai_login,
)
from ..loop import build_system_prompt
from ..session import MessageEntry, Session
from ..themes import get_theme_options
from .chat import ChatLog
from .clipboard import copy_to_clipboard
from .floating_list import FloatingList, ListItem
from .input import InputBox
from .selection_mode import SelectionMode
from .widgets import InfoBar, StatusLine, format_path

if TYPE_CHECKING:
    pass


class CommandsMixin:
    HANDOFF_BACKLINK_TYPE = "handoff_backlink"
    HANDOFF_FORWARD_LINK_TYPE = "handoff_forward_link"

    # Attributes provided by the App subclass
    _cwd: str
    _thinking_level: str
    _model: str
    _model_provider: str | None
    _api_key: str | None
    _provider: BaseProvider | None
    _session: Session | None
    _agent: Any
    _is_running: bool
    _selection_mode: Any
    _tools: list

    # Methods from App - declared for type checking
    if TYPE_CHECKING:
        exit: Any
        notify: Any
        query_one: Any
        run_worker: Any
        call_later: Any

    # Methods from other mixins/main class
    if TYPE_CHECKING:

        def _get_provider_api_type(self, provider: BaseProvider) -> ApiType: ...
        def _create_provider(self, api_type: ApiType, config: ProviderConfig) -> BaseProvider: ...
        def _sync_slash_commands(self) -> None: ...
        def _render_session_entries(self, session: Session) -> None: ...
        def _apply_theme(self, theme_id: str) -> None: ...

    def _handle_command(self, text: str) -> bool:
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            self.exit()
            return True
        if cmd == "help":
            self._show_help()
            return True
        if cmd == "clear":
            self._clear_conversation()
            return True
        if cmd == "model":
            self._handle_model_command(args)
            return True
        if cmd == "new":
            self._new_conversation()
            return True
        if cmd == "themes":
            self._handle_themes_command(args)
            return True
        if cmd == "handoff":
            self._handle_handoff_command(args)
            return True
        if cmd == "resume":
            self._show_resume_sessions()
            return True
        if cmd == "session":
            self._show_session_info()
            return True
        if cmd == "login":
            self._handle_login_command(args)
            return True
        if cmd == "logout":
            self._handle_logout_command(args)
            return True
        if cmd == "export":
            self._handle_export_command()
            return True
        if cmd == "copy":
            self._handle_copy_command()
            return True
        if cmd == "compact":
            self._handle_compact_command()
            return True

        return False

    def _show_help(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        help_text = """Commands:
  /help      - Show this help
  /quit      - Quit (or ctrl+c twice)
  /clear     - Clear conversation history
  /compact   - Compact current conversation now
  /model     - Change model (/model gpt-4o)
  /themes    - Change UI theme (/themes gruvbox-dark)
  /new       - Start new conversation
  /handoff   - Start focused handoff in new session
  /resume    - Resume a session
  /session   - Show session info and stats
  /login     - Login to a provider
  /logout    - Logout from a provider
  /export    - Export session to HTML file
  /copy      - Copy last agent response text to clipboard

Keybindings:
  @          - File path search (inline)
  /          - Slash commands (at start of input)
  escape     - Cancel completion / interrupt agent
  ctrl+c     - Clear input (press twice to quit)
  ctrl+t     - Toggle thinking visibility
  shift+tab  - Cycle thinking levels

Extra tools:
  --extra-tools web_search,web_fetch  or  [tools] extra in ~/.kon/config.toml"""
        chat.add_info_message(help_text)

    def _clear_conversation(self) -> None:
        if self._session:
            self._session = self._create_new_session()
            if self._provider:
                self._provider.config.session_id = self._session.id
            info_bar = self.query_one("#info-bar", InfoBar)
            info_bar.set_session_id(self._session.id[:8])
            info_bar.set_tokens(0, 0, 0, 0)
            info_bar.set_file_changes({})
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info_message("Conversation cleared")

    def _handle_model_command(self, args: str) -> None:
        models = get_all_models()
        if not models:
            self.notify("No models configured", title="Models", timeout=3, severity="warning")
            return

        models.sort(key=lambda m: (m.provider, m.id))

        items: list[ListItem] = []
        for m in models:
            parts = [m.provider]
            if not m.supports_images:
                parts.append("[no-vision]")
            if m.id == self._model and m.provider == self._model_provider:
                parts.append("✓")
            caption = " ".join(parts)
            items.append(ListItem(value=m, label=m.id, description=caption))

        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.show(items, searchable=True)

        input_box = self.query_one("#input-box", InputBox)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        input_box.focus()
        self._selection_mode = SelectionMode.MODEL

    def _handle_themes_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        requested = args.strip()
        if requested:
            try:
                self._select_theme(requested)
            except ValueError as e:
                chat.add_info_message(str(e), error=True)
            return

        current_theme = config.ui.theme
        items = [
            ListItem(
                value=theme_id,
                label=label,
                description=f"{theme_id} ✓" if theme_id == current_theme else theme_id,
            )
            for theme_id, label in get_theme_options()
        ]

        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.show(items, searchable=True)

        input_box = self.query_one("#input-box", InputBox)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        input_box.focus()
        self._selection_mode = SelectionMode.THEME

    def _select_theme(self, theme_id: str) -> None:
        set_theme(theme_id)
        self._apply_theme(theme_id)
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info_message(
            f"Theme changed to {theme_id}. Full theme refresh applies when kon is restarted.",
            warning=True,
        )

    def _select_model(self, model) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)

        current_api_type = self._get_provider_api_type(self._provider) if self._provider else None

        self._model = model.id
        self._model_provider = model.provider

        if model.api != current_api_type:
            provider_config = ProviderConfig(
                api_key=self._api_key,
                base_url=model.base_url,
                model=model.id,
                max_tokens=get_max_tokens(model.id),
                thinking_level=self._thinking_level,
                provider=model.provider,
                session_id=self._session.id if self._session else None,
                openai_compat_auth_mode=self._openai_compat_auth_mode,
                anthropic_compat_auth_mode=self._anthropic_compat_auth_mode,
            )
            try:
                self._provider = self._create_provider(model.api, provider_config)
            except ValueError as e:
                chat.add_info_message(str(e), error=True)
                return
        elif self._provider:
            self._provider.config.model = model.id
            self._provider.config.base_url = model.base_url

        info_bar.set_model(model.id, model.provider)

        if self._session:
            self._session.set_model(model.provider, model.id, model.base_url)

        chat.add_info_message(f"Model changed to {model.id} ({model.provider})")

    def _new_conversation(self) -> None:
        self._session = self._create_new_session()
        if self._provider:
            self._provider.config.session_id = self._session.id

        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)
        status = self.query_one("#status-line", StatusLine)

        self.run_worker(self._do_new_conversation(chat, info_bar, status), exclusive=False)

    async def _do_new_conversation(self, chat: ChatLog, info_bar, status) -> None:
        await self._reset_session_ui(chat, info_bar, status)
        chat.add_info_message("Started new conversation")

    async def _reset_session_ui(self, chat: ChatLog, info_bar, status) -> None:
        await chat.remove_all_children()

        status.reset()

        info_bar.set_tokens(0, 0, 0, 0)
        info_bar.set_file_changes({})
        if self._session:
            info_bar.set_session_id(self._session.id[:8])
        info_bar.set_thinking_level(self._thinking_level)

        chat.add_session_info(getattr(self, "VERSION", ""))

        if self._agent is not None:
            self._agent.reload_context()
            self._agent.session = self._session
            self._sync_slash_commands()
            # TODO: Surface self._agent.context.skill_warnings in UI
            chat.add_loaded_resources(
                context_paths=[format_path(f.path) for f in self._agent.context.agents_files],
                skill_paths=[format_path(s.path) for s in self._agent.context.skills],
            )

    def _handle_handoff_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if self._is_running:
            chat.add_info_message("Cannot handoff while agent is running", error=True)
            return

        if self._provider is None or self._session is None or self._agent is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        query = args.strip()
        if not query:
            chat.add_info_message(
                "Usage: /handoff <query>. Example: /handoff implement phase two", error=True
            )
            return

        if not self._session.all_messages:
            chat.add_info_message("No conversation to handoff", error=True)
            return

        chat.show_spinner_status("Creating handoff...")
        self.run_worker(self._do_handoff(query), exclusive=False)

    def _create_new_session(self) -> Session:
        selected_model = get_model(self._model, self._model_provider)
        model_provider = (
            selected_model.provider
            if selected_model
            else (self._provider.name if self._provider else "openai")
        )
        self._model_provider = model_provider
        system_prompt = build_system_prompt(self._cwd, tools=self._tools)
        session = Session.create(
            self._cwd,
            provider=model_provider,
            model_id=self._model,
            thinking_level=self._thinking_level,
            system_prompt=system_prompt,
        )
        model_base_url = (
            selected_model.base_url
            if selected_model
            else (self._provider.config.base_url if self._provider else None)
        )
        session.append_model_change(model_provider, self._model, model_base_url)
        return session

    async def _do_handoff(self, query: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)
        status = self.query_one("#status-line", StatusLine)
        input_box = self.query_one("#input-box", InputBox)

        if self._provider is None or self._session is None or self._agent is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        source_session = self._session

        try:
            prompt = await generate_handoff_prompt(
                source_session.all_messages,
                self._provider,
                system_prompt=self._agent.system_prompt,
                query=query,
            )
        except Exception as e:
            chat.show_status("Handoff failed")
            chat.add_info_message(f"Handoff failed: {e}", error=True)
            return

        source_session_id = source_session.id
        new_session = self._create_new_session()

        new_session.append_custom_message(
            self.HANDOFF_BACKLINK_TYPE,
            f"Handoff from {source_session_id[:8]}",
            display=False,
            details={"target_session_id": source_session_id, "query": query},
        )
        source_session.append_custom_message(
            self.HANDOFF_FORWARD_LINK_TYPE,
            f"Handoff to {new_session.id[:8]}",
            display=False,
            details={"target_session_id": new_session.id, "query": query},
        )

        new_session.ensure_persisted()
        source_session.ensure_persisted()

        self._session = new_session
        if self._provider:
            self._provider.config.session_id = self._session.id
        if self._agent is not None:
            self._agent.session = self._session

        await self._reset_session_ui(chat, info_bar, status)
        self._render_session_entries(self._session)

        input_box.clear()
        input_box.insert(prompt)
        chat.show_status("Handoff ready")
        input_box.focus()

    def _show_session_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        if not self._session:
            chat.add_info_message("No active session")
            return

        session_file = self._session.session_file
        file_path = str(session_file) if session_file else "(in-memory session)"

        full_id = self._session._header.id if self._session._header else self._session.id

        user_count = 0
        assistant_count = 0
        tool_call_count = 0
        tool_result_count = 0

        for entry in self._session.entries:
            if isinstance(entry, MessageEntry):
                message = entry.message
                if message.role == "user":
                    user_count += 1
                elif message.role == "assistant":
                    assistant_count += 1
                    for part in message.content:
                        if isinstance(part, ToolCall):
                            tool_call_count += 1

        tool_result_count = sum(
            1
            for e in self._session.entries
            if isinstance(e, MessageEntry) and isinstance(e.message, ToolResultMessage)
        )

        total_messages = user_count + assistant_count

        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        for entry in self._session.entries:
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage:
                    input_tokens += usage.input_tokens
                    output_tokens += usage.output_tokens
                    cache_read_tokens += usage.cache_read_tokens
                    cache_write_tokens += usage.cache_write_tokens

        total_tokens = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens

        lines = [
            "Session Info",
            "",
            "File:",
            f"{file_path}",
            f"ID: {full_id}",
            "",
            "Messages",
            f"User: {user_count}",
            f"Assistant: {assistant_count}",
            f"Tool Calls: {tool_call_count}",
            f"Tool Results: {tool_result_count}",
            f"Total: {total_messages}",
            "",
            "Tokens",
            f"Input: {input_tokens:,}",
            f"Output: {output_tokens:,}",
            f"Cache read: {cache_read_tokens:,}",
            f"Cache write: {cache_write_tokens:,}",
            f"Total: {total_tokens:,}",
        ]

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info_message("\n".join(lines))

    def _build_resume_items(self) -> list[ListItem]:
        sessions = Session.list(self._cwd)
        items: list[ListItem] = []
        for session in sessions:
            label = self._format_session_label(session.first_message)
            caption = f"{self._format_session_age(session.modified)} {session.message_count}"
            items.append(ListItem(value=session, label=label, description=caption))
        return items

    def _show_resume_sessions(self) -> None:
        items = self._build_resume_items()
        if not items:
            self.notify(
                "No saved sessions found", title="Sessions", timeout=3, severity="information"
            )
            return

        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.show(items, searchable=True)

        input_box = self.query_one("#input-box", InputBox)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        input_box.focus()
        self._selection_mode = SelectionMode.SESSION

    def _delete_selected_resume_session(self) -> None:
        if self._selection_mode != SelectionMode.SESSION:
            return

        completion_list = self.query_one("#completion-list", FloatingList)
        selected_item = completion_list.selected_item
        if selected_item is None:
            return

        session_info = selected_item.value
        session_path = Path(session_info.path)

        current_session_path: Path | None = None
        if self._session and self._session.session_file is not None:
            current_session_path = Path(self._session.session_file)

        if current_session_path is not None and session_path == current_session_path:
            self.notify(
                "Cannot delete current session", title="Sessions", timeout=2, severity="warning"
            )
            return

        try:
            session_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.notify(
                f"Failed to delete session: {exc}", title="Sessions", timeout=3, severity="error"
            )
            return

        items = self._build_resume_items()
        if not items:
            completion_list.hide()
            input_box = self.query_one("#input-box", InputBox)
            input_box.set_autocomplete_enabled(True)
            input_box.set_completing(False)
            self._selection_mode = None
            self.notify(
                "Session deleted (no saved sessions left)",
                title="Sessions",
                timeout=2,
                severity="information",
            )
            return

        completion_list.update_items(items)
        self.notify("Session deleted", title="Sessions", timeout=2, severity="information")

    def _handle_login_command(self, args: str) -> None:
        providers = [
            {
                "id": "github-copilot",
                "name": "GitHub Copilot",
                "logged_in": is_copilot_logged_in(),
            },
            {"id": "openai", "name": "OpenAI (ChatGPT/Codex)", "logged_in": is_openai_logged_in()},
        ]

        items: list[ListItem] = []
        for p in providers:
            status = "✓ logged in" if p["logged_in"] else ""
            items.append(ListItem(value=p["id"], label=p["name"], description=status))

        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.show(items, searchable=True)

        input_box = self.query_one("#input-box", InputBox)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        input_box.focus()
        self._selection_mode = SelectionMode.LOGIN

    def _select_login_provider(self, provider_id: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if provider_id == "github-copilot":
            if is_copilot_logged_in():
                chat.add_info_message("Already logged in to GitHub Copilot")
                return

            chat.add_info_message("Starting GitHub Copilot login...")
            self.run_worker(self._copilot_login_flow(), exclusive=False)
            return

        if provider_id == "openai":
            if is_openai_logged_in():
                chat.add_info_message("Already logged in to OpenAI")
                return

            chat.add_info_message("Starting OpenAI login...")
            self.run_worker(self._openai_login_flow(), exclusive=False)

    async def _copilot_login_flow(self) -> None:
        import webbrowser

        from kon.llm import copilot_login

        chat = self.query_one("#chat-log", ChatLog)

        def on_user_code(url: str, code: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                f"Opening browser to: {url}\n"
                f"Enter this code: {code}\n\n"
                "Waiting for authorization...",
            )

        try:
            await copilot_login(on_user_code=on_user_code)
            chat.add_info_message(
                "Successfully logged in to GitHub Copilot!\n"
                "You can now use /model to select Copilot models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    async def _openai_login_flow(self) -> None:
        import webbrowser

        chat = self.query_one("#chat-log", ChatLog)

        def on_auth_url(url: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                "Opening browser for OpenAI OAuth...\n"
                f"If browser does not open, visit:\n{url}\n\n"
                "Waiting for authorization callback on http://localhost:1455/auth/callback ...",
            )

        try:
            await openai_login(on_auth_url=on_auth_url)
            chat.add_info_message(
                "Successfully logged in to OpenAI!\n"
                "You can now use /model to select openai-codex models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    def _handle_logout_command(self, args: str) -> None:
        providers = []
        if is_copilot_logged_in():
            providers.append({"id": "github-copilot", "name": "GitHub Copilot"})
        if is_openai_logged_in():
            providers.append({"id": "openai", "name": "OpenAI (ChatGPT/Codex)"})

        if not providers:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info_message("No providers logged in")
            return

        items: list[ListItem] = []
        for p in providers:
            items.append(ListItem(value=p["id"], label=p["name"], description=""))

        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.show(items, searchable=True)

        input_box = self.query_one("#input-box", InputBox)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        input_box.focus()
        self._selection_mode = SelectionMode.LOGOUT

    def _select_logout_provider(self, provider_id: str) -> None:
        from kon.llm import clear_copilot_credentials

        chat = self.query_one("#chat-log", ChatLog)

        if provider_id == "github-copilot":
            clear_copilot_credentials()
            chat.add_info_message("Logged out of GitHub Copilot")
            return

        if provider_id == "openai":
            clear_openai_credentials()
            chat.add_info_message("Logged out of OpenAI")

    def _handle_export_command(self) -> None:
        from .export import export_session_html

        chat = self.query_one("#chat-log", ChatLog)

        if not self._session:
            chat.add_info_message("No active session to export")
            return

        if not self._session.entries:
            chat.add_info_message("Session has no messages to export")
            return

        system_prompt = self._agent.system_prompt
        tools = self._tools

        provider_name = self._provider.name if self._provider else "unknown"

        try:
            path = export_session_html(
                session=self._session,
                system_prompt=system_prompt,
                tools=tools,
                output_dir=self._cwd,
                model_id=self._model,
                provider=provider_name,
                version=getattr(self, "VERSION", ""),
                title_color=config.ui.colors.title,
            )
            chat.add_info_message(f"Session exported to {path.name}")
        except Exception as e:
            chat.add_info_message(f"Export failed: {e}", error=True)

    def _handle_copy_command(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if not self._session:
            chat.add_info_message("No agent messages to copy yet", error=True)
            return

        text = self._session.get_last_assistant_text()
        if not text:
            chat.add_info_message("No agent messages to copy yet", error=True)
            return

        copy_to_clipboard(text)
        chat.show_status("Copied last agent message to clipboard")

    def _handle_compact_command(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if self._is_running:
            chat.add_info_message("Cannot compact while agent is running", error=True)
            return

        if self._provider is None or self._session is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        if not self._session.all_messages:
            chat.add_info_message("No conversation to compact")
            return

        chat.show_spinner_status("Compacting...")
        self.run_worker(self._do_compact(), exclusive=False)

    async def _do_compact(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if self._provider is None or self._session is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        tokens_before = 0
        for entry in reversed(self._session.entries):
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage is None:
                    continue
                tokens_before = (
                    usage.input_tokens
                    + usage.output_tokens
                    + usage.cache_read_tokens
                    + usage.cache_write_tokens
                )
                break

        try:
            summary = await generate_summary(
                self._session.all_messages, self._provider, system_prompt=self._agent.system_prompt
            )
            self._session.append_compaction(
                summary=summary,
                first_kept_entry_id=self._session.leaf_id or "",
                tokens_before=tokens_before,
            )
            chat.add_compaction_message(tokens_before)
        except Exception as e:
            chat.show_status("Compaction failed")
            chat.add_info_message(f"Compaction failed: {e}", error=True)

    def _format_session_label(self, message: str, width: int = 72) -> str:
        normalized = " ".join(message.split())
        if len(normalized) > width:
            normalized = normalized[: width - 3].rstrip() + "..."
        return normalized.ljust(width)

    def _format_session_age(self, modified: datetime) -> str:
        now = datetime.now(UTC)
        delta = max(0, int((now - modified).total_seconds()))
        minutes = delta // 60
        hours = delta // 3600
        days = delta // 86400
        weeks = days // 7

        if minutes < 60:
            value, unit = minutes, "m"
        elif hours < 24:
            value, unit = hours, "h"
        elif days < 7:
            value, unit = days, "d"
        elif weeks < 52:
            value, unit = weeks, "w"
        else:
            value, unit = weeks // 52, "y"

        return f"{value:>2}{unit}"
