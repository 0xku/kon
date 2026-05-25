from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from kon import (
    config,
    set_notifications_enabled,
    set_permissions_mode,
    set_show_welcome_shortcuts,
    set_theme,
)
from kon.config import NOTIFICATION_MODES, PERMISSION_MODES, NotificationMode, PermissionMode

from ..llm import (
    clear_openai_credentials,
    copilot_login,
    get_all_models,
    get_copilot_token,
    get_valid_openai_credentials,
    openai_login,
)
from ..llm import is_copilot_logged_in as has_saved_copilot_credentials
from ..llm import is_openai_logged_in as has_saved_openai_credentials
from ..runtime import ConversationRuntime
from ..session import Session, SessionInfo
from ..themes import get_theme_options
from .chat import ChatLog
from .clipboard import copy_to_clipboard
from .floating_list import FloatingList, ListItem
from .input import InputBox
from .selection_mode import SelectionMode
from .tree import TreeSelector
from .widgets import InfoBar, StatusLine, format_path

if TYPE_CHECKING:
    pass


Choice = TypeVar("Choice", bound=str)
SettingsSelectionResult = Literal["reopened-picker", "closed"]


class CommandsMixin:
    HANDOFF_BACKLINK_TYPE = "handoff_backlink"
    HANDOFF_FORWARD_LINK_TYPE = "handoff_forward_link"

    _cwd: str
    _api_key: str | None
    _agent: Any
    _is_running: bool
    _selection_mode: Any
    _tools: list
    _openai_compat_auth_mode: Any
    _anthropic_compat_auth_mode: Any
    _runtime: ConversationRuntime

    # Methods from App - declared for type checking
    if TYPE_CHECKING:
        exit: Any
        notify: Any
        query_one: Any
        run_worker: Any
        call_later: Any
        _settings_active: bool
        _settings_selected_value: str | None

    # Methods from other mixins/main class
    if TYPE_CHECKING:

        def _sync_runtime_state(self) -> None: ...
        def _sync_slash_commands(self) -> None: ...
        def _render_session_entries(self, session: Session) -> None: ...
        def _apply_theme(self, theme_id: str) -> None: ...
        def _apply_thinking_level_style(self, level: str) -> None: ...
        def _show_completion_list(
            self,
            items: list[ListItem],
            *,
            searchable: bool = False,
            max_label_width: int | None = None,
        ) -> None: ...
        def _hide_completion_list(self, *, restore_info_bar: bool = True) -> None: ...
        def _is_chat_at_bottom(self) -> bool: ...
        def _restore_chat_scroll_after_refresh(self, was_at_bottom: bool) -> None: ...

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
        if cmd == "settings":
            self._handle_settings_command()
            return True
        if cmd == "themes":
            self._handle_themes_command(args)
            return True
        if cmd == "permissions":
            self._handle_permissions_command(args)
            return True
        if cmd == "thinking":
            self._handle_thinking_command(args)
            return True
        if cmd == "notifications":
            self._handle_notifications_command(args)
            return True
        if cmd == "handoff":
            self._handle_handoff_command(args)
            return True
        if cmd == "resume":
            self._show_resume_sessions()
            return True
        if cmd == "tree":
            self._show_tree_selector()
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
        chat.add_help_details()

    def _clear_conversation(self) -> None:
        if self._runtime.session:
            self._runtime.new_session()
            self._sync_runtime_state()
            info_bar = self.query_one("#info-bar", InfoBar)
            info_bar.set_tokens(0, 0, 0, 0)
            info_bar.set_file_changes({})
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info_message("Conversation cleared")

    def _show_selection_picker(
        self,
        items: list[ListItem],
        selection_mode: SelectionMode,
        *,
        searchable: bool = True,
        max_label_width: int | None = None,
    ) -> None:
        input_box = self.query_one("#input-box", InputBox)
        was_at_bottom = self._is_chat_at_bottom()

        with self.batch_update():  # type: ignore[attr-defined]
            self._show_completion_list(
                items, searchable=searchable, max_label_width=max_label_width
            )
            input_box.clear()
            input_box.set_autocomplete_enabled(False)
            input_box.set_completing(True)
            input_box.focus()

        self._selection_mode = selection_mode
        self._restore_chat_scroll_after_refresh(was_at_bottom)

    def _build_choice_items(
        self, choices: Sequence[Choice], current: Choice, descriptions: Mapping[Choice, str]
    ) -> list[ListItem[Choice]]:
        return [
            ListItem(
                value=choice,
                label=f"{choice} ✓" if choice == current else choice,
                description=descriptions[choice],
            )
            for choice in choices
        ]

    def _handle_choice_command(
        self,
        args: str,
        *,
        name: str,
        choices: Sequence[Choice],
        current: Choice,
        descriptions: Mapping[Choice, str],
        selection_mode: SelectionMode,
        select: Callable[[Choice], None],
    ) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        requested = args.strip()
        if requested:
            if requested in choices:
                select(cast(Choice, requested))
            else:
                valid = ", ".join(choices)
                chat.add_info_message(
                    f"Invalid {name} mode: {requested}. Use one of: {valid}", error=True
                )
            return

        self._show_selection_picker(
            self._build_choice_items(choices, current, descriptions), selection_mode
        )

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
            caption = " ".join(parts)
            label = (
                f"{m.id} ✓"
                if m.id == self._runtime.model and m.provider == self._runtime.model_provider
                else m.id
            )
            items.append(ListItem(value=m, label=label, description=caption))

        self._show_selection_picker(items, SelectionMode.MODEL)

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
            ListItem(value=theme_id, label=f"{label} ✓" if theme_id == current_theme else label)
            for theme_id, label in get_theme_options()
        ]

        self._show_selection_picker(items, SelectionMode.THEME)

    def _handle_permissions_command(self, args: str) -> None:
        descriptions: dict[PermissionMode, str] = {
            "prompt": "ask before mutating tool calls",
            "auto": "allow tool calls without approval prompts",
        }
        self._handle_choice_command(
            args,
            name="permission",
            choices=PERMISSION_MODES,
            current=config.permissions.mode,
            descriptions=descriptions,
            selection_mode=SelectionMode.PERMISSIONS,
            select=self._select_permission_mode,
        )

    def _select_permission_mode(self, mode: PermissionMode) -> None:
        set_permissions_mode(mode)
        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.set_permission_mode(mode)
        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Permission mode changed to {mode}")

    def _handle_thinking_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        if self._runtime.provider is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        requested = args.strip()
        if requested:
            if requested in self._runtime.provider.thinking_levels:
                self._select_thinking_level(requested)
            else:
                valid_levels = ", ".join(self._runtime.provider.thinking_levels)
                chat.add_info_message(
                    f"Invalid thinking level: {requested}. Use one of: {valid_levels}", error=True
                )
            return

        items = [
            ListItem(
                value=level, label=f"{level} ✓" if level == self._runtime.thinking_level else level
            )
            for level in self._runtime.provider.thinking_levels
        ]
        self._show_selection_picker(items, SelectionMode.THINKING)

    def _select_thinking_level(self, level: str) -> None:
        if self._runtime.provider is None:
            return

        self._runtime.set_thinking_level(level)
        self._sync_runtime_state()

        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.set_thinking_level(level)
        self._apply_thinking_level_style(level)

        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Thinking level changed to {level}")

    def _handle_notifications_command(self, args: str) -> None:
        current: NotificationMode = "on" if config.notifications.enabled else "off"
        descriptions: dict[NotificationMode, str] = {
            "on": "play notification sounds",
            "off": "disable notification sounds",
        }
        self._handle_choice_command(
            args,
            name="notifications",
            choices=NOTIFICATION_MODES,
            current=current,
            descriptions=descriptions,
            selection_mode=SelectionMode.NOTIFICATIONS,
            select=self._select_notifications_mode,
        )

    def _select_notifications_mode(self, mode: NotificationMode) -> None:
        set_notifications_enabled(mode == "on")
        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Notifications turned {mode}")

    # -------------------------------------------------------------------------
    # Settings (unified panel for themes, permissions, notifications, thinking)
    # -------------------------------------------------------------------------

    def _build_settings_items(self) -> list[ListItem[str]]:
        notification_status = "on" if config.notifications.enabled else "off"
        try:
            thinking_level = self._runtime.thinking_level or "off"
        except Exception:
            thinking_level = "off"

        shortcut_status = "on" if config.ui.show_welcome_shortcuts else "off"
        return [
            ListItem(
                value="notifications", label="notifications", description=notification_status
            ),
            ListItem(value="show-shortcuts", label="show-shortcuts", description=shortcut_status),
            ListItem(
                value="permissions", label="permissions", description=config.permissions.mode
            ),
            ListItem(value="themes", label="themes", description=config.ui.theme),
            ListItem(value="thinking", label="thinking", description=thinking_level),
        ]

    def _show_settings_picker(self, selected_value: str | None = None) -> None:
        items = self._build_settings_items()
        self._show_selection_picker(items, SelectionMode.SETTINGS, max_label_width=40)
        self._settings_selected_value = selected_value
        if selected_value is not None:
            completion_list = self.query_one("#completion-list", FloatingList)
            completion_list.select_value(selected_value)

    def _handle_settings_command(self) -> None:
        self._show_settings_picker()

    def _handle_settings_select(self, item_value: str) -> SettingsSelectionResult:
        if item_value == "notifications":
            current_enabled = config.notifications.enabled
            set_notifications_enabled(not current_enabled)
            mode: NotificationMode = "on" if not current_enabled else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Notifications turned {mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "show-shortcuts":
            shortcuts_current = config.ui.show_welcome_shortcuts
            set_show_welcome_shortcuts(not shortcuts_current)
            mode = "on" if not shortcuts_current else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Welcome shortcuts turned {mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "permissions":
            current: PermissionMode = config.permissions.mode
            new_mode: PermissionMode = "auto" if current == "prompt" else "prompt"
            set_permissions_mode(new_mode)
            info_bar = self.query_one("#info-bar", InfoBar)
            info_bar.set_permission_mode(new_mode)
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Permission mode changed to {new_mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "themes":
            self._settings_active = True
            self._handle_themes_command("")
            return "reopened-picker"

        elif item_value == "thinking":
            if self._runtime.provider is None:
                self._handle_thinking_command("")
                return "closed"
            self._settings_active = True
            self._handle_thinking_command("")
            return "reopened-picker"

        return "closed"

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

        try:
            self._runtime.switch_model(model)
        except ValueError as e:
            chat.add_info_message(str(e), error=True)
            return
        self._sync_runtime_state()

        info_bar.set_model(model.id, model.provider)

        chat.add_info_message(f"Model changed to {model.id} ({model.provider})")

    def _new_conversation(self) -> None:
        self._runtime.new_session()
        self._sync_runtime_state()

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
        info_bar.set_thinking_level(self._runtime.thinking_level)

        chat.add_session_info(getattr(self, "VERSION", ""))

        self._runtime.reload_context()
        self._sync_runtime_state()
        if self._runtime.agent is not None:
            self._sync_slash_commands()
            # TODO: Surface self._runtime.agent.context.skill_warnings in UI
            chat.add_loaded_resources(
                context_paths=[
                    format_path(f.path) for f in self._runtime.agent.context.agents_files
                ],
                skills=self._runtime.agent.context.skills,
                tools=self._runtime.tools,
            )

    def _handle_handoff_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if self._is_running:
            chat.add_info_message("Cannot handoff while agent is running", error=True)
            return

        if (
            self._runtime.provider is None
            or self._runtime.session is None
            or self._runtime.agent is None
        ):
            chat.add_info_message("Agent not initialized", error=True)
            return

        query = args.strip()
        if not query:
            chat.add_info_message(
                "Usage: /handoff <query>. Example: /handoff implement phase two", error=True
            )
            return

        if not self._runtime.session.all_messages:
            chat.add_info_message("No conversation to handoff", error=True)
            return

        chat.show_spinner_status("Creating handoff...")
        self.run_worker(self._do_handoff(query), exclusive=False)

    def _resolve_system_prompt(self, session: Session | None = None) -> str:
        return self._runtime.resolve_system_prompt(session)

    def _create_new_session(self) -> Session:
        return self._runtime.create_session()

    async def _do_handoff(self, query: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)
        status = self.query_one("#status-line", StatusLine)
        input_box = self.query_one("#input-box", InputBox)

        if (
            self._runtime.provider is None
            or self._runtime.session is None
            or self._runtime.agent is None
        ):
            chat.add_info_message("Agent not initialized", error=True)
            return

        try:
            result = await self._runtime.create_handoff(query)
        except Exception as e:
            chat.show_status("Handoff failed")
            chat.add_info_message(f"Handoff failed: {e}", error=True)
            return

        self._sync_runtime_state()
        await self._reset_session_ui(chat, info_bar, status)
        self._render_session_entries(result.new_session)

        input_box.clear()
        input_box.insert(result.prompt)
        chat.show_status("Handoff ready")
        input_box.focus()

    def _show_session_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        if not self._runtime.session:
            chat.add_info_message("No active session")
            return

        session_path = self._runtime.session.session_file
        session_dir = str(session_path.parent) if session_path else None
        session_file = session_path.name if session_path else "(in-memory session)"

        counts = self._runtime.session.message_counts()
        token_totals = self._runtime.session.token_totals()

        chat.add_session_details(
            session_dir=session_dir,
            session_file=session_file,
            user_messages=counts.user_messages,
            assistant_messages=counts.assistant_messages,
            tool_calls=counts.tool_calls,
            tool_results=counts.tool_results,
            total_messages=counts.total_messages,
            input_tokens=token_totals.input_tokens,
            output_tokens=token_totals.output_tokens,
            cache_read_tokens=token_totals.cache_read_tokens,
            cache_write_tokens=token_totals.cache_write_tokens,
            total_tokens=token_totals.total_tokens,
        )

    def _build_resume_items(self) -> list[ListItem]:
        sessions = Session.list(self._cwd)

        # Build tree structure from handoff relationships
        by_id: dict[str, SessionInfo] = {s.id: s for s in sessions}
        children: dict[str, list[SessionInfo]] = {}
        roots: list[SessionInfo] = []

        for session in sessions:
            pid = session.parent_session_id
            if pid and pid in by_id:
                children.setdefault(pid, []).append(session)
            else:
                roots.append(session)

        # Sort children within each parent by modified time (newest first,
        # matching the root-level sort from Session.list)
        for kids in children.values():
            kids.sort(key=lambda s: s.modified, reverse=True)

        # DFS flatten: roots are already sorted by modified (from Session.list)
        items: list[ListItem] = []
        accent = config.ui.colors.accent

        def _walk(node: SessionInfo, depth: int) -> None:
            prefix = ""
            if depth > 0:
                prefix = f"{'   ' * (depth - 1)} └ [handoff] "
            label = self._format_session_label(node.first_message)
            caption = f"{self._format_session_age(node.modified)} {node.message_count}"
            items.append(
                ListItem(
                    value=node,
                    label=label,
                    description=caption,
                    prefix=prefix,
                    prefix_style=accent,
                )
            )
            for child in children.get(node.id, []):
                _walk(child, depth + 1)

        for root in roots:
            _walk(root, 0)

        return items

    def _show_tree_selector(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        input_box = self.query_one("#input-box", InputBox)
        if self._is_running:
            chat.add_info_message("Cannot open tree while agent is running", error=True)
            return
        if not self._runtime.session or not self._runtime.session.all_entries:
            chat.add_info_message("No entries in session")
            return
        tree = self._runtime.session.get_tree()
        selector = self.query_one("#tree-selector", TreeSelector)
        input_box.clear()
        input_box.set_autocomplete_enabled(False)
        input_box.set_completing(True)
        selector.show(
            tree,
            self._runtime.session.leaf_id,
            getattr(self, "size", None).height if getattr(self, "size", None) else 24,  # pyright: ignore[reportOptionalMemberAccess]
        )
        self._selection_mode = SelectionMode.TREE

    def _show_resume_sessions(self) -> None:
        items = self._build_resume_items()
        if not items:
            self.notify(
                "No saved sessions found", title="Sessions", timeout=3, severity="information"
            )
            return

        self._show_selection_picker(items, SelectionMode.SESSION, max_label_width=87)

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
        if self._runtime.session and self._runtime.session.session_file is not None:
            current_session_path = Path(self._runtime.session.session_file)

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
        was_at_bottom = self._is_chat_at_bottom()
        if not items:
            self._hide_completion_list()
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
        else:
            completion_list.update_items(items)
            self.notify("Session deleted", title="Sessions", timeout=2, severity="information")

        self._restore_chat_scroll_after_refresh(was_at_bottom)

    def _handle_login_command(self, args: str) -> None:
        providers = [
            ("github-copilot", "GitHub Copilot", has_saved_copilot_credentials()),
            ("openai", "OpenAI (ChatGPT/Codex)", has_saved_openai_credentials()),
        ]

        self._show_selection_picker(
            [
                ListItem(
                    value=provider_id,
                    label=name,
                    description="saved credentials" if has_credentials else "",
                )
                for provider_id, name, has_credentials in providers
            ],
            SelectionMode.LOGIN,
        )

    def _select_login_provider(self, provider_id: str) -> None:
        if provider_id == "github-copilot":
            self.run_worker(self._copilot_login_flow(), exclusive=False)
            return

        if provider_id == "openai":
            self.run_worker(self._openai_login_flow(), exclusive=False)

    async def _copilot_login_flow(self) -> None:
        import webbrowser

        chat = self.query_one("#chat-log", ChatLog)
        had_saved_credentials = has_saved_copilot_credentials()

        def on_user_code(url: str, code: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                f"Opening browser to: {url}\n"
                f"Enter this code: {code}\n\n"
                "Waiting for authorization...",
            )

        try:
            if await get_copilot_token():
                chat.add_info_message("Already logged in to GitHub Copilot")
                return

            if had_saved_credentials:
                chat.add_info_message(
                    "Your saved GitHub Copilot session is no longer valid.", warning=True
                )
            else:
                chat.add_info_message("Starting GitHub Copilot login...")

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
        had_saved_credentials = has_saved_openai_credentials()

        def on_auth_url(url: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                "Opening browser for OpenAI OAuth...\n"
                f"If browser does not open, visit:\n{url}\n\n"
                "Waiting for authorization callback on http://localhost:1455/auth/callback ...",
            )

        try:
            if await get_valid_openai_credentials():
                chat.add_info_message("Already logged in to OpenAI")
                return

            if had_saved_credentials:
                chat.add_info_message(
                    "Your saved OpenAI session is no longer valid.", warning=True
                )
            else:
                chat.add_info_message("Starting OpenAI login...")

            await openai_login(on_auth_url=on_auth_url)
            chat.add_info_message(
                "Successfully logged in to OpenAI!\n"
                "You can now use /model to select openai-codex models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    def _handle_logout_command(self, args: str) -> None:
        providers = []
        if has_saved_copilot_credentials():
            providers.append(("github-copilot", "GitHub Copilot"))
        if has_saved_openai_credentials():
            providers.append(("openai", "OpenAI (ChatGPT/Codex)"))

        if not providers:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info_message("No providers logged in")
            return

        self._show_selection_picker(
            [
                ListItem(value=provider_id, label=name, description="")
                for provider_id, name in providers
            ],
            SelectionMode.LOGOUT,
        )

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

        if not self._runtime.session:
            chat.add_info_message("No active session to export")
            return

        if not self._runtime.session.entries:
            chat.add_info_message("Session has no messages to export")
            return

        try:
            path = export_session_html(
                cwd=self._cwd,
                session_id=self._runtime.session.id,
                output_dir=self._cwd,
                version=getattr(self, "VERSION", ""),
            )
            chat.add_info_message(f"Session exported to {path.name}")
        except Exception as e:
            chat.add_info_message(f"Export failed: {e}", error=True)

    def _handle_copy_command(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if not self._runtime.session:
            chat.add_info_message("No agent messages to copy yet", error=True)
            return

        text = self._runtime.session.get_last_assistant_text()
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

        if self._runtime.provider is None or self._runtime.session is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        if not self._runtime.session.all_messages:
            chat.add_info_message("No conversation to compact", error=True)
            return

        chat.show_spinner_status("Compacting...")
        self.run_worker(self._do_compact(), exclusive=False)

    async def _do_compact(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        if self._runtime.provider is None or self._runtime.session is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        try:
            result = await self._runtime.compact_now()
            chat.add_compaction_message(result.tokens_before)
        except Exception as e:
            chat.show_status("Compaction failed")
            chat.add_info_message(f"Compaction failed: {e}", error=True)

    def _format_session_label(self, message: str) -> str:
        return " ".join(message.split())

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
