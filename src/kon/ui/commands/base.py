"""Shared plumbing for the command mixins: expected app surface and picker helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ...runtime import ConversationRuntime
from ...session import Session
from ..chat import ChatLog
from ..floating_list import ListItem
from ..input import InputBox
from ..selection_mode import SelectionMode

Choice = TypeVar("Choice", bound=str)


class CommandSupport:
    """Attributes and helpers every command mixin can rely on."""

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
