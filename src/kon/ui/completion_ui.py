"""Completion list, selection-mode pickers and tree-selector message handling.

The @on-decorated handlers defined here must be re-bound in the Kon class body:
Textual registers them through a metaclass that only scans the namespace of
classes created with it, and this mixin is a plain class.
"""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING, Any

from textual import on

from ..runtime import ConversationRuntime
from .autocomplete import FilePathProvider, PullRequestProvider, SlashCommandProvider
from .chat import ChatLog
from .floating_list import FloatingList, ListItem
from .input import InputBox
from .selection_mode import SelectionMode
from .tree import TreeSelector
from .widgets import InfoBar, StatusLine, format_path


class CompletionUIMixin:
    _selection_mode: SelectionMode | None
    _settings_active: bool
    _settings_selected_value: str | None
    _runtime: ConversationRuntime

    if TYPE_CHECKING:
        query_one: Any
        batch_update: Any
        call_later: Any
        call_after_refresh: Any
        run_worker: Any

        def _reset_ctrl_d_delete_state(self) -> None: ...
        def _show_settings_picker(self, selected_value: str | None = None) -> None: ...
        def _handle_settings_select(self, item_value: str) -> str: ...
        def _select_model(self, model) -> None: ...
        def _select_theme(self, theme_id: str) -> None: ...
        def _select_permission_mode(self, mode) -> None: ...
        def _select_thinking_level(self, level: str) -> None: ...
        def _select_thinking_lines(self, lines) -> None: ...
        def _select_notifications_mode(self, mode) -> None: ...
        def _select_login_provider(self, provider_id: str) -> None: ...
        def _select_logout_provider(self, provider_id: str) -> None: ...
        def _render_session_entries(self, session) -> None: ...
        async def _load_session(self, session_path) -> None: ...

    def _is_chat_at_bottom(self) -> bool:
        chat = self.query_one("#chat-log", ChatLog)
        return abs(chat.max_scroll_y - chat.scroll_y) < 1

    def _restore_chat_scroll_if_needed(self, was_at_bottom: bool) -> None:
        # The completion list is a normal grid row, not a true overlay. Showing or
        # hiding it changes the available height for ChatLog. When the chat is
        # already bottom-aligned, that resize can leave the viewport briefly at an
        # intermediate scroll offset and cause a visible flicker. Restore the
        # bottom scroll position after Textual has applied the layout change.
        if was_at_bottom:
            chat = self.query_one("#chat-log", ChatLog)
            chat.scroll_end(animate=False)

    def _restore_chat_scroll_after_refresh(self, was_at_bottom: bool) -> None:
        self.call_after_refresh(lambda: self._restore_chat_scroll_if_needed(was_at_bottom))

    def _set_bottom_info_displaced(self, displaced: bool) -> None:
        info_bar = self.query_one("#info-bar", InfoBar)
        if displaced:
            info_bar.add_class("-completion-hidden")
        else:
            info_bar.remove_class("-completion-hidden")

    def _show_completion_list(
        self,
        items: list[ListItem],
        *,
        searchable: bool = False,
        max_label_width: int | None = None,
    ) -> None:
        completion_list = self.query_one("#completion-list", FloatingList)
        self._set_bottom_info_displaced(True)
        completion_list.show(items, searchable=searchable, max_label_width=max_label_width)

    def _hide_completion_list(self, *, restore_info_bar: bool = True) -> None:
        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.hide()
        if restore_info_bar:
            self._set_bottom_info_displaced(False)

    @on(InputBox.CompletionUpdate)
    def on_completion_update(self, event: InputBox.CompletionUpdate) -> None:
        if self._selection_mode is not None:
            return

        completion_list = self.query_one("#completion-list", FloatingList)
        was_at_bottom = self._is_chat_at_bottom()
        if completion_list.is_visible:
            self._set_bottom_info_displaced(True)
            completion_list.update_items(event.items)
        else:
            self._show_completion_list(event.items)
        self._restore_chat_scroll_after_refresh(was_at_bottom)

    @on(InputBox.CompletionHide)
    def on_completion_hide(self, event: InputBox.CompletionHide) -> None:
        input_box = self.query_one("#input-box", InputBox)
        was_at_bottom = self._is_chat_at_bottom()

        with self.batch_update():
            if self._selection_mode is not None:
                # If we were in a sub-picker from settings, go back to settings
                if self._settings_active:
                    self._hide_completion_list(restore_info_bar=False)
                    self._settings_active = False
                    self._show_settings_picker()
                    self._restore_chat_scroll_after_refresh(was_at_bottom)
                    return

                self._hide_completion_list()
                self._selection_mode = None
                input_box.clear()
                input_box.set_autocomplete_enabled(True)
                self._reset_ctrl_d_delete_state()
            else:
                self._hide_completion_list()

            input_box.set_completing(False)

        self._restore_chat_scroll_after_refresh(was_at_bottom)

    @on(InputBox.CompletionSelect)
    def on_completion_select(self, event: InputBox.CompletionSelect) -> None:
        input_box = self.query_one("#input-box", InputBox)
        if self._selection_mode == SelectionMode.TREE:
            self.query_one("#tree-selector", TreeSelector).action_select()
            return
        was_at_bottom = self._is_chat_at_bottom()
        completion_list = self.query_one("#completion-list", FloatingList)
        item = completion_list.selected_item

        if not item:
            self._hide_completion_list()
            input_box.set_completing(False)
            input_box.submit_raw()
            self._restore_chat_scroll_after_refresh(was_at_bottom)
            return

        if self._selection_mode is not None:
            self._apply_selection_mode_choice(item, input_box, was_at_bottom)
            return

        if input_box.is_tab_completing:
            self._hide_completion_list()
            input_box.apply_tab_path_completion(item)
            self._restore_chat_scroll_after_refresh(was_at_bottom)
            return

        provider = input_box.active_provider
        self._hide_completion_list()

        if isinstance(provider, SlashCommandProvider):
            input_box.apply_slash_command(item)
        elif isinstance(provider, FilePathProvider | PullRequestProvider):
            input_box.apply_provider_completion(item)

        input_box.set_completing(False)
        self._restore_chat_scroll_after_refresh(was_at_bottom)

    def _apply_selection_mode_choice(
        self, item: ListItem, input_box: InputBox, was_at_bottom: bool
    ) -> None:
        """Apply the picked item for the active selection mode (model/theme/session/...)."""
        selection_mode = self._selection_mode
        keeps_info_bar_displaced = selection_mode == SelectionMode.SETTINGS or (
            selection_mode
            in (SelectionMode.THEME, SelectionMode.THINKING, SelectionMode.THINKING_LINES)
            and self._settings_active
        )
        with self.batch_update():
            self._hide_completion_list(restore_info_bar=not keeps_info_bar_displaced)
            self._selection_mode = None
            input_box.clear()
            input_box.set_autocomplete_enabled(True)
            input_box.set_completing(False)
            self._reset_ctrl_d_delete_state()

        def show_settings_picker_and_restore() -> None:
            self._show_settings_picker()
            self._restore_chat_scroll_after_refresh(was_at_bottom)

        match selection_mode:
            case SelectionMode.SETTINGS:
                settings_result = self._handle_settings_select(item.value)
                if settings_result == "closed":
                    self._set_bottom_info_displaced(False)
            case SelectionMode.SESSION:
                self.run_worker(self._load_session(item.value.path), exclusive=True)
            case SelectionMode.URL:
                webbrowser.open(item.value)
            case SelectionMode.TREE:
                pass
            case SelectionMode.MODEL:
                self._select_model(item.value)
            case SelectionMode.THEME:
                self._select_theme(item.value)
                if self._settings_active:
                    self._settings_active = False
                    self.call_later(show_settings_picker_and_restore)
                    return
            case SelectionMode.PERMISSIONS:
                self._select_permission_mode(item.value)
            case SelectionMode.THINKING:
                self._select_thinking_level(item.value)
                if self._settings_active:
                    self._settings_active = False
                    self.call_later(show_settings_picker_and_restore)
                    return
            case SelectionMode.THINKING_LINES:
                self._select_thinking_lines(item.value)
                if self._settings_active:
                    self._settings_active = False
                    self.call_later(show_settings_picker_and_restore)
                    return
            case SelectionMode.NOTIFICATIONS:
                self._select_notifications_mode(item.value)
            case SelectionMode.LOGIN:
                self._select_login_provider(item.value)
            case SelectionMode.LOGOUT:
                self._select_logout_provider(item.value)

        self._restore_chat_scroll_after_refresh(was_at_bottom)

    @on(InputBox.SearchUpdate)
    def on_search_update(self, event: InputBox.SearchUpdate) -> None:
        if self._selection_mode is None or self._selection_mode == SelectionMode.TREE:
            return
        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.set_search_query(event.query)
        if (
            self._selection_mode == SelectionMode.SETTINGS
            and not event.query
            and self._settings_selected_value is not None
        ):
            completion_list.select_value(self._settings_selected_value)

    @on(InputBox.CompletionMove)
    def on_completion_move(self, event: InputBox.CompletionMove) -> None:
        if self._selection_mode == SelectionMode.TREE:
            selector = self.query_one("#tree-selector", TreeSelector)
            if event.direction < 0:
                selector.action_move_up()
            else:
                selector.action_move_down()
            return
        completion_list = self.query_one("#completion-list", FloatingList)
        if event.direction < 0:
            completion_list.move_up()
        else:
            completion_list.move_down()

    @on(TreeSelector.Selected)
    async def on_tree_selected(self, event: TreeSelector.Selected) -> None:
        selector = self.query_one("#tree-selector", TreeSelector)
        input_box = self.query_one("#input-box", InputBox)
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)
        status = self.query_one("#status-line", StatusLine)
        try:
            result = self._runtime.navigate_tree(event.entry_id)
        except Exception as exc:
            chat.add_info_message(f"Tree navigation failed: {exc}", error=True)
            return
        selector.hide()
        self._selection_mode = None
        input_box.set_autocomplete_enabled(True)
        input_box.set_completing(False)
        await chat.remove_all_children()
        chat.add_session_info(getattr(self, "VERSION", ""))
        if self._runtime.context:
            chat.add_loaded_resources(
                context_paths=[format_path(f.path) for f in self._runtime.context.agents_files],
                skills=self._runtime.context.skills,
                tools=self._runtime.tools,
            )
        if self._runtime.session:
            self._render_session_entries(self._runtime.session)
            totals = self._runtime.session.token_totals()
            info_bar.set_tokens(
                totals.input_tokens,
                totals.output_tokens,
                totals.context_tokens,
                totals.cache_read_tokens,
                totals.cache_write_tokens,
            )
            info_bar.set_file_changes(self._runtime.session.file_changes_summary())
        status.reset()
        if result.editor_text and not input_box.text.strip():
            input_box.insert(result.editor_text)
        chat.show_status("Navigated to selected point")
        input_box.focus()

    @on(TreeSelector.Cancelled)
    def on_tree_cancelled(self, event: TreeSelector.Cancelled) -> None:
        selector = self.query_one("#tree-selector", TreeSelector)
        input_box = self.query_one("#input-box", InputBox)
        selector.hide()
        self._selection_mode = None
        input_box.set_autocomplete_enabled(True)
        input_box.set_completing(False)
        input_box.focus()
