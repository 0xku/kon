from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import TextArea

from .autocomplete import (
    DEFAULT_COMMANDS,
    AutocompleteProvider,
    FilePathProvider,
    SlashCommand,
    SlashCommandProvider,
)
from .floating_list import ListItem
from .path_complete import PathComplete
from .prompt_history import PromptHistory

if TYPE_CHECKING:
    pass


_PASTE_LINE_THRESHOLD = 5
_PASTE_CHAR_THRESHOLD = 500
_PASTE_MARKER_RE = re.compile(r"\[paste #(\d+)(?: (\+\d+ lines|\d+ chars))?\]")
_SKILL_TRIGGER_MARKER = "\u2063"


class Kon(TextArea):
    class ScrollInfo(Message):
        def __init__(self, lines_above: int, lines_below: int) -> None:
            super().__init__()
            self.lines_above = lines_above
            self.lines_below = lines_below

    def __init__(self, on_paste: Callable[[str], str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._on_paste_transform = on_paste

    async def _on_key(self, event: events.Key) -> None:
        future = getattr(self.app, "_approval_future", None)
        if future and not future.done() and event.key in ("y", "Y", "n", "N"):
            self.app.on_key(event)
            return
        await super()._on_key(event)

    async def _on_paste(self, event: events.Paste) -> None:
        # Prevent TextArea._on_paste from also running on the original event.
        event.prevent_default()
        transformed = self._on_paste_transform(event.text)
        await super()._on_paste(events.Paste(transformed))

    def _notify_scroll_info(self) -> None:
        total_lines = self.document.line_count
        visible_lines = self.scrollable_content_region.height
        if visible_lines <= 0:
            return
        lines_above = int(self.scroll_y)
        lines_below = max(0, total_lines - lines_above - visible_lines)
        self.post_message(self.ScrollInfo(lines_above, lines_below))

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.call_after_refresh(self._notify_scroll_info)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self.call_after_refresh(self._notify_scroll_info)


class InputBox(Vertical):
    """
    Multi-line input with inline completion support.

    - Enter: Submit
    - Shift+Enter/Ctrl+J: Newline
    - Up/Down: History navigation when at top/bottom, or list navigation when completing
    - @ triggers file search (inline)
    - / triggers slash commands (inline, at start of input)
    - Escape: Cancel completion or clear input

    The FloatingList is managed externally (at app level) but controlled
    via messages from InputBox.
    """

    BINDINGS: ClassVar[list] = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("ctrl+j,shift+enter", "newline", "New line", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("up", "cursor_up", "Up", priority=True),
        Binding("down", "cursor_down", "Down", priority=True),
        Binding("tab", "tab_complete", "Tab complete", priority=True),
    ]

    DEFAULT_CSS = """
    InputBox {
        height: auto;
        min-height: 3;
        max-height: 30vh;
        border-top: solid grey;
        border-bottom: solid grey;
        border-title-align: left;
        border-subtitle-align: left;
        border-title-color: grey;
        border-subtitle-color: grey;
    }

    InputBox .input-textarea {
        width: 1fr;
        height: auto;
        max-height: 100%;
        border: none;
        background: transparent;
        padding: 0 1;
    }

    InputBox .input-textarea:focus {
        border: none;
    }
    """

    def __init__(
        self, cwd: str | None = None, id: str | None = None, classes: str | None = None
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._cwd = cwd or os.getcwd()
        self._history = PromptHistory()

        # Autocomplete providers
        self._slash_provider = SlashCommandProvider(DEFAULT_COMMANDS.copy())
        self._file_provider = FilePathProvider(self._cwd)
        self._providers: list[AutocompleteProvider] = [self._slash_provider, self._file_provider]

        # Active completion state (the list itself is external)
        self._active_provider: AutocompleteProvider | None = None
        self._completion_prefix: str = ""
        self._is_completing: bool = False
        self._autocomplete_enabled: bool = True
        self._suppress_autocomplete: int = 0  # Skip N autocomplete triggers

        # Tab path completion state
        self._path_complete = PathComplete()
        self._tab_completing: bool = False
        self._tab_start_col: int = 0
        self._tab_base_fragment: str = ""

        # Large paste compaction
        self._pastes: dict[int, str] = {}
        self._paste_counter: int = 0

        # Skill command triggers selected from slash autocomplete
        self._selected_skill_commands: list[str] = []

    def compose(self) -> ComposeResult:
        yield Kon(self._transform_paste, id="input-textarea", classes="input-textarea")

    def on_mount(self) -> None:
        textarea = self.query_one("#input-textarea", TextArea)
        textarea.cursor_blink = False
        textarea.show_line_numbers = False
        textarea.highlight_cursor_line = False

    def on_kon_scroll_info(self, event: Kon.ScrollInfo) -> None:
        event.stop()
        self.border_title = f"↑ {event.lines_above} more" if event.lines_above > 0 else ""
        self.border_subtitle = f"↓ {event.lines_below} more" if event.lines_below > 0 else ""

    @property
    def text(self) -> str:
        return self.query_one("#input-textarea", TextArea).text

    @property
    def is_completing(self) -> bool:
        return self._is_completing

    @property
    def is_tab_completing(self) -> bool:
        return self._tab_completing

    def clear(self, *, reset_pastes: bool = True) -> None:
        self.query_one("#input-textarea", TextArea).clear()
        self._selected_skill_commands.clear()
        self.border_title = ""
        self.border_subtitle = ""
        if reset_pastes:
            self._reset_pastes()

    def insert(self, text: str) -> None:
        self.query_one("#input-textarea", TextArea).insert(text)

    def focus(self, scroll_visible: bool = True) -> InputBox:
        self.query_one("#input-textarea", TextArea).focus(scroll_visible)
        return self

    def set_commands(self, commands: list[SlashCommand]) -> None:
        self._slash_provider.commands = commands

    def set_fd_path(self, fd_path: str | None) -> None:
        self._file_provider.set_fd_path(fd_path)

    def set_file_paths(self, paths: list[str]) -> None:
        self._file_provider.set_paths(paths)

    def set_cwd(self, cwd: str) -> None:
        self._cwd = cwd
        self._file_provider.set_cwd(cwd)
        self._path_complete.clear_cache()

    def set_autocomplete_enabled(self, enabled: bool) -> None:
        self._autocomplete_enabled = enabled

    def set_completing(self, is_completing: bool) -> None:
        self._is_completing = is_completing
        if not is_completing:
            self._active_provider = None
            self._completion_prefix = ""
            self._tab_completing = False
            self._tab_start_col = 0
            self._tab_base_fragment = ""

    def _transform_paste(self, pasted_text: str) -> str:
        normalized = pasted_text.replace("\r\n", "\n").replace("\r", "\n")
        filtered = "".join(char for char in normalized if char == "\n" or ord(char) >= 32)
        line_count = len(filtered.split("\n"))
        char_count = len(filtered)

        if line_count > _PASTE_LINE_THRESHOLD or char_count > _PASTE_CHAR_THRESHOLD:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered
            if line_count > _PASTE_LINE_THRESHOLD:
                return f"[paste #{paste_id} +{line_count} lines]"
            return f"[paste #{paste_id} {char_count} chars]"

        return filtered

    def _expand_paste_markers(self, text: str) -> str:
        def replace_match(match: re.Match[str]) -> str:
            paste_id = int(match.group(1))
            return self._pastes.get(paste_id, match.group(0))

        return _PASTE_MARKER_RE.sub(replace_match, text)

    def _reset_pastes(self) -> None:
        self._pastes.clear()
        self._paste_counter = 0

    def _strip_skill_markers(self, text: str) -> str:
        return text.replace(_SKILL_TRIGGER_MARKER, "")

    def _extract_selected_skill_submission(self, text: str) -> tuple[str | None, str | None]:
        pattern = re.compile(rf"{_SKILL_TRIGGER_MARKER}/([a-z0-9-]+){_SKILL_TRIGGER_MARKER}")
        match = pattern.search(text)
        if not match:
            return None, None

        skill_name = match.group(1)
        if skill_name not in self._selected_skill_commands:
            return None, None

        query = (text[: match.start()] + text[match.end() :]).strip()
        return skill_name, self._strip_skill_markers(query)

    # -------------------------------------------------------------------------
    # Text change handling - trigger autocomplete
    # -------------------------------------------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        event.stop()

        # Skip autocomplete if we just applied a completion
        if self._suppress_autocomplete > 0:
            self._suppress_autocomplete -= 1
            return

        if not self._autocomplete_enabled:
            # When completing with autocomplete disabled (selection mode),
            # route text to the floating list search
            if self._is_completing:
                self.post_message(self.SearchUpdate(self.text))
            return

        self._try_autocomplete()

    def _cursor_offset(self, text: str, cursor: tuple[int, int]) -> int:
        row, col = cursor
        lines = text.split("\n")
        if row <= 0:
            return max(0, min(col, len(lines[0]) if lines else 0))
        clamped_row = min(row, len(lines) - 1)
        prefix_len = sum(len(line) + 1 for line in lines[:clamped_row])
        return prefix_len + max(0, min(col, len(lines[clamped_row])))

    def _try_autocomplete(self) -> None:
        textarea = self.query_one("#input-textarea", TextArea)
        text = textarea.text
        cursor_col = self._cursor_offset(text, textarea.selection.end)

        # Check each provider
        for provider in self._providers:
            if provider.should_trigger(text, cursor_col):
                result = provider.get_suggestions(text, cursor_col)
                if result and result.items:
                    self._active_provider = provider
                    self._completion_prefix = result.prefix
                    self._is_completing = True
                    # Post message for app to show/update the list
                    self.post_message(self.CompletionUpdate(result.items))
                    return

        # No provider matched - hide completion
        if self._is_completing:
            self._is_completing = False
            self._active_provider = None
            self._completion_prefix = ""
            self.post_message(self.CompletionHide())

    # -------------------------------------------------------------------------
    # Key handling
    # -------------------------------------------------------------------------

    def action_submit(self) -> None:
        if self._is_completing:
            # Tell app to apply the current selection
            self.post_message(self.CompletionSelect())
            return
        self._do_submit()

    def _do_submit(self) -> None:
        raw_text = self.text.strip()
        if not raw_text:
            return
        query_text = self._expand_paste_markers(raw_text)
        selected_skill_name, selected_skill_query = self._extract_selected_skill_submission(
            query_text
        )
        display_text = self._strip_skill_markers(raw_text)
        query_text = self._strip_skill_markers(query_text)
        self._add_to_history(query_text)
        self.post_message(
            self.Submitted(
                display_text,
                query_text=query_text,
                selected_skill_name=selected_skill_name,
                selected_skill_query=selected_skill_query,
            )
        )
        self.clear(reset_pastes=True)

    def submit_raw(self) -> None:
        self._is_completing = False
        self._active_provider = None
        self._completion_prefix = ""
        self._do_submit()

    def action_newline(self) -> None:
        self.query_one("#input-textarea", TextArea).insert("\n")

    def action_cancel(self) -> None:
        if self._is_completing:
            self._is_completing = False
            self._active_provider = None
            self._completion_prefix = ""
            self.post_message(self.CompletionHide())
            return

        app = self.app
        if getattr(app, "deny_pending_approval", lambda: False)():
            return
        if getattr(app, "_is_running", False):
            app.action_interrupt_agent()  # type: ignore
        else:
            self.clear()

    def action_cursor_up(self) -> None:
        if self._is_completing:
            self.post_message(self.CompletionMove(-1))
            return
        textarea = self.query_one("#input-textarea", TextArea)
        row, _ = textarea.selection.start
        if row > 0:
            textarea.action_cursor_up()
        elif not textarea.text.strip() or self._history.is_browsing:
            self._history_navigate(-1)
        else:
            textarea.action_cursor_line_start()

    def action_cursor_down(self) -> None:
        if self._is_completing:
            self.post_message(self.CompletionMove(1))
            return
        textarea = self.query_one("#input-textarea", TextArea)
        row, _ = textarea.selection.start
        if row < textarea.document.line_count - 1:
            textarea.action_cursor_down()
        elif self._history.is_browsing:
            self._history_navigate(1)
        else:
            textarea.action_cursor_line_end()

    def action_tab_complete(self) -> None:
        """Handle Tab key for path completion."""
        self.run_worker(self._do_tab_complete())

    async def _do_tab_complete(self) -> None:
        """Perform tab completion asynchronously."""
        # If already completing, treat Tab as moving down in the list
        if self._is_completing:
            self.post_message(self.CompletionMove(1))
            return

        textarea = self.query_one("#input-textarea", TextArea)
        cursor_pos = textarea.selection.end
        text = textarea.text

        # Get text before cursor on current line
        row, col = cursor_pos
        lines = text.split("\n")
        if row >= len(lines):
            return
        line = lines[row]
        text_before_cursor = line[:col]

        # Extract path fragment (last word/token before cursor)
        path_fragment, start_col = PathComplete.extract_path_fragment(text_before_cursor)
        if not path_fragment:
            # No path to complete - insert literal tab (spaces)
            self._suppress_autocomplete = 1
            textarea.insert("    ")
            return

        # Call PathComplete
        completion, alternatives = await self._path_complete(self._cwd, path_fragment)

        if not completion and not alternatives:
            # No matches - beep
            self.app.bell()
            return

        if completion and not alternatives:
            # Unique completion - insert directly
            self._suppress_autocomplete = 1
            textarea.insert(completion)
            # Add space after files (not directories)
            if not completion.endswith(os.sep):
                textarea.insert(" ")
            return

        # Multiple alternatives - show floating list
        # First, insert any common prefix
        if completion:
            self._suppress_autocomplete = 1
            textarea.insert(completion)
            # Update cursor position after insertion
            col = col + len(completion)

        # Prepare items for floating list
        base_fragment = PathComplete.get_base_path(path_fragment + completion)
        items = []
        for alt in alternatives[:20]:  # Limit to 20 items
            label = alt
            # Show the base path as description
            description = base_fragment if base_fragment else "."
            items.append(ListItem(value=alt, label=label, description=description))

        # Save state for applying completion later
        self._tab_completing = True
        self._tab_start_col = start_col
        self._tab_base_fragment = base_fragment
        self._is_completing = True

        # Show the floating list
        self.post_message(self.CompletionUpdate(items))

    # -------------------------------------------------------------------------
    # Completion application (called by app after selection)
    # -------------------------------------------------------------------------

    def apply_slash_command(self, item: ListItem) -> None:
        cmd: SlashCommand = item.value
        self._is_completing = False
        self._active_provider = None

        if cmd.submit_on_select and not cmd.is_skill:
            self._completion_prefix = ""
            self._suppress_autocomplete = 1  # clear() = 1 event
            self.clear(reset_pastes=True)
            self.post_message(self.Submitted(f"/{cmd.name}"))
            return

        if not cmd.is_skill:
            prefix = self._completion_prefix
            self._completion_prefix = ""

            textarea = self.query_one("#input-textarea", TextArea)
            text = textarea.text
            cursor_col = self._cursor_offset(text, textarea.selection.end)
            new_text, _ = self._slash_provider.apply_completion(text, cursor_col, item, prefix)

            self._suppress_autocomplete = 2  # clear() + insert() = 2 events
            textarea.clear()
            textarea.insert(new_text)
            return

        prefix = self._completion_prefix
        self._completion_prefix = ""

        textarea = self.query_one("#input-textarea", TextArea)
        text = textarea.text
        cursor_col = self._cursor_offset(text, textarea.selection.end)

        new_text, _ = self._slash_provider.apply_completion(text, cursor_col, item, prefix)

        if cmd.name not in self._selected_skill_commands:
            self._selected_skill_commands.append(cmd.name)
        marker_wrapped = f"{_SKILL_TRIGGER_MARKER}/{cmd.name}{_SKILL_TRIGGER_MARKER} "
        plain = f"/{cmd.name} "
        if plain in new_text:
            new_text = new_text.replace(plain, marker_wrapped, 1)

        self._suppress_autocomplete = 2  # clear() + insert() = 2 events
        textarea.clear()
        textarea.insert(new_text)

    def apply_file_completion(self, item: ListItem) -> None:
        textarea = self.query_one("#input-textarea", TextArea)
        text = textarea.text
        cursor_col = self._cursor_offset(text, textarea.selection.end)

        new_text, _ = self._file_provider.apply_completion(
            text, cursor_col, item, self._completion_prefix
        )

        self._is_completing = False
        self._active_provider = None
        self._completion_prefix = ""
        self._suppress_autocomplete = 2  # clear() + insert() = 2 events
        textarea.clear()
        textarea.insert(new_text)

    def apply_tab_path_completion(self, item: ListItem) -> None:
        """Apply a tab path completion selection."""
        textarea = self.query_one("#input-textarea", TextArea)
        text = textarea.text
        cursor_col = self._cursor_offset(text, textarea.selection.end)

        # Get the selected path
        selected_path: str = item.value

        # Build the new path: base_fragment + selected
        new_path = self._tab_base_fragment + selected_path

        # Quote if contains spaces
        if " " in new_path and not new_path.startswith('"'):
            new_path = f'"{new_path}"'

        # Replace from start_col to cursor
        text_before = text[: self._tab_start_col]
        text_after = text[cursor_col:]

        # Add space after files (not directories)
        is_dir = selected_path.endswith("/") or selected_path.endswith(os.sep)
        suffix = "" if is_dir else " "

        new_text = text_before + new_path + suffix + text_after

        # Clear state
        self._is_completing = False
        self._tab_completing = False
        self._tab_start_col = 0
        self._tab_base_fragment = ""
        self._suppress_autocomplete = 2  # clear() + insert() = 2 events

        textarea.clear()
        textarea.insert(new_text)

    @property
    def active_provider(self) -> AutocompleteProvider | None:
        return self._active_provider

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    def _add_to_history(self, text: str) -> None:
        self._history.append(text)

    def _history_navigate(self, direction: int) -> None:
        textarea = self.query_one("#input-textarea", TextArea)
        result = self._history.navigate(direction, textarea.text)
        if result is None:
            return
        textarea.clear()
        textarea.insert(result)

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    class Submitted(Message):
        def __init__(
            self,
            text: str,
            query_text: str | None = None,
            selected_skill_name: str | None = None,
            selected_skill_query: str | None = None,
        ) -> None:
            super().__init__()
            self.text = text
            self.query_text = query_text if query_text is not None else text
            self.selected_skill_name = selected_skill_name
            self.selected_skill_query = selected_skill_query

    class CompletionUpdate(Message):
        def __init__(self, items: list[ListItem]) -> None:
            super().__init__()
            self.items = items

    class CompletionHide(Message):
        pass

    class CompletionSelect(Message):
        pass

    class CompletionMove(Message):
        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    class SearchUpdate(Message):
        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query
