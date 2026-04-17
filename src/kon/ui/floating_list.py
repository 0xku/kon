"""
Floating list overlay for inline completion.

A reusable overlay component that renders below the input, showing
a paginated list with arrow indicator and counter. Used for:
- Slash commands (/)
- File path search (@)
- Session selection
- Any other searchable list
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from kon import config

T = TypeVar("T")


@dataclass
class ListItem[T]:
    value: T
    label: str
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.label, self.description))


class FloatingList[T](Widget):
    """
    A floating overlay list with pagination and selection.

    Features:
    - Arrow indicator (→) for selected item
    - Position counter (x/total)
    - Window-based pagination (shows subset of items)
    - Keyboard navigation (up/down)
    - Optional search bar for filtering (two-layer commands)
    - Hidden by default, show/hide controlled by parent

    The parent widget is responsible for:
    - Calling show(items) with filtered items
    - Calling hide() to dismiss
    - Calling move_up()/move_down() on key events
    - Reading selected_item when user confirms
    """

    DEFAULT_CSS = """
    FloatingList {
        height: auto;
        display: none;
        padding: 0 1;
    }

    FloatingList.-visible {
        display: block;
    }
    """

    # Reactive to trigger re-render
    _selected_index: reactive[int] = reactive(0, repaint=False)
    _visible: reactive[bool] = reactive(False, repaint=False)
    _render_key: reactive[int] = reactive(0)  # Force re-render

    def __init__(
        self,
        window_size: int = 5,
        label_width: int = 12,
        max_label_width: int = 30,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._window_size = window_size
        self._min_label_width = label_width
        self._max_label_width = max_label_width
        self._label_width = label_width
        self._items: list[ListItem[T]] = []

        # Search state
        self._search_enabled = False
        self._search_query = ""
        self._all_items: list[ListItem[T]] = []

    @property
    def items(self) -> list[ListItem[T]]:
        return self._items

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def selected_item(self) -> ListItem[T] | None:
        if self._items and 0 <= self._selected_index < len(self._items):
            return self._items[self._selected_index]
        return None

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def search_enabled(self) -> bool:
        return self._search_enabled

    def _compute_label_width(self) -> int:
        # Compute from all items when search is enabled to keep stable column width
        source = self._all_items if self._search_enabled and self._all_items else self._items
        if not source:
            return self._min_label_width
        max_len = max(len(item.label) for item in source)
        return max(self._min_label_width, min(max_len, self._max_label_width))

    def show(
        self,
        items: list[ListItem[T]],
        searchable: bool = False,
        max_label_width: int | None = None,
    ) -> None:
        self._search_enabled = searchable
        self._search_query = ""
        self._all_items = items if searchable else []
        self._items = items
        self._selected_index = 0
        if max_label_width is not None:
            self._max_label_width = max_label_width
        self._label_width = self._compute_label_width()
        self._visible = True
        self.add_class("-visible")
        self._render_key += 1
        # Force layout refresh to prevent visual artifacts in adjacent widgets
        if self.screen:
            self.screen.refresh(layout=True)

    def hide(self) -> None:
        self._visible = False
        self._items = []
        self._all_items = []
        self._selected_index = 0
        self._search_enabled = False
        self._search_query = ""
        self.remove_class("-visible")
        # Force layout refresh to prevent visual artifacts in adjacent widgets
        if self.screen:
            self.screen.refresh(layout=True)

    def set_search_query(self, query: str) -> None:
        if not self._search_enabled:
            return
        self._search_query = query
        if not query:
            self._items = self._all_items
        else:
            self._items = self._fuzzy_filter(query, self._all_items)
        self._label_width = self._compute_label_width()
        self._selected_index = 0
        self._render_key += 1

    def update_items(self, items: list[ListItem[T]]) -> None:
        self._items = items
        if self._search_enabled:
            self._all_items = items
        self._label_width = self._compute_label_width()
        # Clamp selected index
        if self._selected_index >= len(items):
            self._selected_index = max(0, len(items) - 1)
        self._render_key += 1

    def move_up(self) -> None:
        if not self._items:
            return

        if self._selected_index > 0:
            self._selected_index -= 1
        else:
            self._selected_index = len(self._items) - 1
        self._render_key += 1

    def move_down(self) -> None:
        if not self._items:
            return

        if self._selected_index < len(self._items) - 1:
            self._selected_index += 1
        else:
            self._selected_index = 0
        self._render_key += 1

    def render(self) -> Text:
        _ = self._render_key  # Subscribe to changes

        if not self._visible:
            return Text("")

        lines = []

        if not self._items:
            if self._search_enabled:
                dim_color = config.ui.colors.dim
                lines.append(Text("  No matches", style=dim_color))
            result = Text()
            for i, line in enumerate(lines):
                if i > 0:
                    result.append("\n")
                result.append_text(line)
            return result

        total = len(self._items)
        selected = self._selected_index

        # Calculate window
        half_window = self._window_size // 2
        start = max(0, selected - half_window)
        end = min(total, start + self._window_size)

        # Adjust start if we're near the end
        if end - start < self._window_size and start > 0:
            start = max(0, end - self._window_size)

        # Render visible items
        for i in range(start, end):
            item = self._items[i]
            is_selected = i == selected
            lines.append(self._render_row(item, is_selected))

        # Add counter row
        dim_color = config.ui.colors.dim
        counter = Text(f"  ({selected + 1}/{total})", style=dim_color)
        lines.append(counter)

        # Join with newlines
        result = Text()
        for i, line in enumerate(lines):
            if i > 0:
                result.append("\n")
            result.append_text(line)

        return result

    def _render_row(self, item: ListItem[T], is_selected: bool) -> Text:
        selected_color = config.ui.colors.selected
        dim_color = config.ui.colors.dim
        text = Text()

        # Arrow indicator
        if is_selected:
            text.append("→ ", style=f"bold {selected_color}")
        else:
            text.append("  ")

        # Label (truncated if too long, padded to computed width + gap)
        raw_label = item.label
        if len(raw_label) > self._label_width:
            label = raw_label[: self._label_width - 1] + "…"
        else:
            label = raw_label
        label = label.ljust(self._label_width + 4)
        if is_selected:
            text.append(label, style=selected_color)
        else:
            text.append(label)

        # Description (if any)
        if item.description:
            text.append(" ")
            text.append(item.description, style=dim_color)

        return text

    @staticmethod
    def _fuzzy_match(query: str, candidate: str) -> tuple[float, Sequence[int]]:
        q = query.lower()
        c = candidate.lower()
        positions: list[int] = []
        idx = 0
        for char in q:
            idx = c.find(char, idx)
            if idx == -1:
                return (0.0, [])
            positions.append(idx)
            idx += 1

        # Simple scoring: consecutive matches and early matches score higher
        score = float(len(positions))
        if positions and positions[0] == 0:
            score *= 1.2
        groups = 1
        for i in range(1, len(positions)):
            if positions[i] != positions[i - 1] + 1:
                groups += 1
        if len(positions) > 1:
            score *= 1 + (len(positions) - groups + 1) / len(positions)
        return (score, positions)

    @classmethod
    def _fuzzy_filter(cls, query: str, items: list[ListItem[T]]) -> list[ListItem[T]]:
        scored = []
        for item in items:
            # Match against both label and description
            label_score, _ = cls._fuzzy_match(query, item.label)
            desc_score, _ = cls._fuzzy_match(query, item.description)
            best = max(label_score, desc_score * 0.8)
            if best > 0:
                scored.append((best, item))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored]

    def watch__visible(self, visible: bool) -> None:
        if visible:
            self.add_class("-visible")
        else:
            self.remove_class("-visible")
