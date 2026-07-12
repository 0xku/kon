"""Pending/steer message queue state and its QueueDisplay rendering."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from kon.core.types import ImageContent

from .input import InputBox
from .widgets import QueueDisplay

QueuedPrompt = tuple[str, str] | tuple[str, str, list[ImageContent]]


def _unpack_queue_item(item: QueuedPrompt) -> tuple[str, str, list[ImageContent]]:
    if len(item) == 2:
        display, query = item
        return display, query, []
    return item


class QueueUIMixin:
    """Manages the two message queues: steer (injected mid-run) and pending (run next)."""

    _pending_queue: deque[QueuedPrompt]
    _steer_queue: deque[QueuedPrompt]
    _queue_selection: tuple[bool, int] | None
    _queue_editing: tuple[bool, int, QueuedPrompt] | None

    if TYPE_CHECKING:
        query_one: Any

    def _queue_items(self) -> list[tuple[bool, int, str, str, list[ImageContent]]]:
        steer = [
            (True, index, display, query, images)
            for index, item in enumerate(self._steer_queue)
            for display, query, images in [_unpack_queue_item(item)]
        ]
        pending = [
            (False, index, display, query, images)
            for index, item in enumerate(self._pending_queue)
            for display, query, images in [_unpack_queue_item(item)]
        ]
        return steer + pending

    def _selected_queue_flat_index(self) -> int | None:
        if self._queue_selection is None:
            return None
        for flat_index, (is_steer, index, _, _, _) in enumerate(self._queue_items()):
            if self._queue_selection == (is_steer, index):
                return flat_index
        return None

    def _set_queue_selection_by_flat_index(self, flat_index: int | None) -> None:
        items = self._queue_items()
        if flat_index is None or flat_index < 0 or flat_index >= len(items):
            self._queue_selection = None
        else:
            is_steer, index, _, _, _ = items[flat_index]
            self._queue_selection = (is_steer, index)
        self._update_queue_display()

    def _update_queue_display(self) -> None:
        queue_display = self.query_one("#queue-display", QueueDisplay)
        steer_items = [(_unpack_queue_item(item)[0], True) for item in self._steer_queue]
        normal_items = [(_unpack_queue_item(item)[0], False) for item in self._pending_queue]
        selected = self._selected_queue_flat_index()
        editing = None

        if self._queue_editing:
            is_steer, index, original = self._queue_editing
            target_items = steer_items if is_steer else normal_items
            edit_index = min(index, len(target_items))
            target_items.insert(edit_index, (original[0], is_steer))
            editing = edit_index if is_steer else len(steer_items) + edit_index
            selected = editing

        queue_display.update_items(steer_items + normal_items, selected=selected, editing=editing)

    def select_queue_from_input(self, direction: int) -> bool:
        if self._queue_editing is not None:
            return False
        items = self._queue_items()
        if not items:
            return False
        current = self._selected_queue_flat_index()
        if current is None:
            next_index = len(items) - 1 if direction < 0 else 0
        else:
            next_index = current + direction
        if next_index < 0 or next_index >= len(items):
            self._set_queue_selection_by_flat_index(None)
            return False
        self._set_queue_selection_by_flat_index(next_index)
        return True

    def delete_selected_queue_item(self) -> bool:
        if self._queue_editing is not None or self._queue_selection is None:
            return False
        is_steer, index = self._queue_selection
        queue = self._steer_queue if is_steer else self._pending_queue
        if index >= len(queue):
            self._set_queue_selection_by_flat_index(None)
            return False
        del queue[index]
        items = self._queue_items()
        if items:
            self._set_queue_selection_by_flat_index(min(index, len(items) - 1))
        else:
            self._set_queue_selection_by_flat_index(None)
        return True

    def start_queue_edit(self) -> bool:
        if self._queue_selection is None or self._queue_editing is not None:
            return False
        is_steer, index = self._queue_selection
        queue = self._steer_queue if is_steer else self._pending_queue
        if index >= len(queue):
            self._set_queue_selection_by_flat_index(None)
            return False
        original = queue[index]
        del queue[index]
        self._queue_editing = (is_steer, index, original)
        input_box = self.query_one("#input-box", InputBox)
        input_box.clear(reset_pastes=False)
        input_box.insert(original[1])
        input_box.focus()
        self._update_queue_display()
        return True

    def finish_queue_edit(
        self, display_text: str, query_text: str, images: list[ImageContent] | None = None
    ) -> bool:
        if self._queue_editing is None:
            return False
        is_steer, index, original = self._queue_editing
        queue = self._steer_queue if is_steer else self._pending_queue
        original_images = _unpack_queue_item(original)[2]
        item_images = images or original_images
        item: QueuedPrompt = (
            (display_text, query_text, item_images) if item_images else (display_text, query_text)
        )
        queue.insert(min(index, len(queue)), item)
        self._queue_editing = None
        self._queue_selection = (is_steer, min(index, len(queue) - 1))
        self._update_queue_display()
        return True

    def cancel_queue_edit(self) -> bool:
        if self._queue_editing is None:
            return False
        is_steer, index, original = self._queue_editing
        queue = self._steer_queue if is_steer else self._pending_queue
        queue.insert(min(index, len(queue)), original)
        self._queue_editing = None
        self._queue_selection = (is_steer, min(index, len(queue) - 1))
        self._update_queue_display()
        input_box = self.query_one("#input-box", InputBox)
        input_box.clear(reset_pastes=False)
        return True
