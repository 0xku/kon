import os
import subprocess
import time
from typing import ClassVar

from rich.spinner import Spinner
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Label

from kon import config
from kon.config import PermissionMode

from .formatting import format_tokens


def format_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def get_git_branch(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return ""


class FileChangesModal(ModalScreen[None]):
    BINDINGS: ClassVar[list] = [("escape", "dismiss_modal", "Close")]

    CSS = """
    FileChangesModal {
        align: center middle;
    }

    #file-changes-container {
        width: 80;
        max-width: 90%;
        max-height: 50%;
        padding: 1 2;
        border: solid grey;
    }

    #file-changes-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #file-changes-list {
        width: 100%;
    }
    """

    def __init__(self, file_changes: dict[str, tuple[int, int]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._file_changes = file_changes

    def compose(self) -> ComposeResult:
        with Vertical(id="file-changes-container"):
            yield Label(self._format_title(), id="file-changes-title")
            yield Label(self._format_file_list(), id="file-changes-list")

    def _format_title(self) -> Text:
        colors = config.ui.colors
        n_files = len(self._file_changes)
        total_added = sum(a for a, _ in self._file_changes.values())
        total_removed = sum(r for _, r in self._file_changes.values())

        result = Text()
        result.append(f"{n_files} file{'s' if n_files != 1 else ''} changed", style="bold")
        result.append("  ")
        result.append(f"+{total_added}", style=f"bold {colors.diff_added}")
        result.append(" ")
        result.append(f"-{total_removed}", style=f"bold {colors.diff_removed}")
        return result

    def _format_file_list(self) -> Text:
        colors = config.ui.colors
        cwd = os.getcwd()

        # Sort by filename for stable display
        entries = sorted(self._file_changes.items(), key=lambda x: x[0])

        # Calculate column widths
        max_added_w = max((len(str(a)) for a, _ in self._file_changes.values()), default=1)
        max_removed_w = max((len(str(r)) for _, r in self._file_changes.values()), default=1)

        result = Text()
        for i, (path, (added, removed)) in enumerate(entries):
            if i > 0:
                result.append("\n")

            # Shorten path: strip cwd prefix, then home prefix
            display_path = path
            if display_path.startswith(cwd + "/"):
                display_path = display_path[len(cwd) + 1 :]
            else:
                display_path = format_path(display_path)

            added_str = f"+{added}".rjust(max_added_w + 1)
            removed_str = f"-{removed}".rjust(max_removed_w + 1)

            result.append(f"  {added_str}", style=colors.diff_added)
            result.append(f" {removed_str}", style=colors.diff_removed)
            result.append(f"  {display_path}", style=colors.dim)

        return result

    def on_click(self, event: events.Click) -> None:
        # Dismiss when clicking anywhere on the modal overlay
        if self.get_widget_at(event.screen_x, event.screen_y)[0] is self:
            self.dismiss()

    def action_dismiss_modal(self) -> None:
        self.dismiss()


class InfoBar(Vertical):
    def __init__(
        self,
        cwd: str,
        model: str,
        context_window: int | None = None,
        thinking_level: str | None = None,
        hide_thinking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cwd = format_path(cwd)
        self._git_branch = get_git_branch(cwd)
        self._model = model
        self._model_provider: str | None = None
        self._context_window = context_window or config.agent.default_context_window
        self._thinking_level = thinking_level or config.llm.default_thinking_level
        self._hide_thinking = hide_thinking
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._context_tokens: int | None = None
        self._file_changes: dict[str, tuple[int, int]] = {}  # path -> (added, removed)
        self._permission_mode = config.permissions.mode
        self._file_changes_text_start: int | None = None
        self._row1_right: Label | None = None
        self._row2_left: Label | None = None
        self._row2_right: Label | None = None
        self.add_class("info-bar")

    @property
    def _label_row1_right(self) -> Label:
        if self._row1_right is None:
            self._row1_right = self.query_one("#info-row1-right", Label)
        return self._row1_right

    @property
    def _label_row2_left(self) -> Label:
        if self._row2_left is None:
            self._row2_left = self.query_one("#info-row2-left", Label)
        return self._row2_left

    @property
    def _label_row2_right(self) -> Label:
        if self._row2_right is None:
            self._row2_right = self.query_one("#info-row2-right", Label)
        return self._row2_right

    def compose(self) -> ComposeResult:
        with Horizontal(id="info-row-1"):
            yield Label(self._format_row1_left(), id="info-cwd")
            yield Label(self._format_row1_right(), id="info-row1-right")
        with Horizontal(id="info-row-2"):
            yield Label(self._format_row2_right(), id="info-row2-left")
            yield Label(self._format_row2_left(), id="info-row2-right")

    def _format_row1_left(self) -> Text:
        result = Text(self._cwd)
        if self._git_branch:
            result.append(" ", style="")
            result.append(f"(⌥ {self._git_branch})", style=config.ui.colors.accent)
        return result

    def _format_row1_right(self) -> Text:
        result = Text()
        parts = []

        # Context size
        if self._context_tokens is not None:
            ctx = f"{format_tokens(self._context_tokens)}/{format_tokens(self._context_window)}"
        else:
            ctx = f"--/{format_tokens(self._context_window)}"
        parts.append(Text(ctx))

        input_t = format_tokens(self._input_tokens)
        output_t = format_tokens(self._output_tokens)
        usage = f"↑{input_t} ↓{output_t}"
        if self._cache_read_tokens > 0:
            usage += f" R{format_tokens(self._cache_read_tokens)}"
        if self._cache_write_tokens > 0:
            usage += f" W{format_tokens(self._cache_write_tokens)}"
        parts.append(Text(usage))

        # Build string with kon separators
        for i, part in enumerate(parts):
            if i > 0:
                result.append(" • ")
            result.append_text(part)

        return result

    def _format_row2_left(self) -> Text:
        model_text = self._model
        if self._model_provider:
            model_text = f"{self._model} ({self._model_provider})"
        result = Text(model_text)
        result.append(f" • {self._thinking_level}")
        return result

    def _format_permission_mode(self) -> Text:
        result = Text()
        if self._permission_mode == "auto":
            result.append("✓✓ auto", style=config.ui.colors.badge.label)
        else:
            result.append("⏸ prompt", style=config.ui.colors.notice)
        return result

    def _format_row2_right(self) -> Text:
        result = self._format_permission_mode()
        self._file_changes_text_start = None
        if not self._file_changes:
            return result

        n_files = len(self._file_changes)
        total_added = sum(a for a, _ in self._file_changes.values())
        total_removed = sum(r for _, r in self._file_changes.values())
        result.append(" • ", style=config.ui.colors.dim)
        self._file_changes_text_start = len(result.plain)
        result.append(f"{n_files} file{'s' if n_files != 1 else ''}")
        result.append(f" +{total_added}", style=config.ui.colors.diff_added)
        result.append(f" -{total_removed}", style=config.ui.colors.diff_removed)
        return result

    def update_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cache_read_tokens += cache_read_tokens
        self._cache_write_tokens += cache_write_tokens
        # Context size is latest turn's full token footprint.
        self._context_tokens = (
            input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
        )
        self._label_row1_right.update(self._format_row1_right())

    def set_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        context_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cache_read_tokens = cache_read_tokens
        self._cache_write_tokens = cache_write_tokens
        self._context_tokens = context_tokens if context_tokens > 0 else None
        self._label_row1_right.update(self._format_row1_right())

    def set_model(self, model: str, provider: str | None = None) -> None:
        self._model = model
        self._model_provider = provider
        self._label_row2_right.update(self._format_row2_left())

    def set_thinking_level(self, thinking_level: str) -> None:
        self._thinking_level = thinking_level
        self._label_row2_right.update(self._format_row2_left())

    def set_thinking_visibility(self, hide_thinking: bool) -> None:
        self._hide_thinking = hide_thinking

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self._permission_mode = mode
        self._label_row2_left.update(self._format_row2_right(), layout=False)

    def update_file_changes(self, path: str, added: int, removed: int) -> None:
        prev_added, prev_removed = self._file_changes.get(path, (0, 0))
        self._file_changes[path] = (prev_added + added, prev_removed + removed)
        self._label_row2_left.update(self._format_row2_right(), layout=False)

    def set_file_changes(self, file_changes: dict[str, tuple[int, int]]) -> None:
        self._file_changes = file_changes
        self._label_row2_left.update(self._format_row2_right(), layout=False)

    def _is_file_changes_click(self, widget: object, x: int) -> bool:
        return (
            bool(self._file_changes)
            and self._file_changes_text_start is not None
            and widget is self._label_row2_left
            and x >= self._file_changes_text_start
        )

    def on_click(self, event: events.Click) -> None:
        widget, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
        if self._is_file_changes_click(widget, event.x):
            event.stop()
            self.app.push_screen(FileChangesModal(self._file_changes))


class QueueDisplay(Vertical):
    MAX_QUEUE = 5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: list[tuple[str, bool]] = []  # (text, is_steer)
        self._content_label: Label | None = None

    def compose(self) -> ComposeResult:
        yield Label("", id="queue-content")

    @property
    def _queue_label(self) -> Label:
        if self._content_label is None:
            self._content_label = self.query_one("#queue-content", Label)
        return self._content_label

    def on_mount(self) -> None:
        self.add_class("-hidden")

    def _truncate_text(self, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if len(text) <= max_width:
            return text
        if max_width <= 3:
            return "." * max_width
        return text[: max_width - 3] + "..."

    def _render_items(self) -> Text:
        dim_color = config.ui.colors.dim
        steer_items = [(text, True) for text, is_steer in self._items if is_steer]
        normal_items = [(text, False) for text, is_steer in self._items if not is_steer]
        ordered = steer_items + normal_items

        content_width = max(0, self.size.width - 2) if self.size.width else 0
        result = Text()
        result.append("Queue", style="bold " + dim_color)
        for text, is_steer in ordered:
            prefix = " ↳ "
            steer_prefix = "[steer] " if is_steer else ""
            available = max(0, content_width - len(prefix) - len(steer_prefix))
            truncated = self._truncate_text(text, available)
            result.append("\n" + prefix, style=dim_color)
            if is_steer:
                result.append(steer_prefix, style=dim_color)
            result.append(truncated, style=dim_color)
        return result

    def update_items(self, items: list[tuple[str, bool]]) -> None:
        self._items = items
        if not items:
            self._queue_label.update("")
            self.add_class("-hidden")
            return

        self.remove_class("-hidden")
        self._queue_label.update(self._render_items())

    def on_resize(self, event: events.Resize) -> None:
        del event
        if not self._items:
            return
        self._queue_label.update(self._render_items())


class StatusLine(Horizontal):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = "idle"
        self._spinner = Spinner("dots")
        self._timer: Timer | None = None
        self._start_time: float | None = None
        self._tool_calls = 0
        self._show_exit_hint = False
        self._streaming_token_count = 0
        self._status_label: Label | None = None
        self._hint_label: Label | None = None
        self.add_class("status-line")

    def compose(self) -> ComposeResult:
        yield Label("", id="status-text")
        yield Label("", id="exit-hint")

    @property
    def _status_text(self) -> Label:
        if self._status_label is None:
            self._status_label = self.query_one("#status-text", Label)
        return self._status_label

    @property
    def _exit_hint_label(self) -> Label:
        if self._hint_label is None:
            self._hint_label = self.query_one("#exit-hint", Label)
        return self._hint_label

    def _render_spinner(self) -> Text:
        spinner_color = config.ui.colors.accent
        dim_color = config.ui.colors.dim
        spinner_text = self._spinner.render(time.time())
        result = Text()
        if isinstance(spinner_text, Text):
            result.append(str(spinner_text), style=spinner_color)
        else:
            result.append(str(spinner_text), style=spinner_color)
        result.append(" Working...", style=config.ui.colors.muted)
        result.append(" (esc to interrupt)", style=dim_color)
        if self._streaming_token_count > 20:
            result.append(f" ↓{self._streaming_token_count!s}", style=dim_color)
        return result

    def _format_complete_status(self) -> Text:
        elapsed = time.time() - self._start_time if self._start_time else 0
        elapsed_str = f"{int(elapsed)}s"
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = round(elapsed % 60)
            elapsed_str = f"{minutes}m {seconds}s"

        dim_color = config.ui.colors.dim
        result = Text()
        status = f"{elapsed_str} • {self._tool_calls}x"
        result.append(status, style=dim_color)
        return result

    def _start_spinner_timer(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(0.15, self._update_spinner)

    def _stop_spinner_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _update_spinner(self) -> None:
        if self._status != "idle":
            self._status_text.update(self._render_spinner(), layout=False)

    def set_status(self, status: str) -> None:
        old_status = self._status
        self._status = status

        if status == "idle":
            self._stop_spinner_timer()
            self._streaming_token_count = 0
            if old_status != "idle" and self._start_time is not None:
                self._status_text.update(self._format_complete_status(), layout=False)
            elif old_status == "idle" and self._start_time is None:
                self._status_text.update("", layout=False)
        else:
            if old_status == "idle":
                self._start_time = time.time()
                self._tool_calls = 0
                self._streaming_token_count = 0
            self._start_spinner_timer()
            self._status_text.update(self._render_spinner(), layout=False)

    def increment_tool_calls(self) -> None:
        self._tool_calls += 1

    def set_streaming_tokens(self, token_count: int) -> None:
        self._streaming_token_count = token_count
        self._update_spinner()

    def show_exit_hint(self) -> None:
        self._show_exit_hint = True
        muted_color = config.ui.colors.muted
        dim_color = config.ui.colors.dim
        text = Text()
        text.append("ctrl+c", style=muted_color)
        text.append(" again to exit", style=dim_color)
        self._exit_hint_label.update(text)

    def show_delete_session_hint(self) -> None:
        muted_color = config.ui.colors.muted
        dim_color = config.ui.colors.dim
        text = Text()
        text.append("ctrl+d", style=muted_color)
        text.append(" again to delete session", style=dim_color)
        self._exit_hint_label.update(text)

    def hide_exit_hint(self) -> None:
        self._show_exit_hint = False
        self._exit_hint_label.update("")

    def reset(self) -> None:
        self._stop_spinner_timer()
        self._start_time = None
        self._tool_calls = 0
        self._show_exit_hint = False
        self._status_text.update("", layout=False)
        self._exit_hint_label.update("")
