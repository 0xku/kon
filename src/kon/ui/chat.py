import time
from typing import Literal

from rich.spinner import Spinner
from rich.text import Text
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widgets import Label

from kon import config

from .blocks import (
    ContentBlock,
    HandoffLinkBlock,
    LaunchWarning,
    LaunchWarningsBlock,
    ThinkingBlock,
    ToolBlock,
    UpdateAvailableBlock,
    UserBlock,
    stylize_badge_markers,
)

MAX_CHILDREN = 300
PRUNE_TO = 200


class ChatLog(VerticalScroll):
    can_focus = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_block: ThinkingBlock | ContentBlock | None = None
        self._tool_blocks: dict[str, ToolBlock] = {}
        self._anchor_released: bool = False
        self._last_status_label: Label | None = None
        self._spinner_label: Label | None = None
        self._spinner: Spinner | None = None
        self._spinner_timer: Timer | None = None

    def on_mount(self) -> None:
        self.anchor()

    def _scroll_if_anchored(self, animate: bool = False) -> None:
        if not self._anchor_released:
            self.scroll_end(animate=animate)
            return

        max_y = self.max_scroll_y
        current_y = self.scroll_y

        if abs(max_y - current_y) < 3:
            self._anchor_released = False
            self.scroll_end(animate=animate)

    def _prune_if_needed(self) -> None:
        children = list(self.children)
        if len(children) <= MAX_CHILDREN:
            return
        to_remove = children[: len(children) - PRUNE_TO]
        active_tool_ids = {tid for tid, block in self._tool_blocks.items() if block in to_remove}
        for tid in active_tool_ids:
            del self._tool_blocks[tid]
        if self._last_status_label in to_remove:
            self._last_status_label = None
        self.call_after_refresh(lambda: self.remove_children(to_remove))

    async def remove_all_children(self) -> None:
        self._stop_spinner()
        children = list(self.children)
        if children:
            await self.remove_children(children)
        self._tool_blocks.clear()
        self._current_block = None
        self._last_status_label = None

    def on_click(self, event) -> None:
        event.stop()
        from .input import InputBox

        app = self.app
        input_box = app.query_one("#input-box", InputBox)
        input_box.focus()

    def _is_last_child_status(self) -> bool:
        if self._last_status_label is None:
            return False
        children = list(self.children)
        if not children:
            return False
        return children[-1] is self._last_status_label

    def show_status(self, message: str) -> None:
        self._stop_spinner()
        info_color = config.ui.colors.info
        text = Text(f"✓ {message}", style=info_color)

        # If our tracked status label is still the last child, update it
        if self._is_last_child_status() and self._last_status_label is not None:
            self._last_status_label.update(text)
            self._scroll_if_anchored(animate=False)
            return

        # Otherwise create a new status label
        label = Label(text)
        label.add_class("info-message")
        self.mount(label)
        self._last_status_label = label
        self._scroll_if_anchored(animate=False)

    def show_spinner_status(self, message: str) -> None:
        self._stop_spinner()
        self._spinner = Spinner("dots")
        self._spinner_label = Label(self._render_spinner_text(message))
        self._spinner_label.add_class("info-message")
        self.mount(self._spinner_label)
        self._last_status_label = self._spinner_label
        self._spinner_timer = self.set_interval(0.15, lambda: self._tick_spinner(message))
        self._scroll_if_anchored(animate=False)

    def _render_spinner_text(self, message: str) -> Text:
        info_color = config.ui.colors.info
        spinner_text = self._spinner.render(time.time()) if self._spinner else ""
        result = Text()
        result.append(str(spinner_text), style=info_color)
        result.append(f" {message}", style=info_color)
        return result

    def _tick_spinner(self, message: str) -> None:
        if self._spinner_label is not None and self._spinner is not None:
            self._spinner_label.update(self._render_spinner_text(message))

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self._spinner = None
        self._spinner_label = None

    def add_session_info(self, version: str) -> None:
        info_text = Text()
        accent = config.ui.colors.accent
        dim = config.ui.colors.dim
        muted = config.ui.colors.muted

        info_text.append("kon", style=f"{accent} bold")
        info_text.append(f" v{version}\n", style=dim)

        def append_hint(key: str, description: str, trailing_newline: bool = True) -> None:
            info_text.append(key, style=dim)
            info_text.append(f" {description}", style=muted)
            if trailing_newline:
                info_text.append("\n", style=muted)

        append_hint("escape", "to interrupt")
        append_hint("ctrl+c", "to clear input")
        append_hint("ctrl+c twice", "to exit")
        append_hint("ctrl+t", "to show or hide thinking")
        append_hint("shift+tab", "to cycle thinking level")
        append_hint("/", "for commands")
        append_hint("@", "to search files inline")
        append_hint("tab", "to complete paths")
        append_hint("up and down", "for prompt history")
        append_hint("shift+enter", "for newline", trailing_newline=False)

        info_label = Label(info_text)
        info_label.add_class("session-info")
        self.mount(info_label, before=0)

    def add_loaded_resources(self, context_paths: list[str], skill_paths: list[str]) -> None:
        if not context_paths and not skill_paths:
            return

        dim_color = config.ui.colors.dim
        notice_color = config.ui.colors.notice
        text = Text()

        if context_paths:
            text.append("[Context]\n", style=notice_color)
            for path in context_paths:
                text.append(f"  {path}\n", style=dim_color)

        if skill_paths:
            if context_paths:
                text.append("\n")
            text.append("[Skills]\n", style=notice_color)
            for path in skill_paths:
                text.append(f"  {path}\n", style=dim_color)

        # Remove trailing newline
        text.rstrip()

        label = Label(text)
        label.add_class("info-message")
        label.add_class("loaded-resources")
        self.mount(label)

    def add_launch_warnings(self, warnings: list[LaunchWarning]) -> None:
        if not warnings:
            return
        self.mount(LaunchWarningsBlock(warnings))
        self._scroll_if_anchored(animate=False)

    def add_user_message(self, content: str, highlighted_skill: str | None = None) -> UserBlock:
        block = UserBlock(content, highlighted_skill=highlighted_skill)
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._prune_if_needed()
        return block

    def add_handoff_link_message(
        self, label: str, target_session_id: str, query: str, direction: Literal["back", "forward"]
    ) -> HandoffLinkBlock:
        block = HandoffLinkBlock(
            label=label, target_session_id=target_session_id, query=query, direction=direction
        )
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._prune_if_needed()
        return block

    def add_update_available_message(
        self, latest_version: str, changelog_url: str | None = None
    ) -> UpdateAvailableBlock:
        block = UpdateAvailableBlock(latest_version, changelog_url=changelog_url)
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._prune_if_needed()
        return block

    def start_thinking(self) -> ThinkingBlock:
        block = ThinkingBlock()
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._current_block = block
        return block

    def add_thinking(self, content: str) -> ThinkingBlock:
        block = ThinkingBlock(content, finalized=True)
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        return block

    def start_content(self) -> ContentBlock:
        block = ContentBlock()
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._current_block = block
        return block

    def add_content(self, content: str) -> ContentBlock:
        block = ContentBlock(content, finalized=True)
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        return block

    def start_tool(self, name: str, tool_id: str, call_msg: str | None = None) -> ToolBlock:
        block = ToolBlock(name=name, call_msg=call_msg)
        self.mount(block)
        self._scroll_if_anchored(animate=False)
        self._tool_blocks[tool_id] = block
        return block

    async def append_to_current(self, text: str) -> None:
        if self._current_block:
            await self._current_block.append(text)
            self._scroll_if_anchored(animate=False)

    def set_block_content(self, text: str) -> None:
        if self._current_block:
            self._current_block.set_content(text)
            self._scroll_if_anchored(animate=False)

    def set_tool_result(
        self, tool_id: str, content: str, success: bool, markup: bool = True
    ) -> None:
        block = self._tool_blocks.get(tool_id)
        if block:
            block.set_result(content, success, markup=markup)
            self._scroll_if_anchored(animate=False)

    def update_tool_call_msg(self, tool_id: str, call_msg: str) -> None:
        block = self._tool_blocks.get(tool_id)
        if block:
            block.update_call_msg(call_msg)
            self._scroll_if_anchored(animate=False)

    def show_tool_approval(self, tool_id: str) -> None:
        block = self._tool_blocks.get(tool_id)
        if block:
            block.show_approval()
            self._scroll_if_anchored(animate=False)

    def hide_tool_approval(self, tool_id: str) -> None:
        block = self._tool_blocks.get(tool_id)
        if block:
            block.hide_approval()
            self._scroll_if_anchored(animate=False)

    def end_block(self) -> None:
        # Finalize content/thinking blocks to render markdown once
        if isinstance(self._current_block, ContentBlock | ThinkingBlock):
            self._current_block.finalize()
        self._current_block = None

    def add_compaction_message(self, tokens_before: int) -> None:
        self._stop_spinner()
        # Remove the "Auto-compacting..." status if it's still showing
        if self._is_last_child_status() and self._last_status_label is not None:
            self._last_status_label.remove()
            self._last_status_label = None

        dim_color = config.ui.colors.dim
        token_str = f"{tokens_before:,}"

        text = Text(f"[compaction] Compacted from {token_str} tokens", style=dim_color)
        stylize_badge_markers(text, ("[compaction]",))

        label = Label(text)
        label.add_class("compaction-message")
        self.mount(label)
        self._scroll_if_anchored(animate=False)

    def add_aborted_message(self, message: str = "Interrupted by user") -> None:
        error_color = config.ui.colors.error
        text = Text(message, style=error_color)
        label = Label(text)
        label.add_class("aborted-message")
        self.mount(label)
        self._scroll_if_anchored(animate=False)

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        info_color = config.ui.colors.info
        error_color = config.ui.colors.error
        notice_color = config.ui.colors.notice

        cleaned_message = message.strip()
        if not cleaned_message:
            cleaned_message = (
                "Unknown error (no details provided)." if error else "No details provided."
            )

        style = info_color
        prefix = "✓ "
        if warning:
            style = notice_color
            prefix = "! "
        if error:
            style = error_color
            prefix = "✗ "

        text = Text(f"{prefix}{cleaned_message}", style=style)
        label = Label(text)
        label.add_class("info-message")
        self.mount(label)
        self._scroll_if_anchored(animate=False)

    def clear_tool_blocks(self) -> None:
        self._tool_blocks.clear()
