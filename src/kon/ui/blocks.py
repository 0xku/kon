from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from rich.style import Style
from rich.text import Span, Text
from textual import events
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Label, Static

from kon import config
from kon.core.types import ImageContent
from kon.diff_display import DIFF_BG_PAD_MARKER
from kon.permissions import ApprovalResponse

from .formatting import (
    find_stable_block_boundary,
    format_bash_command,
    format_markdown,
    format_markdown_block,
    markdown_render_width,
    strip_markdown_for_collapsed_text,
)

_UPDATE_COMMAND = "uv tool upgrade kon-coding-agent"


@dataclass(frozen=True)
class LaunchWarning:
    message: str
    severity: Literal["warning", "error"] = "warning"


def stylize_badge_markers(text: Text, markers: Iterable[str]) -> None:
    badge_style = f"{config.ui.colors.badge.label} bold"
    plain = text.plain
    for marker in markers:
        search_start = 0
        while True:
            start = plain.find(marker, search_start)
            if start == -1:
                break
            text.stylize(badge_style, start, start + len(marker))
            search_start = start + len(marker)


class _StreamingMarkdownMixin:
    """Block-cached markdown streaming.

    The current unfinished line is buffered until a newline arrives. Completed text is
    split at stable block boundaries (blank lines outside code fences). Closed blocks
    are rendered once and cached, so each refresh only re-renders the open tail block,
    coalesced into the next frame. `_flush_streaming` does one full render at the end,
    so the final display never carries streaming artifacts.
    """

    _pending: str
    _completed: str
    _completed_display: Text
    _committed_blocks: list[Text]
    _committed_len: int
    _committed_width: int
    _stream_update_pending: bool
    _stream_finalized: bool

    def _init_streaming(self) -> None:
        self._pending = ""
        self._completed = ""
        self._completed_display = Text()
        self._committed_blocks = []
        self._committed_len = 0
        self._committed_width = 0
        self._stream_update_pending = False
        self._stream_finalized = False

    def _streaming_update_label(self, display: Text) -> None:
        raise NotImplementedError

    def _streaming_pending_style(self) -> str | None:
        return None

    def _refresh_completed_display(self) -> None:
        width = markdown_render_width()
        if width != self._committed_width:  # cached renders are stale after a resize
            self._committed_blocks = []
            self._committed_len = 0
            self._committed_width = width

        boundary = find_stable_block_boundary(self._completed)
        if boundary > self._committed_len:
            block = format_markdown_block(self._completed[self._committed_len : boundary], width)
            # Some source renders to nothing (HTML comments, link reference definitions).
            # An empty entry here would add a stray blank gap to every later join.
            if block.plain:
                self._committed_blocks.append(block)
            self._committed_len = boundary

        tail = self._completed[self._committed_len :]
        parts = [*self._committed_blocks]
        if tail.strip():
            tail_block = format_markdown_block(tail, width)
            if tail_block.plain:
                parts.append(tail_block)
        self._completed_display = Text("\n\n").join(parts) if parts else Text()

    def _render_streaming_display(self) -> Text:
        display = self._completed_display.copy()
        completed_needs_separator = self._completed.endswith("\n") or self._completed.endswith(
            "\r"
        )

        if (
            not self._stream_finalized
            and completed_needs_separator
            and not self._pending
            and display.plain
        ):
            display.append("\n")

        return display

    def _schedule_streaming_update(self) -> None:
        if self._stream_update_pending:
            return
        self._stream_update_pending = True
        self.call_after_refresh(self._flush_streaming_update)  # pyright: ignore[reportAttributeAccessIssue]

    def _flush_streaming_update(self) -> None:
        self._stream_update_pending = False
        if self._stream_finalized:
            # An update scheduled by the last newline can fire after finalize() already
            # put the final render on the label. Don't overwrite it.
            return
        self._refresh_completed_display()
        self._streaming_update_label(self._render_streaming_display())

    def _append_streaming(self, text: str) -> None:
        self._pending += text

        last_nl = self._pending.rfind("\n")
        if last_nl != -1:
            self._completed += self._pending[: last_nl + 1]
            self._pending = self._pending[last_nl + 1 :]
            self._schedule_streaming_update()

    def _flush_streaming(self) -> Text:
        self._stream_finalized = True
        if self._pending:
            self._completed += self._pending
            self._pending = ""
        self._completed_display = format_markdown(self._completed) if self._completed else Text()
        return self._render_streaming_display()


class ThinkingBlock(_StreamingMarkdownMixin, Static):
    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, content: str = "", finalized: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self._finalized = finalized
        self._label: Label | None = None
        self._init_streaming()
        self.add_class("thinking-block")

    def compose(self) -> ComposeResult:
        if self._finalized and self._content and config.ui.collapse_thinking:
            yield Label(self._format_collapsed(), id="thinking-content", markup=False)
        else:
            yield Label(self._content, id="thinking-content", markup=False)

    @property
    def label(self) -> Label:
        if self._label is None:
            self._label = self.query_one("#thinking-content", Label)
        return self._label

    def _format_collapsed(self) -> Text:
        """Show collapsed thinking with configured line count."""
        lines = self._content.strip().split("\n")
        max_lines = self._get_max_lines()
        style = f"{config.ui.colors.subtle} italic"

        if max_lines is None:
            # No truncation — show everything
            text = Text()
            for i, line in enumerate(lines):
                if i > 0:
                    text.append("\n")
                text.append(strip_markdown_for_collapsed_text(line.strip()), style=style)
            return text

        visible = lines[:max_lines]
        text = Text()
        for i, line in enumerate(visible):
            if i > 0:
                text.append("\n")
            text.append(strip_markdown_for_collapsed_text(line.strip()), style=style)

        remaining = len(lines) - max_lines
        if remaining > 0:
            text.append(f" ... ({remaining} more lines)", style=style)
        return text

    @staticmethod
    def _get_max_lines() -> int | None:
        setting = config.ui.thinking_lines
        if setting == "none":
            return None
        return int(setting)

    def _streaming_update_label(self, display: Text) -> None:
        self.label.update(display)

    def _streaming_pending_style(self) -> str | None:
        return f"{config.ui.colors.subtle} italic"

    async def append(self, text: str) -> None:
        self._content += text
        self._append_streaming(text)

    def finalize(self) -> None:
        if self._content and not self._finalized:
            self._finalized = True
            self.label.update(self._flush_streaming())
            self.call_after_refresh(self._do_finalize)

    def _do_finalize(self) -> None:
        if self._content and config.ui.collapse_thinking:
            self.label.update(self._format_collapsed())

    def set_content(self, text: str) -> None:
        self._content = text
        self._finalized = True
        if config.ui.collapse_thinking:
            self.label.update(self._format_collapsed())
        else:
            self.label.update(text)


class ContentBlock(_StreamingMarkdownMixin, Static):
    # TODO: Consider switching to Textual's Markdown widget + MarkdownStream.write() for
    # incremental rendering during streaming. This would eliminate the visual reflow when
    # finalize() converts plain text to markdown. The tradeoff: our custom Rich-based
    # formatting (CustomMarkdown with LeftJustifiedHeading, PlainListItem, PlainCodeBlock)
    # is incompatible with Textual's Markdown pipeline, so we'd need to reimplement those
    # customizations using Textual's theming/CSS system. See toad and mistral-vibe for
    # reference implementations using MarkdownStream.

    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, content: str = "", finalized: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self._finalized = finalized
        self._label: Label | None = None
        self._init_streaming()
        self.add_class("content-block")

    def compose(self) -> ComposeResult:
        if self._finalized and self._content:
            yield Label(format_markdown(self._content), id="content-text", markup=False)
        else:
            yield Label(self._content, id="content-text", markup=False)

    @property
    def label(self) -> Label:
        if self._label is None:
            self._label = self.query_one("#content-text", Label)
        return self._label

    def _streaming_update_label(self, display: Text) -> None:
        self.label.update(display)

    async def append(self, text: str) -> None:
        self._content += text
        self._append_streaming(text)

    def finalize(self) -> None:
        if self._content and not self._finalized:
            self._finalized = True
            self.label.update(self._flush_streaming())
            self.call_after_refresh(self._do_finalize)

    def _do_finalize(self) -> None:
        if self._content:
            self.label.update(format_markdown(self._content))

    def set_content(self, text: str) -> None:
        self._content = text
        self._finalized = True
        self.label.update(format_markdown(self._content))


class ToolBlock(Static):
    """
    Format:
    TOOL_NAME call_msg
    truncated output
    """

    ALLOW_SELECT = True
    can_focus = False
    MAX_HEADER_LINES = 2

    def __init__(
        self,
        name: str = "",
        call_msg: str | None = None,
        icon: str = "→",
        expanded: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._icon = icon
        self._call_msg = call_msg
        self._ui_summary: str | None = None
        self._ui_details: str | None = None
        self._ui_details_full: str | None = None
        self._images: list[ImageContent] | None = None
        self._result_markup: bool = True
        self._expanded: bool = expanded
        self._success: bool | None = None
        self._awaiting_approval: bool = False
        self._approval_preview: str | None = None
        self._approval_selection: ApprovalResponse = ApprovalResponse.APPROVE
        self.add_class("tool-block")
        self._set_state(None)

    def compose(self) -> ComposeResult:
        yield Label(self._format_header(), id="tool-header")
        yield Label("", id="tool-output", classes="tool-output -hidden")

    def _format_header(self, truncate: bool = True) -> Text:
        colors = config.ui.colors
        result = Text()
        formatted_name = self._name

        success_style = Style(color=colors.muted, bold=True)
        icon_style: str | Style = success_style
        name_style: str | Style = success_style
        if self._success is None:
            icon_style = colors.running
            name_style = colors.running
        elif self._success is False:
            icon_style = colors.failed
            name_style = colors.failed
        elif self._success is True and config.ui.colored_tool_badge:
            badge_style = Style(color=colors.badge.label, bold=True)
            icon_style = badge_style
            name_style = badge_style

        if self._awaiting_approval:
            result.append(
                " △ Permission required ",
                style=Style(bgcolor=colors.notice, color=colors.bg, bold=True),
            )
            result.append("\n\n")

        result.append(f"{self._icon} ", style=icon_style)
        result.append(formatted_name, style=name_style)

        if self._call_msg:
            result.append(" ")
            result.append_text(self._format_call_msg(truncate=truncate))

        if self._ui_summary:
            result.append(" ")
            summary = self._render_markup_safe(self._ui_summary)
            result.append_text(summary)

        if self._success is None and not self._awaiting_approval and not self._call_msg:
            result.append(" ...", style=colors.dim)

        return result

    def _format_call_msg(self, truncate: bool = True) -> Text:
        if not self._call_msg:
            return Text()

        if truncate:
            lines = self._call_msg.split("\n")
            if len(lines) > self.MAX_HEADER_LINES:
                content = "\n".join(lines[: self.MAX_HEADER_LINES])
                content += f"\n... ({len(lines) - self.MAX_HEADER_LINES} more lines)"
            else:
                content = self._call_msg
        else:
            content = self._call_msg

        if self._name == "bash":
            return format_bash_command(content)

        rendered = self._render_markup_safe(content)
        return Text(rendered.plain, style=config.ui.colors.muted)

    def _normalize_markup_style(self, style: str | Style) -> Style:
        parsed = Style.parse(style) if isinstance(style, str) else style
        colors = config.ui.colors
        color_name = parsed.color.name if parsed.color is not None else None
        color = None
        if parsed.dim and parsed.color is None:
            color = colors.subtle
        elif color_name == "red":
            color = colors.diff_removed
        elif color_name == "green":
            color = colors.diff_added
        elif color_name == "yellow":
            color = colors.notice

        if color is None:
            return parsed
        return Style(
            color=color, bold=parsed.bold, italic=parsed.italic, underline=parsed.underline
        )

    def _render_markup_safe(self, content: str) -> Text:
        try:
            text = Text.from_markup(content)
        except Exception:
            return Text(content, style=config.ui.colors.subtle)

        spans: list[Span] = []
        for span in text.spans:
            try:
                style = self._normalize_markup_style(span.style)
            except Exception:
                return Text(content, style=config.ui.colors.subtle)
            spans.append(Span(span.start, span.end, style))
        text.spans = spans

        if not text.spans:
            text.stylize(config.ui.colors.subtle)

        return text

    def _pad_diff_backgrounds(self, text: Text, width: int) -> Text:
        if DIFF_BG_PAD_MARKER not in text.plain or width <= 0:
            return text

        result = Text()
        lines = text.split("\n", allow_blank=True)
        for index, line in enumerate(lines):
            marker_pos = line.plain.find(DIFF_BG_PAD_MARKER)
            if marker_pos != -1:
                line = line.copy()
                marker_end = marker_pos + len(DIFF_BG_PAD_MARKER)
                marker_spans = [span for span in line.spans if span.start <= marker_pos < span.end]
                marker_style = marker_spans[0].style if marker_spans else None
                line.plain = line.plain[:marker_pos] + line.plain[marker_end:]
                line.spans = [
                    span
                    for span in line.spans
                    if not (span.start >= marker_pos and span.end <= marker_end)
                ]
                padding = max(0, width - len(line.plain))
                if padding:
                    line.append(" " * padding, style=marker_style)
            if index > 0:
                result.append("\n")
            result.append_text(line)
        return result

    def _set_state(self, success: bool | None) -> None:
        self.remove_class("-pending", "-success", "-error", "-approval")
        if success is None:
            if self._awaiting_approval:
                self.add_class("-approval")
            else:
                self.add_class("-pending")
        elif success:
            self.add_class("-success")
        else:
            self.add_class("-error")

    def show_approval(
        self, preview: str | None = None, selected: ApprovalResponse | None = None
    ) -> None:
        self._awaiting_approval = True
        self._approval_preview = preview
        if selected is not None:
            self._approval_selection = selected
        self._set_state(None)
        self.query_one("#tool-header", Label).update(self._format_header())
        self._render_approval_output()

    def update_approval_selection(self, selected: ApprovalResponse) -> None:
        if not self._awaiting_approval:
            return
        self._approval_selection = selected
        self._render_approval_output()

    def _render_approval_output(self) -> None:
        output = self.query_one("#tool-output", Label)
        self.remove_class("-with-details")
        output.remove_class("-hidden")
        output.remove_class("-details")

        content = Text()
        if self._approval_preview:
            content.append_text(self._render_markup_safe(self._approval_preview))
            content.append("\n\n")
        content.append_text(self._format_approval_controls(self._approval_selection))
        output.update(content)

    def hide_approval(self) -> None:
        self._awaiting_approval = False
        self._approval_preview = None
        self._approval_selection = ApprovalResponse.APPROVE
        self._set_state(None)
        self.query_one("#tool-header", Label).update(self._format_header())
        output = self.query_one("#tool-output", Label)
        self.remove_class("-with-details")
        output.remove_class("-details")
        output.add_class("-hidden")
        output.update(Text(""))

    def _format_approval_controls(
        self, selected: ApprovalResponse = ApprovalResponse.APPROVE
    ) -> Text:
        colors = config.ui.colors
        text = Text()
        # The non-selected button uses the dim panel_alt background; the
        # selected one gets the accent. Direct y/n keys submit immediately;
        # left/right move the highlight; enter submits the highlight.
        approve_selected = selected == ApprovalResponse.APPROVE
        approve_style = Style(
            bgcolor=colors.accent if approve_selected else colors.panel_alt,
            color=colors.bg if approve_selected else colors.dim,
            bold=True,
        )
        deny_style = Style(
            bgcolor=colors.accent if not approve_selected else colors.panel_alt,
            color=colors.bg if not approve_selected else colors.dim,
            bold=True,
        )
        text.append("[y] approve ", style=approve_style)
        text.append("  ")
        text.append("[n] deny ", style=deny_style)
        text.append("  ")
        text.append("(← → enter)", style=Style(color=colors.dim))
        return text

    def update_call_msg(self, call_msg: str) -> None:
        self._call_msg = call_msg
        self.query_one("#tool-header", Label).update(self._format_header())

    def set_result(
        self,
        ui_summary: str | None,
        ui_details: str | None,
        success: bool,
        markup: bool = True,
        ui_details_full: str | None = None,
        images: list[ImageContent] | None = None,
    ) -> None:
        self._ui_summary = ui_summary
        self._ui_details = ui_details
        self._ui_details_full = ui_details_full
        self._images = images
        self._result_markup = markup
        self._success = success
        self._awaiting_approval = False
        self._set_state(success)
        self._render_result_output()
        self.query_one("#tool-header", Label).update(self._format_header())

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self._render_result_output()

    def on_resize(self, event: events.Resize) -> None:
        del event
        if self._ui_details or self._ui_details_full:
            self._render_result_output()

    def _render_result_output(self) -> None:
        output = self.query_one("#tool-output", Label)
        ui_details = (
            self._ui_details_full if self._expanded and self._ui_details_full else self._ui_details
        )
        if ui_details:
            rendered = (
                self._render_markup_safe(ui_details)
                if self._result_markup
                else Text(ui_details, style=config.ui.colors.subtle)
            )
            is_diff_output = DIFF_BG_PAD_MARKER in rendered.plain
            rendered = self._pad_diff_backgrounds(rendered, output.size.width or self.size.width)
            # Detail blocks need a 1-line gap; drop compact spacing that was
            # applied before we knew this tool would have output.
            self.remove_class("-compact")
            self.add_class("-with-details")
            output.remove_class("-hidden")
            output.remove_class("-details")
            if is_diff_output:
                output.add_class("-diff-output")
            else:
                output.remove_class("-diff-output")
            output.update(rendered)
        elif self._images:
            image_count = len(self._images)
            image_label = "image" if image_count == 1 else "images"
            rendered = Text(f"Attached {image_count} {image_label}", style=config.ui.colors.subtle)
            self.remove_class("-compact")
            self.add_class("-with-details")
            output.remove_class("-hidden")
            output.remove_class("-details")
            output.remove_class("-diff-output")
            output.update(rendered)
        else:
            output.update(Text(""))
            self.remove_class("-with-details")
            output.remove_class("-details")
            output.remove_class("-diff-output")
            output.add_class("-hidden")


class UserBlock(Static):
    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, content: str = "", highlighted_skill: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self._highlighted_skill = highlighted_skill
        self.add_class("user-block")
        if highlighted_skill:
            self.add_class("skill-trigger-message")

    def compose(self) -> ComposeResult:
        text = Text()
        if self._highlighted_skill:
            text.append(self._content)
            stylize_badge_markers(text, [f"[{self._highlighted_skill}]", "[query]"])
        else:
            text.append(self._content)

        yield Label(text)


class HandoffLinkBlock(Static):
    ALLOW_SELECT = True
    can_focus = False

    def __init__(
        self,
        label: str,
        target_session_id: str,
        query: str,
        direction: Literal["back", "forward"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._target_session_id = target_session_id
        self._query = query
        self._direction: Literal["back", "forward"] = direction
        self.add_class("handoff-link-block")

    def compose(self) -> ComposeResult:
        link_text = f"{self._target_session_id[:8]} (click to open)"
        handoff_line = f"{self._label} → {link_text}"
        text = Text(f"[handoff]\n{handoff_line}\n\n[query]\n{self._query}")
        stylize_badge_markers(text, ("[handoff]", "[query]"))

        link_start = text.plain.find(link_text)
        if link_start != -1:
            text.stylize(
                f"{config.ui.colors.notice} underline", link_start, link_start + len(link_text)
            )

        yield Label(text)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        if not self._target_session_id:
            return
        self.post_message(
            self.LinkSelected(self, self._target_session_id, self._query, self._direction)
        )

    class LinkSelected(Message):
        def __init__(
            self,
            block: "HandoffLinkBlock",
            target_session_id: str,
            query: str,
            direction: Literal["back", "forward"],
        ) -> None:
            super().__init__()
            self.block = block
            self.target_session_id = target_session_id
            self.query = query
            self.direction = direction


class UpdateAvailableBlock(Static):
    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, latest_version: str, changelog_url: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._latest_version = latest_version
        self._changelog_url = changelog_url
        self.add_class("update-available-block")

    def compose(self) -> ComposeResult:
        notice_color = config.ui.colors.notice
        dim_color = config.ui.colors.dim
        accent_color = config.ui.colors.accent

        text = Text()
        text.append("Update Available", style=f"{notice_color} bold")
        text.append("\n", style=dim_color)
        text.append(f"New version {self._latest_version} is available. ", style=dim_color)
        text.append("Run: ", style=dim_color)
        text.append(_UPDATE_COMMAND, style=accent_color)

        if self._changelog_url:
            text.append("\n", style=dim_color)
            text.append("Changelog: ", style=dim_color)
            text.append(self._changelog_url, style=accent_color)

        yield Label(text)


class LaunchWarningsBlock(Static):
    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, warnings: list[LaunchWarning], **kwargs) -> None:
        super().__init__(**kwargs)
        self._warnings = warnings
        self.add_class("launch-warnings-block")

    def compose(self) -> ComposeResult:
        notice_color = config.ui.colors.notice
        error_color = config.ui.colors.error
        dim_color = config.ui.colors.dim

        text = Text()
        text.append("Launch Warnings", style=f"{notice_color} bold")

        for warning in self._warnings:
            bullet = "\n✗ " if warning.severity == "error" else "\n! "
            style = error_color if warning.severity == "error" else dim_color
            text.append(bullet, style=style)
            text.append(warning.message, style=style)

        yield Label(text)
