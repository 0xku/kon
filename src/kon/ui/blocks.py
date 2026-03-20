from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Label, Static

from kon import config

from .formatting import format_markdown

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
    """Line-buffered incremental markdown rendering for streaming blocks.

    Buffers incoming chunks until a newline arrives, then re-renders all
    completed lines as a single markdown document. Rendering the full
    accumulated text each time preserves block-level structure (paragraph
    spacing, headings, etc.) that would be lost if chunks were rendered
    independently and stitched together.
    """

    _pending: str
    _completed: str

    def _init_streaming(self) -> None:
        self._pending = ""
        self._completed = ""

    def _append_streaming(self, text: str) -> Text:
        self._pending += text

        last_nl = self._pending.rfind("\n")
        if last_nl != -1:
            self._completed += self._pending[: last_nl + 1]
            self._pending = self._pending[last_nl + 1 :]

        display = Text()
        if self._completed:
            display.append_text(format_markdown(self._completed))
        if self._pending:
            if self._completed:
                display.append("\n")
            display.append(self._pending)
        return display


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
        if self._finalized and self._content:
            yield Label(self._format_collapsed(), id="thinking-content", markup=False)
        else:
            yield Label(self._content, id="thinking-content", markup=False)

    @property
    def label(self) -> Label:
        if self._label is None:
            self._label = self.query_one("#thinking-content", Label)
        return self._label

    def _format_collapsed(self) -> Text:
        """Show only the first line with a truncation indicator."""
        lines = self._content.strip().split("\n")
        first_line = lines[0].strip() if lines else ""
        style = f"{config.ui.colors.dim} italic"
        text = Text(first_line, style=style)
        if len(lines) > 1:
            text.append(f" ... ({len(lines) - 1} more lines)", style=style)
        return text

    async def append(self, text: str) -> None:
        self._content += text
        self.label.update(self._append_streaming(text))

    def finalize(self) -> None:
        if self._content and not self._finalized:
            self._finalized = True
            self.call_after_refresh(self._do_finalize)

    def _do_finalize(self) -> None:
        if self._content:
            self.label.update(self._format_collapsed())

    def set_content(self, text: str) -> None:
        self._content = text
        self._finalized = True
        self.label.update(self._format_collapsed())


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

    async def append(self, text: str) -> None:
        self._content += text
        self.label.update(self._append_streaming(text))

    def finalize(self) -> None:
        if self._content and not self._finalized:
            self._finalized = True
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
        self, name: str = "", call_msg: str | None = None, icon: str = "→", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._icon = icon
        self._call_msg = call_msg
        self._ui_summary: str | None = None
        self._ui_details: str | None = None
        self._success: bool | None = None
        self._awaiting_approval: bool = False
        self.add_class("tool-block")
        self._set_state(None)

    def compose(self) -> ComposeResult:
        yield Label(self._format_header(), id="tool-header")
        yield Label("", id="tool-output", classes="tool-output -hidden")

    def _format_header(self, truncate: bool = True) -> Text:
        colors = config.ui.colors
        result = Text()
        formatted_name = " ".join(word.capitalize() for word in self._name.split("_"))

        icon_style = colors.dim
        name_style = colors.dim
        if self._success is None:
            icon_style = colors.running
            name_style = colors.running
        elif self._success is False:
            icon_style = colors.failed
            name_style = colors.failed

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

        rendered = self._render_markup_safe(content)
        return Text(rendered.plain, style=config.ui.colors.muted)

    def _render_markup_safe(self, content: str) -> Text:
        try:
            text = Text.from_markup(content)
        except Exception:
            return Text(content)

        for span in text.spans:
            style = span.style
            if isinstance(style, str):
                try:
                    Style.parse(style)
                except Exception:
                    return Text(content)

        return text

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

    def show_approval(self, preview: str | None = None) -> None:
        self._awaiting_approval = True
        self._set_state(None)
        self.query_one("#tool-header", Label).update(self._format_header())
        output = self.query_one("#tool-output", Label)
        self.remove_class("-with-details")
        output.remove_class("-hidden")
        output.remove_class("-details")

        content = Text()
        if preview:
            content.append_text(self._render_markup_safe(preview))
            content.append("\n\n")
        content.append_text(self._format_approval_controls())
        output.update(content)

    def hide_approval(self) -> None:
        self._awaiting_approval = False
        self._set_state(None)
        self.query_one("#tool-header", Label).update(self._format_header())
        output = self.query_one("#tool-output", Label)
        self.remove_class("-with-details")
        output.remove_class("-details")
        output.add_class("-hidden")
        output.update(Text(""))

    def _format_approval_controls(self) -> Text:
        accent_style = f"{config.ui.colors.accent} bold"
        dim_style = config.ui.colors.dim
        text = Text()
        text.append("[y]", style=accent_style)
        text.append(" approve  ", style=dim_style)
        text.append("[n]", style=accent_style)
        text.append(" deny", style=dim_style)
        return text

    def update_call_msg(self, call_msg: str) -> None:
        self._call_msg = call_msg
        self.query_one("#tool-header", Label).update(self._format_header())

    def set_result(
        self, ui_summary: str | None, ui_details: str | None, success: bool, markup: bool = True
    ) -> None:
        self._ui_summary = ui_summary
        self._ui_details = ui_details
        self._success = success
        self._awaiting_approval = False
        self._set_state(success)

        output = self.query_one("#tool-output", Label)
        if ui_details:
            rendered = self._render_markup_safe(ui_details) if markup else Text(ui_details)
            # Detail blocks need a 1-line gap; drop compact spacing that was
            # applied before we knew this tool would have output.
            self.remove_class("-compact")
            self.add_class("-with-details")
            output.remove_class("-hidden")
            output.remove_class("-details")
            output.update(rendered)
        else:
            output.update(Text(""))
            self.remove_class("-with-details")
            output.remove_class("-details")
            output.add_class("-hidden")

        self.query_one("#tool-header", Label).update(self._format_header())


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
            text.append("> ", style="bold")
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
