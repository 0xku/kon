from dataclasses import dataclass
from typing import Literal

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Label, Markdown, Static

from kon import config

from .formatting import format_markdown

_UPDATE_COMMAND = "uv tool upgrade kon-coding-agent"


@dataclass(frozen=True)
class LaunchWarning:
    message: str
    severity: Literal["warning", "error"] = "warning"


class ThinkingBlock(Static):
    """Uses plain text during streaming for performance, renders markdown on finalize."""

    ALLOW_SELECT = True
    can_focus = False

    def __init__(self, content: str = "", finalized: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self._finalized = finalized
        self._label: Label | None = None
        self.add_class("thinking-block")

    def compose(self) -> ComposeResult:
        if self._finalized and self._content:
            yield Label(format_markdown(self._content), id="thinking-content", markup=False)
        else:
            yield Label(self._content, id="thinking-content", markup=False)

    @property
    def label(self) -> Label:
        if self._label is None:
            self._label = self.query_one("#thinking-content", Label)
        return self._label

    async def append(self, text: str) -> None:
        self._content += text
        self.label.update(self._content)

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


class ContentBlock(Static):
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
        self.add_class("content-block")

    def compose(self) -> ComposeResult:
        # If created with content (loading history), render markdown immediately
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
        self.label.update(self._content)

    def finalize(self) -> None:
        if self._content and not self._finalized:
            self._finalized = True
            # Use call_after_refresh to batch the update
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

    def __init__(self, name: str = "", call_msg: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._call_msg = call_msg
        self._result: str | None = None
        self._success: bool | None = None
        self.add_class("tool-block")
        self._set_state(None)

    def compose(self) -> ComposeResult:
        yield Label(self._format_header(), id="tool-header")
        yield Label(self._format_pending_output(), id="tool-output")

    def _format_header(self) -> Text:
        result = Text()
        formatted_name = " ".join(word.capitalize() for word in self._name.split("_"))
        result.append(formatted_name, style="bold")
        if self._call_msg:
            result.append(" ")
            result.append_text(self._format_call_msg())
        return result

    def _format_call_msg(self) -> Text:
        if not self._call_msg:
            return Text()
        lines = self._call_msg.split("\n")
        if len(lines) > self.MAX_HEADER_LINES:
            display_msg = "\n".join(lines[: self.MAX_HEADER_LINES])
            display_msg += f"\n... ({len(lines) - self.MAX_HEADER_LINES} more lines)"
            return self._render_markup_safe(display_msg)
        return self._render_markup_safe(self._call_msg)

    def _format_pending_output(self) -> Text:
        dim_color = config.ui.colors.dim
        return Text("...", style=dim_color)

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
        self.remove_class("-pending", "-success", "-error")
        if success is None:
            self.add_class("-pending")
        elif success:
            self.add_class("-success")
        else:
            self.add_class("-error")

    def update_call_msg(self, call_msg: str) -> None:
        self._call_msg = call_msg
        self.query_one("#tool-header", Label).update(self._format_header())

    def set_result(self, content: str, success: bool, markup: bool = True) -> None:
        self._result = content
        self._success = success
        self._set_state(success)

        # Parse Rich markup for colored output (tools control their own truncation/styling)
        rendered = self._render_markup_safe(content) if markup else Text(content)
        self.query_one("#tool-output", Label).update(rendered)
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
            markers = [f"[{self._highlighted_skill}]", "[query]"]
            for marker in markers:
                start = self._content.find(marker)
                if start != -1:
                    text.stylize(
                        f"{config.ui.colors.badge.label} bold", start, start + len(marker)
                    )
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
        markdown_text = (
            f"**[handoff]** {self._label}\n"
            f"[Open linked session](kon-session://{self._target_session_id})\n"
            f"Query: {self._query}"
        )
        markdown = Markdown(markdown_text, open_links=False)
        markdown.add_class("handoff-link-content")
        yield markdown

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        event.stop()
        if not event.href.startswith("kon-session://"):
            return
        target_session_id = event.href[len("kon-session://") :].strip()
        if not target_session_id:
            return
        self.post_message(self.LinkSelected(self, target_session_id, self._query, self._direction))

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
