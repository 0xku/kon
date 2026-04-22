import re
import shutil
from typing import ClassVar

from rich._loop import loop_first
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, ListElement, ListItem, Markdown
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from kon import config

from .latex import preprocess_latex

_MARKDOWN_THEME: Theme | None = None


def get_markdown_theme() -> Theme:
    global _MARKDOWN_THEME
    if _MARKDOWN_THEME is None:
        heading_color = config.ui.colors.markdown_heading
        code_color = config.ui.colors.markdown_code
        heading_style = Style(color=heading_color, bold=True)
        _MARKDOWN_THEME = Theme(
            {
                "markdown.h1": heading_style,
                "markdown.h2": heading_style,
                "markdown.h3": heading_style,
                "markdown.h4": heading_style,
                "markdown.h5": heading_style,
                "markdown.h6": heading_style,
                "markdown.code": Style(color=code_color),
            }
        )
    return _MARKDOWN_THEME


MARKDOWN_THEME = get_markdown_theme()


class LeftJustifiedHeading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield from console.render(self.text, options=options.update(justify="left"))


class PlainListItem(ListItem):
    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 2)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet = Segment("- ")
        padding = Segment("  ")
        new_line = Segment("\n")
        for first, line in loop_first(lines):
            yield bullet if first else padding
            yield from line
            yield new_line

    def render_number(
        self, console: Console, options: ConsoleOptions, number: int, last_number: int
    ) -> RenderResult:
        number_width = len(str(last_number)) + 2
        render_options = options.update(width=options.max_width - number_width)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        new_line = Segment("\n")
        padding = Segment(" " * number_width)
        numeral = Segment(f"{number}".rjust(number_width - 1) + " ")
        for first, line in loop_first(lines):
            yield numeral if first else padding
            yield from line
            yield new_line


class PlainListElement(ListElement):
    def on_child_close(self, context, child) -> bool:
        assert isinstance(child, ListItem)
        self.items.append(child)
        return False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.list_type == "bullet_list_open":
            for item in self.items:
                if isinstance(item, PlainListItem):
                    yield from item.render_bullet(console, options)
        else:
            number = 1 if self.list_start is None else self.list_start
            last_number = number + len(self.items)
            for index, item in enumerate(self.items):
                if isinstance(item, PlainListItem):
                    yield from item.render_number(console, options, number + index, last_number)


class PlainCodeBlock(CodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        syntax = Syntax(code, self.lexer_name, theme="ansi_dark", word_wrap=True, padding=0)
        yield syntax


class CustomMarkdown(Markdown):
    elements: ClassVar[dict] = {
        **Markdown.elements,
        "heading_open": LeftJustifiedHeading,
        "bullet_list_open": PlainListElement,
        "ordered_list_open": PlainListElement,
        "list_item_open": PlainListItem,
        "fence": PlainCodeBlock,
        "code_block": PlainCodeBlock,
    }


def _strip_inline_code_ticks_in_headings(text: str) -> str:
    lines = text.splitlines(keepends=True)
    in_fence = False
    processed: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            processed.append(line)
            continue

        if in_fence:
            processed.append(line)
            continue

        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            line = re.sub(r"`([^`]+)`", r"\1", line)

        processed.append(line)

    return "".join(processed)


def strip_markdown_for_collapsed_text(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def format_markdown(text: str, width: int | None = None) -> Text:
    text = preprocess_latex(text)
    sanitized = _strip_inline_code_ticks_in_headings(text)
    md = CustomMarkdown(sanitized)
    if width is None:
        term_width = shutil.get_terminal_size().columns
        width = max(40, term_width - 4)
    console = Console(force_terminal=True, no_color=False, theme=MARKDOWN_THEME, width=width)
    with console.capture() as capture:
        console.print(md)
    rendered = capture.get()
    return Text.from_ansi(rendered.rstrip("\n"))


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{int(n / 1_000_000)}m"
    elif n >= 1_000:
        return f"{int(n / 1_000)}k"
    return str(n)
