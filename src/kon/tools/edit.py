import asyncio
import difflib
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from kon import config

from ..core.types import FileChanges
from ..shared import shorten_path
from .base import BaseTool, ToolResult

CONTEXT_LINES = 4
MAX_DIFF_LINE_DISPLAY_CHARS = 105


class EditParams(BaseModel):
    path: str = Field(description="Absolute path of the file to edit")
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(
        description="The text to replace it with (must be different from old_string)"
    )
    replace_all: bool = Field(
        description="Replace all occurrences of old_string (default false)", default=False
    )


def generate_diff(
    old_content: str, new_content: str, context_lines: int = CONTEXT_LINES
) -> tuple[str, int, int]:
    """
    Generate a diff with line numbers and context.

    Returns:
        tuple: (diff_string, added_count, removed_count)

    Format:
        " 42 context line"      (space prefix = context)
        "-43 removed line"      (minus prefix = removed)
        "+43 added line"        (plus prefix = added)
        "    ..."               (ellipsis = skipped lines)
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()

    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    output: list[str] = []
    added, removed = 0, 0
    last_was_change = False

    for i, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            # Context lines - only show around changes
            equal_lines = old_lines[i1:i2]
            next_is_change = i < len(opcodes) - 1 and opcodes[i + 1][0] != "equal"

            if last_was_change or next_is_change:
                # Both sides need context - show trailing from prev change and leading to next
                if last_was_change and next_is_change:
                    if len(equal_lines) > context_lines * 2:
                        # Show first N lines (trailing context from prev change)
                        for idx, line in enumerate(equal_lines[:context_lines]):
                            line_num = i1 + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                        # Add ellipsis
                        output.append(f" {''.rjust(line_num_width)} ...")
                        # Show last N lines (leading context to next change)
                        for idx, line in enumerate(equal_lines[-context_lines:]):
                            line_num = i1 + len(equal_lines) - context_lines + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                    else:
                        # Show all lines
                        for idx, line in enumerate(equal_lines):
                            line_num = i1 + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                elif last_was_change:
                    # Only trailing context from prev change
                    if len(equal_lines) > context_lines:
                        for idx, line in enumerate(equal_lines[:context_lines]):
                            line_num = i1 + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                        output.append(f" {''.rjust(line_num_width)} ...")
                    else:
                        for idx, line in enumerate(equal_lines):
                            line_num = i1 + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                else:
                    # Only leading context to next change
                    if len(equal_lines) > context_lines:
                        output.append(f" {''.rjust(line_num_width)} ...")
                        for idx, line in enumerate(equal_lines[-context_lines:]):
                            line_num = i1 + len(equal_lines) - context_lines + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")
                    else:
                        for idx, line in enumerate(equal_lines):
                            line_num = i1 + idx + 1
                            output.append(f" {str(line_num).rjust(line_num_width)} {line}")

            last_was_change = False

        elif tag == "replace":
            for idx, line in enumerate(old_lines[i1:i2]):
                line_num = i1 + idx + 1
                output.append(f"-{str(line_num).rjust(line_num_width)} {line}")
                removed += 1
            for idx, line in enumerate(new_lines[j1:j2]):
                line_num = j1 + idx + 1
                output.append(f"+{str(line_num).rjust(line_num_width)} {line}")
                added += 1
            last_was_change = True

        elif tag == "delete":
            for idx, line in enumerate(old_lines[i1:i2]):
                line_num = i1 + idx + 1
                output.append(f"-{str(line_num).rjust(line_num_width)} {line}")
                removed += 1
            last_was_change = True

        elif tag == "insert":
            for idx, line in enumerate(new_lines[j1:j2]):
                line_num = j1 + idx + 1
                output.append(f"+{str(line_num).rjust(line_num_width)} {line}")
                added += 1
            last_was_change = True

    return "\n".join(output), added, removed


def truncate_diff_line(line: str, max_chars: int = MAX_DIFF_LINE_DISPLAY_CHARS) -> str:
    if len(line) <= max_chars:
        return line
    if max_chars <= 3:
        return "." * max_chars
    return f"{line[: max_chars - 3]}..."


def format_diff_display(diff: str) -> str:
    colors = config.ui.colors
    lines = diff.split("\n")
    formatted = []

    for line in lines:
        if not line:
            continue

        truncated = truncate_diff_line(line)
        escaped = truncated.replace("[", "\\[")

        if line.startswith("-"):
            formatted.append(f"[{colors.diff_removed}]{escaped}[/{colors.diff_removed}]")
        elif line.startswith("+"):
            formatted.append(f"[{colors.diff_added}]{escaped}[/{colors.diff_added}]")
        else:
            formatted.append(f"[dim]{escaped}[/dim]")

    return "\n".join(formatted)


class EditTool(BaseTool):
    name = "edit"
    params = EditParams
    description = (
        "Edit a file by replacing exact text. The old_string must match exactly "
        "(including whitespaces). Use this for precise, surgical edits."
    )

    def format_call(self, params: EditParams) -> str:
        accent = config.ui.colors.accent
        header = f"[{accent}]{shorten_path(params.path)}[/{accent}]"
        diff, _, _ = generate_diff(params.old_string, params.new_string)
        return f"{header}\n{format_diff_display(diff)}"

    async def execute(
        self, params: EditParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        file_path = Path(params.path)

        if not file_path.exists():
            msg = f"File not found: {file_path}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        async with aiofiles.open(file_path, encoding="utf-8") as f:
            content = await f.read()

        if params.old_string not in content:
            msg = "old_string not found in file"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        if params.replace_all:
            new_content = content.replace(params.old_string, params.new_string)
        else:
            new_content = content.replace(params.old_string, params.new_string, 1)

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(new_content)

        diff, added, removed = generate_diff(content, new_content)
        diff_display = format_diff_display(diff)

        short_path = shorten_path(str(file_path))
        colors = config.ui.colors
        result = f"Updated {file_path} +{added} -{removed}"
        display = (
            f"[dim]Updated {short_path}[/dim] [{colors.diff_added}]+{added}[/{colors.diff_added}] "
            f"[{colors.diff_removed}]-{removed}[/{colors.diff_removed}]"
        )
        display += f"\n{diff_display}"

        return ToolResult(
            success=True,
            result=result,
            display=display,
            file_changes=FileChanges(path=str(file_path), added=added, removed=removed),
        )
