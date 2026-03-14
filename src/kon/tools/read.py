import asyncio
import shlex
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from kon import config

from ..core.types import ImageContent
from ..shared import shorten_path
from ._read_image import is_image_file, read_and_process_image
from .base import BaseTool, ToolResult
from .bash import BashParams, BashTool

MAX_CHARS_PER_LINE = 2000
MAX_LINES_PER_FILE = 2000


class ReadParams(BaseModel):
    path: str = Field(description="Absolute path of the file to read")
    offset: int | None = Field(
        description="Line number to start reading from. "
        "Only provide if the file is too large to read at once.",
        default=None,
    )
    limit: int | None = Field(
        description="Number of lines to read. "
        "Only provide if the file is too large to read at once.",
        default=None,
    )


class ReadTool(BaseTool):
    name = "read"
    params = ReadParams
    mutating = False
    description = (
        "Read the contents of a file. "
        f"Truncates to {MAX_LINES_PER_FILE} lines and {MAX_CHARS_PER_LINE} chars per line. "
        "Use offset/limit to paginate large files. Supports reading jpg/jpeg/png/gif/webp images."
    )

    def format_call(self, params: ReadParams) -> str:
        accent = config.ui.colors.accent
        path = f"[{accent}]{shorten_path(params.path)}[/{accent}]"
        if params.offset or params.limit:
            start = params.offset or 1
            end = (start + params.limit - 1) if params.limit else "?"
            return f"{path}[yellow]:{start}-{end}[/yellow]"
        return path

    async def read_file(self, file_path: Path, offset: int | None, limit: int | None) -> str:
        lines = []
        start = (offset - 1) if offset else 0
        effective_limit = min(limit, MAX_LINES_PER_FILE) if limit else MAX_LINES_PER_FILE
        line_number = 0

        async with aiofiles.open(file_path, encoding="utf-8") as f:
            async for line in f:
                line_number += 1
                if line_number <= start:
                    continue
                if len(lines) == effective_limit:
                    if effective_limit == MAX_LINES_PER_FILE:
                        lines.append(f"[output truncated after {MAX_LINES_PER_FILE} lines]")
                    break

                if len(line) > MAX_CHARS_PER_LINE:
                    line = (
                        line[:MAX_CHARS_PER_LINE]
                        + f" [output truncated after {MAX_CHARS_PER_LINE} chars]\n"
                    )
                lines.append(f"{line_number:6d}\t{line}")

        return "".join(lines)

    async def execute(
        self, params: ReadParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        file_path = Path(params.path)

        if not file_path.exists():
            msg = "Path not found"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        if not file_path.is_file():
            if file_path.is_dir():
                bash_tool = BashTool()
                ls_result = await bash_tool.execute(
                    BashParams(command=f"ls -la {shlex.quote(str(file_path))}")
                )
                output = ls_result.result or "(no output)"
                warning = (
                    "\n\nWARNING: read tool is only supposed to be used for file reads; "
                    "for listing dirs, used bash ls tool"
                )
                return ToolResult(
                    success=ls_result.success,
                    result=f"{output}{warning}",
                    display=ls_result.display,
                )
            msg = "Path is not a file"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        if is_image_file(str(file_path)):
            try:
                base64_data, mime_type, resize_note = read_and_process_image(str(file_path))

                text_note = f"Read image file [{mime_type}]"
                if resize_note:
                    text_note += f" {resize_note}"

                display_note = "[dim]Read image[/dim]"
                if resize_note:
                    display_note = f"{display_note} {resize_note}"

                return ToolResult(
                    success=True,
                    result=text_note,
                    images=[ImageContent(data=base64_data, mime_type=mime_type)],
                    display=display_note,
                )
            except Exception as e:
                msg = f"Failed to read image: {e}"
                return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        try:
            content = await self.read_file(file_path, params.offset, params.limit)
        except OSError as e:
            msg = f"Failed to read: {e}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        lines_read = len(content.splitlines()) if content else 0
        return ToolResult(
            success=True, result=content, display=f"[dim]Read {lines_read} lines[/dim]"
        )
