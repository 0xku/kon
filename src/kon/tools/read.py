import asyncio
from datetime import datetime
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from ..core.types import ImageContent
from ..tools_manager import ensure_tool
from ._read_image import is_image_file, read_and_process_image
from ._tool_utils import ToolCancelledError, communicate_or_cancel, shorten_path
from .base import BaseTool, ToolResult

MAX_CHARS_PER_LINE = 2000
MAX_LINES_PER_FILE = 2000
DIRECTORY_DEPTH_ROW_LIMIT = 200
MAX_DIRECTORY_ROWS = 1000


class ReadParams(BaseModel):
    path: str = Field(description="Absolute path of the file or directory to read")
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
    tool_icon = "→"
    params = ReadParams
    mutating = False
    description = (
        "Read the contents of a file or directory. "
        f"File reads truncate to {MAX_LINES_PER_FILE} lines and "
        f"{MAX_CHARS_PER_LINE} chars per line. "
        "Use offset/limit to paginate large files. "
        "Supports reading jpg/jpeg/png/gif/webp images."
    )

    def format_call(self, params: ReadParams) -> str:
        path = shorten_path(params.path)
        if params.offset or params.limit:
            start = params.offset or 1
            end = (start + params.limit - 1) if params.limit else "?"
            return f"{path}:{start}-{end}"
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

    def _format_directory_entry(self, entry_path: Path, relative: Path) -> str:
        modified = datetime.fromtimestamp(entry_path.stat().st_mtime)
        timestamp = f"{modified.day:2d} {modified.strftime('%b %H:%M')}"
        display = relative.as_posix()
        if entry_path.is_dir():
            display += "/"
        return f"{timestamp}  {display}"

    async def _list_directory_entries(
        self,
        fd_path: str,
        dir_path: Path,
        max_depth: int,
        max_results: int,
        cancel_event: asyncio.Event | None,
    ) -> list[str]:
        proc = await asyncio.create_subprocess_exec(
            fd_path,
            "--hidden",
            "--color=never",
            "--max-depth",
            str(max_depth),
            "--max-results",
            str(max_results),
            ".",
            str(dir_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await communicate_or_cancel(proc, cancel_event)
        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode not in (0, 1):
            raise RuntimeError(f"fd failed: {error_output or 'unknown error'}")

        if not output:
            return []

        entries: list[str] = []
        for line in output.split("\n"):
            if not line.strip():
                continue

            entry_path = Path(line)
            if entry_path.is_absolute():
                try:
                    relative = entry_path.relative_to(dir_path)
                except ValueError:
                    relative = entry_path
            else:
                relative = entry_path
                entry_path = dir_path / entry_path

            entries.append(self._format_directory_entry(entry_path, relative))

        return entries

    async def read_directory(
        self, dir_path: Path, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        fd_path = await ensure_tool("fd", silent=True)
        if not fd_path:
            msg = "fd is not available and could not be downloaded"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        try:
            for max_depth in (3, 2):
                entries = await self._list_directory_entries(
                    fd_path,
                    dir_path,
                    max_depth=max_depth,
                    max_results=DIRECTORY_DEPTH_ROW_LIMIT + 1,
                    cancel_event=cancel_event,
                )
                if len(entries) <= DIRECTORY_DEPTH_ROW_LIMIT:
                    result = "\n".join(entries) if entries else "(empty directory)"
                    return ToolResult(
                        success=True,
                        result=result,
                        ui_summary=f"[dim]({len(entries)} entries)[/dim]",
                    )

            entries = await self._list_directory_entries(
                fd_path,
                dir_path,
                max_depth=1,
                max_results=MAX_DIRECTORY_ROWS + 1,
                cancel_event=cancel_event,
            )
        except ToolCancelledError:
            return ToolResult(
                success=False, result="Read aborted", ui_summary="[yellow]Read aborted[/yellow]"
            )
        except RuntimeError as e:
            msg = str(e)
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        truncated = len(entries) > MAX_DIRECTORY_ROWS
        visible_entries = entries[:MAX_DIRECTORY_ROWS]
        result = "\n".join(visible_entries) if visible_entries else "(empty directory)"
        if truncated:
            result += f"\n[output truncated after {MAX_DIRECTORY_ROWS} lines]"

        shown = min(len(entries), MAX_DIRECTORY_ROWS)
        return ToolResult(
            success=True, result=result, ui_summary=f"[dim]({shown} entries shown)[/dim]"
        )

    async def execute(
        self, params: ReadParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        file_path = Path(params.path)

        if not file_path.exists():
            msg = "Path not found"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        if file_path.is_dir():
            return await self.read_directory(file_path, cancel_event)

        if not file_path.is_file():
            msg = "Path is not a file or directory"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

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
                    ui_summary=display_note,
                )
            except Exception as e:
                msg = f"Failed to read image: {e}"
                return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        try:
            content = await self.read_file(file_path, params.offset, params.limit)
        except OSError as e:
            msg = f"Failed to read: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        lines_read = len(content.splitlines()) if content else 0
        return ToolResult(
            success=True, result=content, ui_summary=f"[dim]({lines_read} lines)[/dim]"
        )
