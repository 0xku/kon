import asyncio
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from kon import config

from ..core.types import FileChanges
from ..shared import shorten_path
from .base import BaseTool, ToolResult


class WriteParams(BaseModel):
    path: str = Field(description="Absolute path of the file to write to")
    content: str = Field(description="Content to be written to the file")


class WriteTool(BaseTool):
    name = "write"
    tool_icon = "+"
    params = WriteParams
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )

    def format_call(self, params: WriteParams) -> str:
        return shorten_path(params.path)

    def format_preview(self, params: WriteParams) -> str | None:
        colors = config.ui.colors
        lines = params.content.splitlines()
        colored = []
        for line in lines[:20]:
            escaped = line.replace("[", "\\[")
            colored.append(f"[{colors.diff_added}]+{escaped}[/{colors.diff_added}]")
        if len(lines) > 20:
            colored.append(f"[dim]... ({len(lines) - 20} more lines)[/dim]")
        return "\n".join(colored)

    async def execute(
        self, params: WriteParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        file_path = Path(params.path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_existed = file_path.exists()

        old_line_count = 0
        if file_existed:
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    old_content = await f.read()
                old_line_count = old_content.count("\n") + 1
            except (OSError, UnicodeDecodeError):
                pass

        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(params.content)
        except OSError as e:
            msg = f"Failed to write: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        n_lines = params.content.count("\n") + 1
        diff_added = config.ui.colors.diff_added

        if file_existed:
            result = f"Overwrote {file_path} +{n_lines}"
            display = f"[{diff_added}]+{n_lines}[/{diff_added}]"
        else:
            result = f"Created {file_path} +{n_lines}"
            display = f"[{diff_added}]+{n_lines}[/{diff_added}]"

        return ToolResult(
            success=True,
            result=result,
            ui_summary=display,
            file_changes=FileChanges(path=str(file_path), added=n_lines, removed=old_line_count),
        )
