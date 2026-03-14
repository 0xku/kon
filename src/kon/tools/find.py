import asyncio
import os

from pydantic import BaseModel, Field

from kon import config

from ..core.types import ToolResult
from ..tools_manager import ensure_tool
from .base import BaseTool

MAX_RESULTS = 100
MAX_OUTPUT_BYTES = 50 * 1024


class FindParams(BaseModel):
    pattern: str = Field(
        description="Glob pattern to match files, e.g. '*.py', '**/*.json', or 'src/**/*.spec.ts'"
    )
    path: str | None = Field(
        description="Directory to search in (default: current directory)", default=None
    )


class FindTool(BaseTool):
    name = "find"
    params = FindParams
    mutating = False
    description = (
        "Search for files by glob pattern using fd. "
        "Returns matching file paths relative to the search directory, "
        "sorted by modification time."
        f"Respects .gitignore. Truncated to {MAX_RESULTS} results."
    )

    def format_call(self, params: FindParams) -> str:
        accent = config.ui.colors.accent
        pattern = params.pattern.replace('"', '\\"')
        parts = [f'[{accent}]"{pattern}"[/{accent}]']
        if params.path:
            parts.append(f"in [{accent}]{params.path}[/{accent}]")
        return " ".join(parts)

    async def execute(
        self, params: FindParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        fd_path = await ensure_tool("fd", silent=True)
        if not fd_path:
            msg = "fd is not available and could not be downloaded"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        search_path = params.path or os.getcwd()
        if not os.path.isabs(search_path):
            search_path = os.path.join(os.getcwd(), search_path)

        if not os.path.exists(search_path):
            msg = f"Path not found: {search_path}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        args = [
            fd_path,
            "--glob",
            "--color=never",
            "--hidden",
            "--max-results",
            str(MAX_RESULTS),
            params.pattern,
            search_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        if cancel_event:
            comm_task = asyncio.create_task(proc.communicate())
            cancel_wait = asyncio.create_task(cancel_event.wait())
            done, pending = await asyncio.wait(
                [comm_task, cancel_wait], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if cancel_wait in done:
                proc.kill()
                return ToolResult(success=False, result="Search aborted")
            stdout, stderr = comm_task.result()
        else:
            stdout, stderr = await proc.communicate()

        exit_code = proc.returncode
        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()

        if exit_code not in (0, 1) and not output:
            msg = f"fd failed: {error_output}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        if not output:
            return ToolResult(
                success=True,
                result="No files found matching pattern",
                display="[dim]No files found[/dim]",
            )

        lines = [line.strip() for line in output.split("\n") if line.strip()]

        # Relativize and collect mtime for sorting
        files: list[tuple[str, float]] = []
        for line in lines:
            if line.startswith(search_path):
                rel = line[len(search_path) :].lstrip(os.sep)
                rel = rel if rel else line
            else:
                rel = os.path.relpath(line, search_path)
            try:
                mtime = os.path.getmtime(line)
            except OSError:
                mtime = 0.0
            files.append((rel, mtime))

        files.sort(key=lambda f: f[1], reverse=True)

        relativized = [f[0] for f in files]
        truncated = len(relativized) >= MAX_RESULTS
        result_text = "\n".join(relativized)

        if truncated:
            result_text += (
                f"\n\n[{MAX_RESULTS} results limit reached; "
                "refine the pattern for more specific results]"
            )

        if len(result_text.encode("utf-8")) > MAX_OUTPUT_BYTES:
            result_text = result_text[: MAX_OUTPUT_BYTES // 2] + "\n\n[output truncated]"

        count = len(relativized)
        display = f"[dim]{count} files found[/dim]"

        return ToolResult(success=True, result=result_text, display=display)
