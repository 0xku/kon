import asyncio
import contextlib
import os

from pydantic import BaseModel, Field

from kon import config

from ..core.types import ToolResult
from ..tools_manager import ensure_tool
from .base import BaseTool

MAX_MATCHES = 100
MAX_OUTPUT_BYTES = 50 * 1024
MAX_LINE_LENGTH = 2000


class GrepParams(BaseModel):
    pattern: str = Field(description="The regex pattern to search for in file contents")
    path: str | None = Field(
        description="Directory or file to search (default: current directory)", default=None
    )
    include: str | None = Field(
        description='File pattern to include in the search (e.g. "*.py", "*.{ts,tsx}")',
        default=None,
    )


class GrepTool(BaseTool):
    name = "grep"
    params = GrepParams
    mutating = False
    description = (
        "Search file contents using ripgrep. "
        "Returns matching lines with file paths and line numbers, sorted by modification time. "
        f"Respects .gitignore. Truncated to {MAX_MATCHES} matches."
    )

    def format_call(self, params: GrepParams) -> str:
        accent = config.ui.colors.accent
        pattern = params.pattern.replace('"', '\\"')
        parts = [f'[{accent}]"{pattern}"[/{accent}]']
        if params.path:
            parts.append(f"in [{accent}]{params.path}[/{accent}]")
        if params.include:
            parts.append(f"--include={params.include}")
        return " ".join(parts)

    async def execute(
        self, params: GrepParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        rg_path = await ensure_tool("rg", silent=True)
        if not rg_path:
            msg = "ripgrep (rg) is not available and could not be downloaded"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        search_path = params.path or os.getcwd()
        if not os.path.isabs(search_path):
            search_path = os.path.join(os.getcwd(), search_path)

        if not os.path.exists(search_path):
            msg = f"Path not found: {search_path}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        args = [
            rg_path,
            "-nH",
            "--hidden",
            "--no-messages",
            "--field-match-separator=|",
            "--regexp",
            params.pattern,
        ]
        if params.include:
            args.extend(["--glob", params.include])
        args.append(search_path)

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

        output = stdout.decode("utf-8", errors="replace")
        error_output = stderr.decode("utf-8", errors="replace")

        # Exit codes: 0 = matches, 1 = no matches, 2 = errors (may still have matches)
        if exit_code == 1 or (exit_code == 2 and not output.strip()):
            return ToolResult(
                success=True, result="No matches found", display="[dim]No matches found[/dim]"
            )

        if exit_code not in (0, 2):
            msg = f"ripgrep failed: {error_output}"
            return ToolResult(success=False, result=msg, display=f"[red]{msg}[/red]")

        lines = output.strip().split("\n")
        matches = []

        for line in lines:
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            file_path, line_num_str, line_text = parts
            try:
                line_num = int(line_num_str)
            except ValueError:
                continue
            mtime = 0
            with contextlib.suppress(OSError):
                mtime = os.path.getmtime(file_path)
            matches.append((file_path, mtime, line_num, line_text))

        matches.sort(key=lambda m: m[1], reverse=True)

        truncated = len(matches) > MAX_MATCHES
        matches = matches[:MAX_MATCHES]

        if not matches:
            return ToolResult(
                success=True, result="No matches found", display="[dim]No matches found[/dim]"
            )

        total_matches = len(lines)
        output_lines = [
            f"Found {total_matches} matches"
            + (f" (showing first {MAX_MATCHES})" if truncated else "")
        ]

        current_file = ""
        for file_path, _, line_num, line_text in matches:
            if current_file != file_path:
                if current_file:
                    output_lines.append("")
                current_file = file_path
                output_lines.append(f"{file_path}:")
            if len(line_text) > MAX_LINE_LENGTH:
                line_text = line_text[:MAX_LINE_LENGTH] + "..."
            output_lines.append(f"  Line {line_num}: {line_text}")

        result_text = "\n".join(output_lines)

        if truncated:
            output_lines += (
                f"\n\n[showing {MAX_MATCHES} of {total_matches} matches; "
                "refine the pattern or path for more specific results]"
            )

        if len(result_text.encode("utf-8")) > MAX_OUTPUT_BYTES:
            result_text = result_text[: MAX_OUTPUT_BYTES // 2] + "\n\n[output truncated]"

        match_count = min(total_matches, MAX_MATCHES)
        display = f"[dim]{match_count} matches[/dim]"

        return ToolResult(success=True, result=result_text, display=display)
