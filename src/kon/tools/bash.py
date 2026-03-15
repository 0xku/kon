import asyncio
import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from kon import config

from ..config import AVAILABLE_BINARIES
from ..core.types import ToolResult
from .base import BaseTool

DEFAULT_TIMEOUT = 180
MAX_OUTPUT_BYTES = 50 * 1024
MAX_OUTPUT_LINES = 2000

_IS_WINDOWS: bool = sys.platform == "win32"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[()][AB012]")


def _get_env() -> dict[str, str]:
    return {
        **os.environ,
        "CI": "true",
        "NO_COLOR": "1",
        "TERM": "dumb",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }


def _get_shell() -> str | None:
    if _IS_WINDOWS:
        program_files = os.environ.get("ProgramFiles", "")  # noqa: SIM112
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "")  # noqa: SIM112
        paths = [
            os.path.join(program_files, "Git", "bin", "bash.exe"),
            os.path.join(program_files_x86, "Git", "bin", "bash.exe"),
        ]
        for path in paths:
            if path and os.path.exists(path):
                return path
        return None
    return os.environ.get("SHELL", "/bin/bash")


def _sanitize_output(text: str) -> str:
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "")
    text = "".join(c for c in text if c >= " " or c in "\t\n")
    return text


def _transform_command(command: str) -> str:
    """
    Transform command if a better alternative is available.
    Currently only transforms 'ls' to 'eza' if eza is installed.
    """
    stripped = command.strip()
    if not (stripped == "ls" or stripped.startswith("ls ")):
        return command

    if "eza" not in AVAILABLE_BINARIES:
        return command

    # Replace 'ls' with 'eza --git-ignore', preserving any following arguments
    # We only match 'ls' as a word (not part of another command)
    result = re.sub(r"\bls\b", "eza --git-ignore", command, count=1)
    # eza blocks on stdin in subprocess envs when no path arg is given
    if all(a.startswith("-") for a in stripped.split()[1:]):
        result += " ."
    return result


class TruncationResult:
    def __init__(self, content: str, truncated: bool, lines_kept: int, total_lines: int):
        self.content = content
        self.truncated = truncated
        self.lines_kept = lines_kept
        self.total_lines = total_lines


def _truncate_tail(text: str) -> TruncationResult:
    """
    Truncate from the head, keeping the last N lines/bytes (tail truncation).
    Better for bash output where errors/results are at the end.
    """
    lines = text.split("\n")
    total_lines = len(lines)

    if total_lines <= MAX_OUTPUT_LINES and len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES:
        return TruncationResult(text, False, total_lines, total_lines)

    output_lines: list[str] = []
    output_bytes = 0

    for i in range(total_lines - 1, -1, -1):
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)

        if output_bytes + line_bytes > MAX_OUTPUT_BYTES:
            break
        if len(output_lines) >= MAX_OUTPUT_LINES:
            break

        output_lines.insert(0, line)
        output_bytes += line_bytes

    return TruncationResult("\n".join(output_lines), True, len(output_lines), total_lines)


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    try:
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _write_full_output_to_temp(output: str) -> str:
    fd, path = tempfile.mkstemp(prefix="kon-bash-", suffix=".log")
    try:
        os.write(fd, output.encode("utf-8"))
    finally:
        os.close(fd)
    return path


class BashParams(BaseModel):
    command: str = Field(description="The bash command to execute")
    timeout: int = Field(
        description=f"Timeout in seconds (default {DEFAULT_TIMEOUT})", default=DEFAULT_TIMEOUT
    )


class BashTool(BaseTool):
    name = "bash"
    params = BashParams
    description = (
        "Execute a bash command in the current working directory. "
        f"Output truncated to last {MAX_OUTPUT_LINES} lines or {MAX_OUTPUT_BYTES // 1024}KB. "
        "If truncated, full output is saved to a temp file. "
        "Optionally provide a timeout in seconds. "
        "IMPORTANT: Do NOT use bash for file search (use grep/find tools instead), "
        "reading files (use read), or editing files (use edit)."
    )

    # TODO: Add streaming support via an optional `on_chunk` callback parameter
    # Implementation approach:
    # 1. Add `on_chunk: Callable[[str], None] | None = None` parameter to execute()
    # 2. Instead of proc.communicate(), read from proc.stdout/stderr in a loop
    # 3. Keep a rolling buffer of chunks (max 2x MAX_OUTPUT_BYTES) for tail truncation
    # 4. Call on_chunk(sanitized_text) for each chunk received
    # 5. Start writing to temp file once total bytes exceed MAX_OUTPUT_BYTES
    # 6. On completion, apply tail truncation to the rolling buffer

    def format_call(self, params: BashParams) -> str:
        return params.command

    def _format_display(self, output: str, max_lines: int = 5, max_line_chars: int = 500) -> str:
        truncation_color = config.ui.colors.dim

        if not output:
            return f"[{truncation_color}](no output)[/{truncation_color}]"

        lines = output.split("\n")
        hidden_lines = max(0, len(lines) - max_lines)
        display_lines = lines[:max_lines]

        formatted: list[str] = []
        for line in display_lines:
            # Truncate long lines for UI display only
            if len(line) > max_line_chars:
                visible = line[:max_line_chars].replace("[", "\\[")
                hidden_chars = len(line) - max_line_chars
                formatted.append(
                    f"[dim]{visible}[/dim]"
                    f"[{truncation_color}]... ({hidden_chars} more chars)[/{truncation_color}]"
                )
            else:
                escaped = line.replace("[", "\\[")
                formatted.append(f"[dim]{escaped}[/dim]")

        if hidden_lines > 0:
            formatted.append(
                f"[{truncation_color}]... ({hidden_lines} more lines)[/{truncation_color}]"
            )

        return "\n".join(formatted)

    async def execute(
        self, params: BashParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        if not params.command.strip():
            return ToolResult(success=False, display="[red]Command cannot be empty[/red]")

        command = _transform_command(params.command)

        cwd = Path.cwd()
        if not cwd.exists():
            return ToolResult(
                success=False, display=f"[red]Working directory does not exist: {cwd}[/red]"
            )

        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=_get_env(),
                executable=_get_shell(),
                start_new_session=not _IS_WINDOWS,
            )

            comm_task = asyncio.create_task(proc.communicate())

            try:
                if cancel_event:
                    cancel_wait = asyncio.create_task(cancel_event.wait())
                    done, pending = await asyncio.wait(
                        [comm_task, cancel_wait],
                        timeout=params.timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    if not done:
                        await _kill_process_tree(proc)
                        return ToolResult(
                            success=False,
                            display=f"[red]Command timed out after {params.timeout}s[/red]",
                        )

                    if cancel_wait in done and cancel_event.is_set():
                        await _kill_process_tree(proc)
                        return ToolResult(
                            success=False,
                            result="Command aborted",
                            display="[yellow]Command aborted by user[/yellow]",
                        )

                    stdout_bytes, stderr_bytes = comm_task.result()
                else:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        comm_task, timeout=params.timeout
                    )

            except TimeoutError:
                await _kill_process_tree(proc)
                return ToolResult(
                    success=False, display=f"[red]Command timed out after {params.timeout}s[/red]"
                )

            stdout = _sanitize_output(stdout_bytes.decode("utf-8", errors="replace"))
            stderr = _sanitize_output(stderr_bytes.decode("utf-8", errors="replace"))

            full_output = ""
            if stdout:
                full_output += stdout
            if stderr:
                full_output += f"\n[stderr]\n{stderr}" if full_output else f"[stderr]\n{stderr}"
            full_output = full_output.rstrip()

            trunc = _truncate_tail(full_output)

            temp_file_path = None
            if trunc.truncated:
                temp_file_path = _write_full_output_to_temp(full_output)
                trunc.content += (
                    f"\n\n[output truncated to last {trunc.lines_kept} lines "
                    f"of {trunc.total_lines}; full output: {temp_file_path}]"
                )

            result_text = trunc.content or "(no output)"
            display_text = self._format_display(trunc.content)

            if proc.returncode == 0:
                return ToolResult(success=True, result=result_text, display=display_text)
            else:
                return ToolResult(
                    success=False,
                    result=result_text,
                    display=f"[red]Exit code {proc.returncode}[/red]\n{display_text}",
                )

        except Exception as e:
            return ToolResult(success=False, display=f"[red]Error running command: {e}[/red]")
        finally:
            if proc is not None:
                await _kill_process_tree(proc)
