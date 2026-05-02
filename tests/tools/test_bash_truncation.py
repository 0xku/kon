import asyncio
import tempfile

import pytest

from kon.tools.bash import MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, BashParams, BashTool


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


def _patch_subprocess(monkeypatch, proc: _FakeProcess) -> None:
    async def mock_create(*args, **kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create)


def _stdout_with_lines(line_count: int) -> bytes:
    return ("\n".join(f"line{i}" for i in range(1, line_count + 1)) + "\n").encode()


@pytest.mark.asyncio
async def test_inline_output_truncates_without_temp_file_path(monkeypatch):
    line_count = 5000
    _patch_subprocess(monkeypatch, _FakeProcess(_stdout_with_lines(line_count)))

    result = await BashTool().execute(BashParams(command="ignored"), inline_output=True)

    assert result.success is True
    assert result.result is not None
    assert f"[output truncated to last {MAX_OUTPUT_LINES} lines of {line_count}]" in result.result
    assert "full output:" not in result.result
    # Bounded: kept lines + a separator + the marker line.
    assert result.result.count("\n") <= MAX_OUTPUT_LINES + 2


@pytest.mark.asyncio
async def test_default_truncates_with_temp_file_path(monkeypatch):
    line_count = 5000
    _patch_subprocess(monkeypatch, _FakeProcess(_stdout_with_lines(line_count)))
    fake_path = f"{tempfile.gettempdir()}/kon-bash-fake.log"
    monkeypatch.setattr("kon.tools.bash._write_full_output_to_temp", lambda _: fake_path)

    result = await BashTool().execute(BashParams(command="ignored"))

    assert result.success is True
    assert result.result is not None
    assert (
        f"[output truncated to last {MAX_OUTPUT_LINES} lines of {line_count}; "
        f"full output: {fake_path}]"
    ) in result.result


@pytest.mark.asyncio
async def test_inline_output_keeps_excerpt_for_single_oversized_line(monkeypatch):
    oversized_line = b"a" * (MAX_OUTPUT_BYTES + 1)
    _patch_subprocess(monkeypatch, _FakeProcess(oversized_line))

    result = await BashTool().execute(BashParams(command="ignored"), inline_output=True)

    assert result.success is True
    assert result.result is not None
    assert result.result.startswith("a" * 100)
    assert "[output truncated to last 1 lines of 1]" in result.result
    assert "full output:" not in result.result
    assert len(result.result.encode()) < MAX_OUTPUT_BYTES + 200


@pytest.mark.asyncio
async def test_short_output_is_not_truncated(monkeypatch):
    _patch_subprocess(monkeypatch, _FakeProcess(b"hi\n"))

    result = await BashTool().execute(BashParams(command="ignored"), inline_output=True)

    assert result.success is True
    assert result.result is not None
    assert "truncated" not in result.result
    assert result.result.strip() == "hi"
