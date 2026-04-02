import os
from datetime import datetime
from pathlib import Path

import pytest

from kon.tools.read import ReadParams, ReadTool


@pytest.fixture
def read_tool():
    return ReadTool()


@pytest.fixture
def text_file(tmp_path):
    f = tmp_path / "index.py"
    f.write_text("line1\nline2\nline3\nlong-line-number-4\nline-5\nline-6")
    return f


def _touch_with_timestamp(path: Path, year: int, month: int, day: int, hour: int, minute: int):
    ts = datetime(year, month, day, hour, minute).timestamp()
    os.utime(path, (ts, ts))


@pytest.mark.asyncio
async def test_read(read_tool, text_file, monkeypatch):
    monkeypatch.setattr("kon.tools.read.MAX_LINES_PER_FILE", 5)
    monkeypatch.setattr("kon.tools.read.MAX_CHARS_PER_LINE", 10)

    tool_result = await read_tool.execute(ReadParams(path=str(text_file)))
    lines = tool_result.result.split("\n")
    assert len(lines) == 6  # 5 lines + truncation
    assert lines[0] == "     1\tline1"
    assert lines[3] == "     4\tlong-line- [output truncated after 10 chars]"
    assert lines[-1] == "[output truncated after 5 lines]"

    tool_result = await read_tool.execute(ReadParams(path=str(text_file), offset=2, limit=3))
    lines = tool_result.result.split("\n")
    assert len(lines) == 4  # 3 lines + trailing ""
    assert lines[0] == "     2\tline2"
    assert lines[2] == "     4\tlong-line- [output truncated after 10 chars]"


@pytest.mark.asyncio
async def test_read_path_not_found(read_tool, tmp_path):
    result = await read_tool.execute(ReadParams(path=str(tmp_path / "nonexistent.txt")))
    assert not result.success
    assert "Path not found" in result.result
    assert "Path not found" in result.ui_summary


@pytest.mark.asyncio
async def test_read_not_a_file_or_directory(read_tool, tmp_path):
    fifo_path = tmp_path / "myfifo"
    os.mkfifo(fifo_path)
    result = await read_tool.execute(ReadParams(path=str(fifo_path)))
    assert not result.success
    assert "Path is not a file or directory" in result.result
    assert "Path is not a file or directory" in result.ui_summary


@pytest.mark.asyncio
async def test_read_directory_uses_fd_with_depth_3_when_small(read_tool, tmp_path, monkeypatch):
    calls = []

    async def mock_list_directory_entries(
        self, fd_path, dir_path, max_depth, max_results, cancel_event
    ):
        calls.append((fd_path, Path(dir_path), max_depth, max_results))
        return [" 2 Apr 15:19  a.py", " 2 Apr 15:19  nested/", " 2 Apr 15:19  nested/file.py"]

    async def mock_ensure_tool(tool, silent=False):
        assert tool == "fd"
        return "fd"

    monkeypatch.setattr(ReadTool, "_list_directory_entries", mock_list_directory_entries)
    monkeypatch.setattr("kon.tools.read.ensure_tool", mock_ensure_tool)

    tool_result = await read_tool.execute(ReadParams(path=str(tmp_path)))

    assert calls == [("fd", tmp_path, 3, 201)]
    assert tool_result.success is True
    assert tool_result.result == (
        " 2 Apr 15:19  a.py\n 2 Apr 15:19  nested/\n 2 Apr 15:19  nested/file.py"
    )
    assert tool_result.ui_summary == "[dim](3 entries)[/dim]"


@pytest.mark.asyncio
async def test_read_directory_falls_back_to_depth_2(read_tool, tmp_path, monkeypatch):
    calls = []

    async def mock_list_directory_entries(
        self, fd_path, dir_path, max_depth, max_results, cancel_event
    ):
        calls.append(max_depth)
        if max_depth == 3:
            return [f"entry-{i}" for i in range(201)]
        return [" 2 Apr 15:19  a.py", " 2 Apr 15:19  b.py"]

    async def mock_ensure_tool(tool, silent=False):
        return "fd"

    monkeypatch.setattr(ReadTool, "_list_directory_entries", mock_list_directory_entries)
    monkeypatch.setattr("kon.tools.read.ensure_tool", mock_ensure_tool)

    tool_result = await read_tool.execute(ReadParams(path=str(tmp_path)))

    assert calls == [3, 2]
    assert tool_result.success is True
    assert tool_result.result == " 2 Apr 15:19  a.py\n 2 Apr 15:19  b.py"
    assert tool_result.ui_summary == "[dim](2 entries)[/dim]"


@pytest.mark.asyncio
async def test_read_directory_falls_back_to_depth_1_and_truncates(
    read_tool, tmp_path, monkeypatch
):
    calls = []

    async def mock_list_directory_entries(
        self, fd_path, dir_path, max_depth, max_results, cancel_event
    ):
        calls.append((max_depth, max_results))
        if max_depth in (3, 2):
            return [f" 2 Apr 15:19  entry-{i}" for i in range(201)]
        return [f" 2 Apr 15:19  entry-{i}" for i in range(1001)]

    async def mock_ensure_tool(tool, silent=False):
        return "fd"

    monkeypatch.setattr(ReadTool, "_list_directory_entries", mock_list_directory_entries)
    monkeypatch.setattr("kon.tools.read.ensure_tool", mock_ensure_tool)

    tool_result = await read_tool.execute(ReadParams(path=str(tmp_path)))

    assert calls == [(3, 201), (2, 201), (1, 1001)]
    assert tool_result.success is True
    lines = tool_result.result.split("\n")
    assert len(lines) == 1001
    assert lines[0] == " 2 Apr 15:19  entry-0"
    assert lines[-1] == "[output truncated after 1000 lines]"
    assert tool_result.ui_summary == "[dim](1000 entries shown)[/dim]"


@pytest.mark.asyncio
async def test_read_directory_empty(read_tool, tmp_path, monkeypatch):
    async def mock_list_directory_entries(
        self, fd_path, dir_path, max_depth, max_results, cancel_event
    ):
        return []

    async def mock_ensure_tool(tool, silent=False):
        return "fd"

    monkeypatch.setattr(ReadTool, "_list_directory_entries", mock_list_directory_entries)
    monkeypatch.setattr("kon.tools.read.ensure_tool", mock_ensure_tool)

    tool_result = await read_tool.execute(ReadParams(path=str(tmp_path)))

    assert tool_result.success is True
    assert tool_result.result == "(empty directory)"
    assert tool_result.ui_summary == "[dim](0 entries)[/dim]"


@pytest.mark.asyncio
async def test_read_directory_formats_modified_time(read_tool, tmp_path):
    file_path = tmp_path / "alpha.py"
    file_path.write_text("x")
    dir_path = tmp_path / "nested"
    dir_path.mkdir()
    _touch_with_timestamp(file_path, 2026, 4, 2, 15, 19)
    _touch_with_timestamp(dir_path, 2026, 4, 3, 9, 5)

    assert (
        read_tool._format_directory_entry(file_path, Path("alpha.py")) == " 2 Apr 15:19  alpha.py"
    )
    assert read_tool._format_directory_entry(dir_path, Path("nested")) == " 3 Apr 09:05  nested/"
