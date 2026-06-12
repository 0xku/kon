"""Tests for shell resolution and command spawning in the bash tool.

The Windows cases run on any platform by patching the module's _IS_WINDOWS
flag, so the Git-bash-path-with-spaces regression is covered without a
Windows machine.
"""

import os

import pytest

from kon.tools import bash
from kon.tools.bash import BashParams, BashTool, _get_shell, _get_spawn_argv

# Built with os.path.join so the expected separators match the host the test
# runs on; the space in "Program Files" is what the regression is about.
WIN_BASH = os.path.join("C:\\Program Files", "Git", "bin", "bash.exe")
WIN_BASH_X86 = os.path.join("C:\\Program Files (x86)", "Git", "bin", "bash.exe")


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(bash, "_IS_WINDOWS", True)
    monkeypatch.setenv("ProgramFiles", "C:\\Program Files")
    monkeypatch.setenv("ProgramFiles(x86)", "C:\\Program Files (x86)")


def test_get_shell_posix_uses_shell_env(monkeypatch):
    monkeypatch.setattr(bash, "_IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "/usr/local/bin/fish")
    assert _get_shell() == "/usr/local/bin/fish"


def test_get_shell_posix_falls_back_when_shell_unset(monkeypatch):
    monkeypatch.setattr(bash, "_IS_WINDOWS", False)
    monkeypatch.delenv("SHELL", raising=False)
    assert _get_shell() == "/bin/bash"


def test_get_shell_posix_falls_back_when_shell_empty(monkeypatch):
    monkeypatch.setattr(bash, "_IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "")
    assert _get_shell() == "/bin/bash"


def test_get_shell_windows_finds_git_bash(windows, monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: path == WIN_BASH)
    assert _get_shell() == WIN_BASH


def test_get_shell_windows_falls_back_to_x86(windows, monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: path == WIN_BASH_X86)
    assert _get_shell() == WIN_BASH_X86


def test_get_shell_windows_returns_none_without_git_bash(windows, monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    assert _get_shell() is None


def test_spawn_argv_windows_keeps_bash_path_with_spaces_intact(windows, monkeypatch):
    """Regression: the bash path must be a single argv element, not formatted
    unquoted into a shell command line where it splits at "Program Files"."""
    monkeypatch.setattr(os.path, "exists", lambda path: path == WIN_BASH)
    argv = _get_spawn_argv("grep -r 'register_cmd' .")
    assert argv == [WIN_BASH, "-c", "grep -r 'register_cmd' ."]


def test_spawn_argv_windows_without_git_bash_uses_platform_shell(windows, monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    assert _get_spawn_argv("echo hi") is None


def test_spawn_argv_posix(monkeypatch):
    monkeypatch.setattr(bash, "_IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert _get_spawn_argv("echo hi") == ["/bin/zsh", "-c", "echo hi"]


@pytest.mark.asyncio
async def test_execute_runs_command_through_resolved_shell():
    result = await BashTool().execute(BashParams(command="echo $0"))
    assert result.success
    assert result.result is not None
    assert os.environ.get("SHELL", "/bin/bash") in result.result


@pytest.mark.asyncio
async def test_execute_handles_quotes_and_spaces():
    result = await BashTool().execute(BashParams(command="printf '%s' \"a b  c\""))
    assert result.success
    assert result.result == "a b  c"
