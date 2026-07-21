from pathlib import Path

import pytest

import kon.platform_utils as pu


def test_pu_get_git_root_dir(tmp_path: Path):
    repo = tmp_path / "repo"
    deep_dir = repo / ".git" / "deep"
    deep_dir.mkdir(parents=True)
    assert pu.get_git_root_dir(str(deep_dir)) == str(repo)


# def test_pu_get_stop_directory(cwd: str) -> str:
#     pass


def test_pu_get_bash_dir_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", True)
    bin_dir = tmp_path / "Git" / "bin"
    bin_dir.mkdir(parents=True)
    bash_path = bin_dir / "bash.exe"
    bash_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert pu.get_bash_dir() == tmp_path.as_posix() + "/Git/bin/bash.exe"


def test_pu_get_bash_dir_posix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "/usr/local/bin/fish")
    assert pu.get_bash_dir() == "/usr/local/bin/fish"


def test_pu_get_bash_dir_posix_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", False)
    monkeypatch.delenv("SHELL", raising=False)
    assert pu.get_bash_dir() == "/bin/bash"


def test_pu_get_bash_dir_posix_fallback_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", False)
    monkeypatch.setenv("SHELL", "")
    assert pu.get_bash_dir() == "/bin/bash"


def test_pu_get_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    config_dir = repo / ".config" / "kon"
    config_dir.mkdir(parents=True)
    if pu.is_windows():
        monkeypatch.setenv("USERPROFILE", str(repo))
    else:
        monkeypatch.setenv("HOME", str(repo))
    assert pu.get_config_dir() == str(config_dir)


def test_pu_get_tools_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    tools_dir = repo / ".config" / "kon" / "bin"
    tools_dir.mkdir(parents=True)
    if pu.is_windows():
        monkeypatch.setenv("USERPROFILE", str(repo))
    else:
        monkeypatch.setenv("HOME", str(repo))
    assert pu.get_tools_dir() == str(tools_dir)


def test_pu_get_executable_ext_win(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", True)
    assert pu.get_executable_ext() == "exe"


def test_pu_get_executable_ext_posix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pu, "_IS_WINDOWS", False)
    assert pu.get_executable_ext() == ""


def test_pu_shorten_dir():
    assert pu.shorten_dir(pu.get_config_dir()) == str(Path("~/.config/kon"))


def test_pu_as_posix_dir():
    assert pu.as_posix_dir("c:\\test\\..\\test\\t.py") == "c:/test/../test/t.py"
    assert pu.as_posix_dir("/etc/test/p") == "/etc/test/p"
