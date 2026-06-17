"""OS, Platform, Environment functions"""

import os
import platform
import sys

_PLATFORM: str = sys.platform
_IS_WINDOWS: bool = _PLATFORM == "win32"
_IS_LINUX: bool = _PLATFORM == "linux"
_IS_MACOS: bool = _PLATFORM == "darwin"
_ARCH: str = platform.machine().lower()


def is_windows() -> bool:
    global _IS_WINDOWS
    return _IS_WINDOWS


def is_linux() -> bool:
    global _IS_LINUX
    return _IS_LINUX


def is_macos() -> bool:
    global _IS_MACOS
    return _IS_MACOS


def get_platform() -> str:
    global _PLATFORM
    if _PLATFORM in ("linux", "darwin", "win32"):
        return _PLATFORM
    return "linux"


def get_arch() -> str:
    global _ARCH
    return _ARCH


def get_git_root_dir(start: str) -> str | None:
    current = start
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.split(current)[0]
        if parent == current:
            return None
        current = parent


# def get_stop_directory(cwd: str) -> str:
#     git_root = get_git_root_dir(cwd)
#     if git_root:
#         return git_root

#     home = os.path.expanduser("~")
#     try:
#         os.path.relpath(cwd, start=home)
#         return home
#     except ValueError:
#         return cwd


def get_bash_dir() -> str | None:
    global _IS_WINDOWS
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
    return os.environ.get("SHELL") or "/bin/bash"


def get_config_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return os.path.join(base, "kon")
    return os.path.join(os.path.expanduser("~"), ".config", "kon")


def get_tools_dir() -> str:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return os.path.join(base, "kon", "bin")
    return os.path.join(os.path.expanduser("~"), ".config", "kon", "bin")


def get_agents_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".agents")


def get_home_dir() -> str:
    return os.path.expanduser("~")


def get_cwd() -> str:
    return os.getcwd()


def get_executable_ext() -> str:
    global _IS_WINDOWS
    return "exe" if _IS_WINDOWS else ""


# context/skills.py and tools/_tools_utils.py and ui/widgets.py
def shorten_dir(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home) :]
    return path


def as_posix_dir(path: str) -> str:
    return path.replace("\\", "/")
