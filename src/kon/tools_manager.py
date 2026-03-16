import asyncio
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiohttp

from .config import CONFIG_DIR_NAME

ToolName = Literal["fd", "rg", "eza"]

_BIN_DIR = Path.home() / CONFIG_DIR_NAME / "bin"


@dataclass
class _ToolConfig:
    name: str
    repo: str
    binary_name: str
    tag_prefix: str

    def get_asset_name(self, version: str, plat: str, arch: str) -> str | None:
        raise NotImplementedError


class _FdConfig(_ToolConfig):
    def get_asset_name(self, version: str, plat: str, arch: str) -> str | None:
        if plat == "darwin":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"fd-v{version}-{arch_str}-apple-darwin.tar.gz"
        elif plat == "linux":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"fd-v{version}-{arch_str}-unknown-linux-gnu.tar.gz"
        elif plat == "win32":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"fd-v{version}-{arch_str}-pc-windows-msvc.zip"
        return None


class _RgConfig(_ToolConfig):
    def get_asset_name(self, version: str, plat: str, arch: str) -> str | None:
        if plat == "darwin":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"ripgrep-{version}-{arch_str}-apple-darwin.tar.gz"
        elif plat == "linux":
            if arch == "arm64":
                return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
            return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
        elif plat == "win32":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"ripgrep-{version}-{arch_str}-pc-windows-msvc.zip"
        return None


class _EzaConfig(_ToolConfig):
    def get_asset_name(self, version: str, plat: str, arch: str) -> str | None:
        if plat == "linux":
            arch_str = "aarch64" if arch == "arm64" else "x86_64"
            return f"eza_{arch_str}-unknown-linux-gnu.tar.gz"
        elif plat == "win32":
            return "eza.exe_x86_64-pc-windows-gnu.zip"
        # No macOS binary published by eza-community
        return None


_TOOLS: dict[ToolName, _ToolConfig] = {
    "fd": _FdConfig(name="fd", repo="sharkdp/fd", binary_name="fd", tag_prefix="v"),
    "rg": _RgConfig(name="ripgrep", repo="BurntSushi/ripgrep", binary_name="rg", tag_prefix=""),
    "eza": _EzaConfig(name="eza", repo="eza-community/eza", binary_name="eza", tag_prefix="v"),
}


def _get_platform() -> str:
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "win32"
    return "linux"


def _get_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x86_64"


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def get_tool_path(tool: ToolName) -> str | None:
    config = _TOOLS.get(tool)
    if not config:
        return None

    ext = ".exe" if _get_platform() == "win32" else ""
    local_path = _BIN_DIR / (config.binary_name + ext)
    if local_path.exists():
        return str(local_path)

    if _command_exists(config.binary_name):
        return config.binary_name

    return None


async def _get_latest_version(session: aiohttp.ClientSession, repo: str) -> str:
    async with session.get(
        f"https://api.github.com/repos/{repo}/releases/latest", headers={"User-Agent": "kon"}
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        version = data["tag_name"].removeprefix("v")
        if not re.match(r"^\d+\.\d+(\.\d+)?$", version):
            raise ValueError(f"Unexpected version format: {version}")
        return version


async def _download_file(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    async with session.get(url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.content.iter_chunked(8192):
                f.write(chunk)


def _extract_binary(archive_path: Path, binary_name: str, dest: Path) -> Path:
    ext = ".exe" if _get_platform() == "win32" else ""
    target_binary = binary_name + ext
    output_path = dest / target_binary

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        if str(archive_path).endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
        elif str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp)

        # Search for the binary: could be at top level or in a subdirectory
        candidates = list(tmp.rglob(target_binary))
        if not candidates:
            raise FileNotFoundError(f"Binary {target_binary} not found in archive")

        shutil.move(str(candidates[0]), str(output_path))

    if _get_platform() != "win32":
        output_path.chmod(output_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return output_path


async def _download_tool(tool: ToolName) -> str:
    config = _TOOLS[tool]
    plat = _get_platform()
    arch = _get_arch()

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        version = await _get_latest_version(session, config.repo)

        asset_name = config.get_asset_name(version, plat, arch)
        if not asset_name:
            raise RuntimeError(f"No binary available for {config.name} on {plat}/{arch}")

        _BIN_DIR.mkdir(parents=True, exist_ok=True)

        download_url = (
            f"https://github.com/{config.repo}/releases/download/"
            f"{config.tag_prefix}{version}/{asset_name}"
        )

        archive_path = _BIN_DIR / asset_name
        try:
            await _download_file(session, download_url, archive_path)
            binary_path = _extract_binary(archive_path, config.binary_name, _BIN_DIR)
            return str(binary_path)
        finally:
            archive_path.unlink(missing_ok=True)


async def ensure_tool(tool: ToolName, silent: bool = False) -> str | None:
    existing = get_tool_path(tool)
    if existing:
        return existing

    config = _TOOLS.get(tool)
    if not config:
        return None

    asset_name = config.get_asset_name("0", _get_platform(), _get_arch())
    if not asset_name:
        return None

    if not silent:
        print(f"{config.name} not found. Downloading...", file=sys.stderr)

    try:
        path = await _download_tool(tool)
        if not silent:
            print(f"{config.name} installed to {path}", file=sys.stderr)
        return path
    except Exception as e:
        if not silent:
            print(f"Failed to download {config.name}: {e}", file=sys.stderr)
        return None


async def ensure_tools(
    tools: list[ToolName] | None = None, silent: bool = False
) -> dict[ToolName, str | None]:
    if tools is None:
        tools = ["fd", "rg", "eza"]
    results = await asyncio.gather(*(ensure_tool(t, silent=silent) for t in tools))
    return dict(zip(tools, results, strict=True))
