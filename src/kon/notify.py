from __future__ import annotations

import platform
import shutil
import subprocess
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Literal

from kon import config

NotificationEvent = Literal["completion", "permission", "error"]

_SOUND_FILES: dict[NotificationEvent, str] = {
    "completion": "completion.wav",
    "permission": "permission.wav",
    "error": "error.wav",
}


@cache
def _platform() -> str:
    return platform.system().lower()


@cache
def _sound_path(event: NotificationEvent) -> Path:
    return Path(str(resources.files("kon.sounds").joinpath(_SOUND_FILES[event])))


@cache
def _linux_player() -> str | None:
    for player in ("paplay", "aplay", "mpv", "ffplay"):
        if shutil.which(player):
            return player
    return None


def _run(command: list[str]) -> None:
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _play_macos(sound_path: Path, volume: float) -> None:
    _run(["afplay", "-v", str(volume), str(sound_path)])


def _play_linux(sound_path: Path, volume: float) -> None:
    player = _linux_player()
    if player is None:
        return

    sound = str(sound_path)
    match player:
        case "paplay":
            _run(["paplay", f"--volume={round(volume * 65536)}", sound])
        case "aplay":
            _run(["aplay", sound])
        case "mpv":
            _run(
                [
                    "mpv",
                    "--no-video",
                    "--no-terminal",
                    "--script-opts=autoload-disabled=yes",
                    f"--volume={volume * 100}",
                    sound,
                ]
            )
        case "ffplay":
            _run(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-volume",
                    str(round(volume * 100)),
                    sound,
                ]
            )


def _play_windows(sound_path: Path, volume: float) -> None:
    # NOTE volume IGNORED!
    _run(
        [
            "powershell",
            "-c",
            "(New-Object Media.SoundPlayer '" + str(sound_path) + "').PlaySync();",
        ]
    )


def notify(event: NotificationEvent) -> None:
    sound_path = _sound_path(event)
    volume = config.notifications.volume
    os_name = _platform()

    try:
        if os_name == "darwin":
            _play_macos(sound_path, volume)
        elif os_name == "linux":
            _play_linux(sound_path, volume)
        elif os_name == "windows":
            _play_windows(sound_path, volume)
    except Exception:
        return
