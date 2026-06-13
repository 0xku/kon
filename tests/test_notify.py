from pathlib import Path

import pytest

import kon.notify as mod
from kon.config import Config, set_config
from kon.notify import notify


@pytest.fixture(autouse=True)
def default_config():
    set_config(Config({}))
    yield
    set_config(Config({}))


def test_notify_plays_macos_sound(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_platform", lambda: "darwin")
    monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
    monkeypatch.setattr(mod, "_run", commands.append)

    notify("completion")

    assert commands == [["afplay", "-v", "0.5", "/sounds/completion.wav"]]


def test_notify_uses_configured_macos_volume(monkeypatch):
    commands: list[list[str]] = []
    set_config(Config({"notifications": {"volume": 0.25}}))

    monkeypatch.setattr(mod, "_platform", lambda: "darwin")
    monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
    monkeypatch.setattr(mod, "_run", commands.append)

    notify("completion")

    assert commands == [["afplay", "-v", "0.25", "/sounds/completion.wav"]]


def test_notify_plays_linux_sound_with_cached_player(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_platform", lambda: "linux")
    monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
    monkeypatch.setattr(mod, "_linux_player", lambda: "mpv")
    monkeypatch.setattr(mod, "_run", commands.append)

    notify("permission")

    assert commands == [
        [
            "mpv",
            "--no-video",
            "--no-terminal",
            "--script-opts=autoload-disabled=yes",
            "--volume=50.0",
            "/sounds/permission.wav",
        ]
    ]


def test_notify_uses_configured_linux_player_volumes(monkeypatch):
    set_config(Config({"notifications": {"volume": 0.25}}))

    cases = [
        ("paplay", ["paplay", "--volume=16384", "/sounds/error.wav"]),
        (
            "mpv",
            [
                "mpv",
                "--no-video",
                "--no-terminal",
                "--script-opts=autoload-disabled=yes",
                "--volume=25.0",
                "/sounds/error.wav",
            ],
        ),
        (
            "ffplay",
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-volume",
                "25",
                "/sounds/error.wav",
            ],
        ),
    ]

    for player, expected_command in cases:
        commands: list[list[str]] = []
        monkeypatch.setattr(mod, "_platform", lambda: "linux")
        monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
        monkeypatch.setattr(mod, "_linux_player", lambda player=player: player)
        monkeypatch.setattr(mod, "_run", commands.append)

        notify("error")

        assert commands == [expected_command]


def test_notify_plays_windows_sound(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_platform", lambda: "windows")
    monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
    monkeypatch.setattr(mod, "_run", commands.append)

    notify("error")

    assert commands == [
        ["powershell", "-c", "(New-Object Media.SoundPlayer '/sounds/error.wav').PlaySync();"]
    ]


def test_notify_ignores_unsupported_platform(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_platform", lambda: "freebsd")
    monkeypatch.setattr(mod, "_sound_path", lambda event: Path(f"/sounds/{event}.wav"))
    monkeypatch.setattr(mod, "_run", commands.append)

    notify("error")

    assert commands == []


def test_linux_player_prefers_paplay(monkeypatch):
    mod._linux_player.cache_clear()
    monkeypatch.setattr(
        mod.shutil, "which", lambda command: command if command == "paplay" else None
    )

    assert mod._linux_player() == "paplay"

    mod._linux_player.cache_clear()


def test_linux_player_falls_back_to_aplay(monkeypatch):
    mod._linux_player.cache_clear()
    monkeypatch.setattr(
        mod.shutil, "which", lambda command: command if command == "aplay" else None
    )

    assert mod._linux_player() == "aplay"

    mod._linux_player.cache_clear()
