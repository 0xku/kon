from kon import config, reset_config
from kon.config import _read_config_data
from kon.ui.commands import CommandsMixin


class FakeChat:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)


class FakeInfoBar:
    def __init__(self) -> None:
        self.permission_modes: list[str] = []

    def set_permission_mode(self, mode: str) -> None:
        self.permission_modes.append(mode)


class FakeCommands(CommandsMixin):
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.info_bar = FakeInfoBar()

    def query_one(self, selector, widget_type):
        if selector == "#chat-log":
            return self.chat
        if selector == "#info-bar":
            return self.info_bar
        raise AssertionError(f"Unexpected selector: {selector}")


def test_select_permission_mode_is_session_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_config()
    fake = FakeCommands()

    try:
        fake._select_permission_mode("auto")

        assert fake.info_bar.permission_modes == ["auto"]
        assert fake.chat.statuses == ["Permission mode changed to auto"]
        assert config.permissions.mode == "auto"

        # Permission mode is now persisted to disk
        config_file = tmp_path / ".kon" / "config.toml"
        if config_file.exists():
            data = _read_config_data(config_file)
            perms = data.get("permissions", {})
            assert perms.get("mode", "prompt") == "auto"
    finally:
        reset_config()
