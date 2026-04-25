from kon import Config, reset_config, set_config
from kon.ui.autocomplete import DEFAULT_COMMANDS, SlashCommand
from kon.ui.commands import CommandsMixin
from kon.ui.floating_list import ListItem
from kon.ui.selection_mode import SelectionMode


class FakeChat:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.statuses: list[str] = []

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        if error:
            self.errors.append(message)
        else:
            self.infos.append(message)

    def show_status(self, message: str) -> None:
        self.statuses.append(message)


class FakeFloatingList:
    def __init__(self) -> None:
        self.items: list[ListItem] = []
        self.searchable: bool | None = None

    def show(self, items: list[ListItem], searchable: bool = False) -> None:
        self.items = items
        self.searchable = searchable


class FakeInputBox:
    def __init__(self) -> None:
        self.cleared = False
        self.autocomplete_enabled: bool | None = None
        self.completing: bool | None = None
        self.focused = False

    def clear(self) -> None:
        self.cleared = True

    def set_autocomplete_enabled(self, enabled: bool) -> None:
        self.autocomplete_enabled = enabled

    def set_completing(self, completing: bool) -> None:
        self.completing = completing

    def focus(self) -> None:
        self.focused = True


class FakeCommands(CommandsMixin):
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.completion_list = FakeFloatingList()
        self.input_box = FakeInputBox()
        self._selection_mode = None
        self.selected_modes: list[str] = []

    def query_one(self, selector, widget_type):
        if selector == "#chat-log":
            return self.chat
        if selector == "#completion-list":
            return self.completion_list
        if selector == "#input-box":
            return self.input_box
        raise AssertionError(f"Unexpected selector: {selector}")

    def _select_permission_mode(self, mode):
        self.selected_modes.append(mode)


def test_permissions_command_in_default_commands():
    permissions_cmd = next((cmd for cmd in DEFAULT_COMMANDS if cmd.name == "permissions"), None)

    assert permissions_cmd is not None
    assert permissions_cmd.description == "Change permission mode"
    assert isinstance(permissions_cmd, SlashCommand)


def test_permissions_selection_mode():
    assert SelectionMode.PERMISSIONS == "permissions"
    assert SelectionMode.THINKING == "thinking"
    assert SelectionMode.NOTIFICATIONS == "notifications"


def test_permissions_command_with_argument_selects_mode():
    fake = FakeCommands()

    fake._handle_permissions_command("auto")

    assert fake.selected_modes == ["auto"]


def test_permissions_command_rejects_invalid_argument():
    fake = FakeCommands()

    fake._handle_permissions_command("bad")

    assert fake.selected_modes == []
    assert fake.chat.errors == ["Invalid permission mode: bad. Use one of: prompt, auto"]


def test_permissions_command_without_argument_opens_picker():
    set_config(Config({"permissions": {"mode": "auto"}}))
    fake = FakeCommands()

    try:
        fake._handle_permissions_command("")
    finally:
        reset_config()

    assert fake._selection_mode == SelectionMode.PERMISSIONS
    assert fake.input_box.cleared is True
    assert fake.input_box.autocomplete_enabled is False
    assert fake.input_box.completing is True
    assert fake.input_box.focused is True
    assert fake.completion_list.searchable is True
    assert [(item.value, item.label, item.description) for item in fake.completion_list.items] == [
        ("prompt", "prompt", "ask before mutating tool calls"),
        ("auto", "auto ✓", "allow tool calls without approval prompts"),
    ]
