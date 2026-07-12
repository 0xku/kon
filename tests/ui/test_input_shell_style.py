from kon.ui.input import InputBox


class _FakeTextArea:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def clear(self) -> None:
        self.text = ""


class _TestableInputBox(InputBox):
    def __init__(self, text: str = "") -> None:
        super().__init__(cwd="/tmp")
        self._fake_textarea = _FakeTextArea(text)
        self.added_classes: set[str] = set()

    def query_one(self, *args, **kwargs):  # type: ignore[override]
        return self._fake_textarea

    def add_class(self, class_name: str) -> None:  # type: ignore[override]
        self.added_classes.add(class_name)

    def remove_class(self, class_name: str) -> None:  # type: ignore[override]
        self.added_classes.discard(class_name)

    def post_message(self, message) -> None:  # type: ignore[override]
        pass


def test_shell_command_style_is_added_for_bang_prefix() -> None:
    input_box = _TestableInputBox("!pwd")

    input_box._sync_shell_command_style()

    assert "-shell-command" in input_box.added_classes


def test_shell_command_style_is_added_for_double_bang_prefix() -> None:
    input_box = _TestableInputBox("!!git status")

    input_box._sync_shell_command_style()

    assert "-shell-command" in input_box.added_classes


def test_shell_command_style_ignores_leading_whitespace() -> None:
    input_box = _TestableInputBox("  !pwd")

    input_box._sync_shell_command_style()

    assert "-shell-command" in input_box.added_classes


def test_shell_command_style_is_removed_when_bang_prefix_goes_away() -> None:
    input_box = _TestableInputBox("!pwd")
    input_box._sync_shell_command_style()
    input_box._fake_textarea.text = "pwd"

    input_box._sync_shell_command_style()

    assert "-shell-command" not in input_box.added_classes


def test_clear_removes_shell_command_style() -> None:
    input_box = _TestableInputBox("!pwd")
    input_box._sync_shell_command_style()

    input_box.clear()

    assert "-shell-command" not in input_box.added_classes


def test_empty_editor_bangs_cycle_shell_modes() -> None:
    input_box = _TestableInputBox()

    assert input_box._handle_empty_key("!")
    assert input_box._shell_mode == 1
    assert "-shell-command" in input_box.added_classes
    assert "-shell-command-llm" not in input_box.added_classes

    assert input_box._handle_empty_key("!")
    assert input_box._shell_mode == 2
    assert "-shell-command-llm" in input_box.added_classes

    assert input_box._handle_empty_key("backspace")
    assert input_box._shell_mode == 1
    assert "-shell-command-llm" not in input_box.added_classes

    assert input_box._handle_empty_key("backspace")
    assert input_box._shell_mode == 0
    assert "-shell-command" not in input_box.added_classes
