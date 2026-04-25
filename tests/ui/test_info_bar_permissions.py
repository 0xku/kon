from typing import Any, cast

from kon import Config, reset_config, set_config
from kon.ui.widgets import InfoBar


class _FakeLabel:
    def __init__(self) -> None:
        self.content = None
        self.layout_values: list[bool] = []

    def update(self, content="", *, layout: bool = True) -> None:
        self.content = content
        self.layout_values.append(layout)


def test_info_bar_shows_auto_permission_mode_before_file_changes():
    set_config(Config({"permissions": {"mode": "auto"}}))
    try:
        info_bar = InfoBar("/tmp", "model")
        info_bar._file_changes = {"a.txt": (2, 1)}

        rendered = info_bar._format_row2_right()
    finally:
        reset_config()

    assert rendered.plain == "✓✓ auto • 1 file +2 -1"
    assert rendered.spans[0].style == "#d3869b"


def test_info_bar_shows_prompt_permission_mode_without_file_changes():
    set_config(Config({"permissions": {"mode": "prompt"}}))
    try:
        info_bar = InfoBar("/tmp", "model")

        rendered = info_bar._format_row2_right()
    finally:
        reset_config()

    assert rendered.plain == "⏸ prompt"


def test_info_bar_updates_permission_mode_without_layout():
    info_bar = InfoBar("/tmp", "model")
    label = _FakeLabel()
    info_bar._row2_left = cast(Any, label)

    info_bar.set_permission_mode("auto")

    assert cast(Any, label.content).plain == "✓✓ auto"
    assert label.layout_values == [False]
