from kon import Config, reset_config, set_config
from kon.ui.input import _get_textarea_theme


def test_input_cursor_uses_theme_foreground():
    set_config(Config({"ui": {"theme": "solarized-light"}}))

    try:
        theme = _get_textarea_theme()
    finally:
        reset_config()

    assert theme.cursor_style is not None
    assert theme.cursor_style.color is not None
    assert theme.cursor_style.bgcolor is not None

    assert theme.cursor_style.color.get_truecolor().hex == "#fdf6e3"
    assert theme.cursor_style.bgcolor.get_truecolor().hex == "#657b83"
    assert not theme.cursor_style.reverse
