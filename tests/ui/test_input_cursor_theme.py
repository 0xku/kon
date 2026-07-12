from kon import Config, reset_config, set_config
from kon.ui.input import _get_textarea_theme


def test_input_cursor_and_selection_use_theme_foreground():
    set_config(Config({"ui": {"theme": "solarized-light"}}))

    try:
        theme = _get_textarea_theme()
    finally:
        reset_config()

    for style in (theme.cursor_style, theme.selection_style):
        assert style is not None
        assert style.color is not None
        assert style.bgcolor is not None
        assert style.color.get_truecolor().hex == "#fdf6e3"
        assert style.bgcolor.get_truecolor().hex == "#657b83"
        assert not style.reverse
