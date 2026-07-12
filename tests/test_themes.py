import pytest

from kon import Config
from kon.themes import get_theme, get_theme_ids

NEW_THEME_IDS = [
    "ayu",
    "catppuccin-frappe",
    "catppuccin-macchiato",
    "everforest",
    "flexoki",
    "kanagawa",
    "monokai",
    "nightowl",
    "osaka-jade",
    "palenight",
    "rosepine",
]


@pytest.mark.parametrize("theme_id", NEW_THEME_IDS)
def test_new_themes_are_registered(theme_id: str):
    assert theme_id in get_theme_ids()


@pytest.mark.parametrize("theme_id", NEW_THEME_IDS)
def test_new_themes_are_loadable(theme_id: str):
    theme = get_theme(theme_id)

    assert theme.id == theme_id
    assert theme.colors.bg.startswith("#")
    assert theme.colors.fg.startswith("#")


@pytest.mark.parametrize("theme_id", NEW_THEME_IDS)
def test_new_themes_are_valid_config_values(theme_id: str):
    cfg = Config({"ui": {"theme": theme_id}})

    assert cfg.ui.theme == theme_id
