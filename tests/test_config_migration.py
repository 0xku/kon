import tomllib
from pathlib import Path

from kon.config import CURRENT_CONFIG_VERSION, consume_config_warnings, get_config, reset_config


def test_old_config_is_migrated_and_backed_up(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = home / ".kon"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(
        """
[ui.colors]
warning = "#123456"

[ui.colors.compaction]
bg = "#111111"
label = "#222222"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(Path, "home", lambda: home)

    reset_config()
    cfg = get_config()

    assert cfg.ui.colors.notice == "#123456"
    assert cfg.ui.colors.badge.bg == "#111111"
    assert cfg.ui.colors.badge.label == "#222222"

    updated = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert updated["meta"]["config_version"] == CURRENT_CONFIG_VERSION

    backup_files = list(config_dir.glob("config.toml.bak.*"))
    assert len(backup_files) == 1

    warnings = consume_config_warnings()
    assert any("Migrated config" in warning for warning in warnings)


def test_current_version_config_is_not_rewritten(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = home / ".kon"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    original_text = (
        "[meta]\n"
        f"config_version = {CURRENT_CONFIG_VERSION}\n\n"
        "[llm]\n"
        'default_model = "custom-model"\n'
    )
    config_file.write_text(original_text, encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)

    reset_config()
    cfg = get_config()

    assert cfg.llm.default_model == "custom-model"
    assert config_file.read_text(encoding="utf-8") == original_text
    assert list(config_dir.glob("config.toml.bak.*")) == []

    warnings = consume_config_warnings()
    assert all("Migrated config" not in warning for warning in warnings)
