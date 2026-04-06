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
[meta]
config_version = 2

[ui.colors]
warning = "#123456"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(Path, "home", lambda: home)

    reset_config()
    cfg = get_config()

    assert cfg.ui.theme == "gruvbox-dark"
    assert cfg.ui.colors.notice == "#fe8019"

    updated = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert updated["meta"]["config_version"] == CURRENT_CONFIG_VERSION
    assert updated["ui"]["theme"] == "gruvbox-dark"
    assert "colors" not in updated["ui"]
    assert updated["llm"]["auth"]["openai_compat"] == "auto"
    assert updated["llm"]["auth"]["anthropic_compat"] == "auto"

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
        'default_model = "custom-model"\n\n'
        "[llm.system_prompt]\n"
        'content = "custom prompt"\n'
    )
    config_file.write_text(original_text, encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)

    reset_config()
    cfg = get_config()

    assert cfg.llm.default_model == "custom-model"
    assert cfg.llm.system_prompt.content == "custom prompt"
    assert config_file.read_text(encoding="utf-8") == original_text
    assert list(config_dir.glob("config.toml.bak.*")) == []

    warnings = consume_config_warnings()
    assert all("Migrated config" not in warning for warning in warnings)


def test_v1_llm_system_prompt_keys_migrate_to_nested_section(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = home / ".kon"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(
        """
[meta]
config_version = 1

[llm]
default_model = "legacy-model"
system_prompt_git_context = true
system_prompt = "legacy prompt"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(Path, "home", lambda: home)

    reset_config()
    cfg = get_config()

    assert cfg.llm.default_model == "legacy-model"
    assert cfg.llm.system_prompt.content == "legacy prompt"
    assert cfg.llm.system_prompt.git_context is True

    updated = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert updated["meta"]["config_version"] == CURRENT_CONFIG_VERSION
    assert updated["llm"]["system_prompt"]["content"] == "legacy prompt"
    assert updated["llm"]["system_prompt"]["git_context"] is True

    warnings = consume_config_warnings()
    assert any("Migrated config" in warning for warning in warnings)
