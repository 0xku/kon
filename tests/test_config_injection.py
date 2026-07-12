"""Tests for injectable config functionality."""

import pytest

from kon import Config, config, get_config, reload_config, reset_config, set_config


def test_config_proxy_delegates_to_get_config():
    """Test that the module-level config proxy delegates to get_config()."""
    cfg = get_config()
    assert config.llm.default_provider == cfg.llm.default_provider
    assert config.llm.default_model == cfg.llm.default_model
    assert config.ui.colors.dim == cfg.ui.colors.dim


def test_set_config_injects_custom_config():
    """Test that set_config() allows injecting a custom config."""
    original_model = config.llm.default_model

    custom_data = {"llm": {"default_provider": "openai", "default_model": "test-model-123"}}
    custom_config = Config(custom_data)
    set_config(custom_config)

    try:
        assert config.llm.default_model == "test-model-123"
        assert get_config().llm.default_model == "test-model-123"
    finally:
        reset_config()

    assert config.llm.default_model == original_model


def test_reset_config_clears_cache():
    """Test that reset_config() clears the cached config."""
    original_model = config.llm.default_model

    custom_data = {"llm": {"default_model": "temporary-model"}}
    set_config(Config(custom_data))

    assert config.llm.default_model == "temporary-model"

    reset_config()

    assert config.llm.default_model == original_model


def test_reload_config_reloads_from_file():
    """Test that reload_config() reloads the config from file."""
    custom_data = {"llm": {"default_model": "will-be-replaced"}}
    set_config(Config(custom_data))

    assert config.llm.default_model == "will-be-replaced"

    reloaded = reload_config()

    assert reloaded.llm.default_model != "will-be-replaced"
    assert config.llm.default_model == reloaded.llm.default_model


def test_config_injection_for_testing():
    """Test a common testing pattern: inject config, run test, restore."""
    original_dim = config.ui.colors.dim

    test_config = Config({"ui": {"theme": "tokyo-night"}})
    set_config(test_config)

    try:
        assert config.ui.colors.dim == "#828bb8"
    finally:
        reset_config()

    assert config.ui.colors.dim == original_dim


@pytest.fixture
def custom_config():
    """Fixture that provides a custom config and resets after test."""
    test_data = {
        "llm": {"default_model": "fixture-model", "default_thinking_level": "high"},
        "ui": {"theme": "catppuccin-latte"},
    }
    test_config = Config(test_data)
    set_config(test_config)
    yield test_config
    reset_config()


def test_config_with_fixture(custom_config):
    """Test using a pytest fixture for config injection."""
    assert config.llm.default_model == "fixture-model"
    assert config.llm.default_thinking_level == "high"
    assert config.ui.colors.accent == "#1e66f5"
