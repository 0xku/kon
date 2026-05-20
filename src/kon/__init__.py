from kon.config import (
    AVAILABLE_BINARIES,
    CONFIG_DIR_NAME,
    Config,
    consume_config_warnings,
    get_config,
    get_config_dir,
    reload_config,
    reset_config,
    set_config,
    set_notifications_enabled,
    set_permissions_mode,
    set_show_welcome_shortcuts,
    set_theme,
    update_available_binaries,
)
from kon.context._xml import escape_xml


class _ConfigProxy:
    """Proxy that delegates to get_config() for runtime reloading and test injection."""

    def __getattr__(self, name: str):
        return getattr(get_config(), name)


config: Config = _ConfigProxy()  # type: ignore[assignment]

__all__ = [
    "AVAILABLE_BINARIES",
    "CONFIG_DIR_NAME",
    "Config",
    "config",
    "consume_config_warnings",
    "escape_xml",
    "get_config",
    "get_config_dir",
    "reload_config",
    "reset_config",
    "set_config",
    "set_notifications_enabled",
    "set_permissions_mode",
    "set_show_welcome_shortcuts",
    "set_theme",
    "update_available_binaries",
]
