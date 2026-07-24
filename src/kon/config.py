import contextlib
import os
import shutil
import sys
import tempfile
import tomllib
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator

from .themes import ColorsConfig, get_theme, get_theme_ids

CONFIG_DIR_NAME: str = "kon"

OnOverflowMode = Literal["continue", "pause"]
AuthMode = Literal["auto", "required", "none"]
PermissionMode = Literal["prompt", "auto"]
NotificationMode = Literal["on", "off"]
PERMISSION_MODES: tuple[PermissionMode, ...] = get_args(PermissionMode)
NOTIFICATION_MODES: tuple[NotificationMode, ...] = get_args(NotificationMode)


# =================================================================================================
# Persisted Config Schema and Defaults
# =================================================================================================


def _load_default_config_toml() -> str:
    return resources.files("kon.defaults").joinpath("config.toml").read_text(encoding="utf-8")


_DEFAULT_CONFIG_DATA = tomllib.loads(_load_default_config_toml())
CURRENT_CONFIG_VERSION = int(_DEFAULT_CONFIG_DATA.get("meta", {}).get("config_version", 1))

_config_var: ContextVar["Config | None"] = ContextVar("kon_config", default=None)
_config_warnings: list[str] = []


class MetaConfig(BaseModel):
    config_version: int = CURRENT_CONFIG_VERSION


ThinkingLinesOption = Literal["1", "2", "3", "4", "5", "none"]
THINKING_LINES_OPTIONS: tuple[ThinkingLinesOption, ...] = get_args(ThinkingLinesOption)


class UIConfig(BaseModel):
    theme: str = "gruvbox-dark"
    # When true, finalized thinking blocks are collapsed to a single line summary.
    # Set to false to always show the full thinking content.
    collapse_thinking: bool = True
    # Number of lines to show when thinking is collapsed. "none" means no truncation.
    thinking_lines: ThinkingLinesOption = "1"
    # When true, tool icon and name use badge label color on success.
    colored_tool_badge: bool = True
    # Show the list of keyboard shortcuts in the welcome section on launch.
    # Set to false to hide the shortcuts panel.
    show_welcome_shortcuts: bool = True
    # Models hidden from the /model picker. Use a provider name ("github-copilot")
    # to hide all its models, or "provider:model" to hide a specific model.
    # Hidden models remain usable via config defaults or session resume.
    hidden_models: list[str] = []

    @field_validator("theme")
    @classmethod
    def _validate_theme(cls, value: str) -> str:
        if value not in get_theme_ids():
            raise ValueError(f"Unknown theme: {value}")
        return value

    @property
    def colors(self) -> ColorsConfig:
        return get_theme(self.theme).colors


class SystemPromptConfig(BaseModel):
    content: str
    git_context: bool = False


class AuthConfig(BaseModel):
    openai_compat: AuthMode = "auto"
    anthropic_compat: AuthMode = "auto"


class TLSConfig(BaseModel):
    insecure_skip_verify: bool = False


class LLMConfig(BaseModel):
    default_provider: str
    default_model: str
    default_base_url: str = ""
    default_thinking_level: str
    system_prompt: SystemPromptConfig
    tool_call_idle_timeout_seconds: float = 180
    request_timeout_seconds: float = 600
    auth: AuthConfig = AuthConfig()
    tls: TLSConfig = TLSConfig()


class CompactionConfig(BaseModel):
    on_overflow: OnOverflowMode = "continue"
    buffer_tokens: int = 20000


class AgentConfig(BaseModel):
    max_turns: int = 500
    default_context_window: int = 200000
    turn_cooldown_seconds: float = 0


class PermissionsConfig(BaseModel):
    mode: PermissionMode = "prompt"


class ToolsConfig(BaseModel):
    extra: list[str] = []


class NotificationsConfig(BaseModel):
    enabled: bool = False
    volume: float = Field(default=0.5, ge=0.0, le=1.0)


class ConfigSchema(BaseModel):
    meta: MetaConfig
    llm: LLMConfig
    ui: UIConfig
    compaction: CompactionConfig
    agent: AgentConfig
    tools: ToolsConfig = ToolsConfig()
    permissions: PermissionsConfig
    notifications: NotificationsConfig = NotificationsConfig()


# =================================================================================================
# Runtime Config Accessors
# =================================================================================================


class _BinariesConfig:
    def __init__(self, binaries: set[str]) -> None:
        self._binaries = binaries

    def has(self, binary: str) -> bool:
        return binary in self._binaries

    @property
    def rg(self) -> bool:
        return "rg" in self._binaries

    @property
    def fd(self) -> bool:
        return "fd" in self._binaries

    @property
    def gh(self) -> bool:
        return "gh" in self._binaries


class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        merged = self.merge_with_defaults(data)
        self._parsed = ConfigSchema.model_validate(merged)

    @staticmethod
    def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in overrides.items():
            current_value = merged.get(key)
            if isinstance(current_value, dict) and isinstance(value, dict):
                merged[key] = Config.deep_merge(current_value, value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _apply_legacy_key_shims(data: dict[str, Any]) -> dict[str, Any]:
        normalized_data = deepcopy(data)

        llm = normalized_data.get("llm")
        if isinstance(llm, dict):
            legacy_prompt = llm.get("system_prompt")
            if isinstance(legacy_prompt, str):
                llm["system_prompt"] = {"content": legacy_prompt}

            legacy_git_context = llm.pop("system_prompt_git_context", None)
            if isinstance(legacy_git_context, bool):
                system_prompt = llm.get("system_prompt")
                if not isinstance(system_prompt, dict):
                    system_prompt = {}
                    llm["system_prompt"] = system_prompt
                system_prompt.setdefault("git_context", legacy_git_context)

        return normalized_data

    @staticmethod
    def merge_with_defaults(data: dict[str, Any]) -> dict[str, Any]:
        normalized_data = Config._apply_legacy_key_shims(data)
        return Config.deep_merge(_DEFAULT_CONFIG_DATA, normalized_data)

    @property
    def llm(self) -> LLMConfig:
        return self._parsed.llm

    @property
    def ui(self) -> UIConfig:
        return self._parsed.ui

    @property
    def compaction(self) -> CompactionConfig:
        return self._parsed.compaction

    @property
    def agent(self) -> AgentConfig:
        return self._parsed.agent

    @property
    def permissions(self) -> PermissionsConfig:
        return self._parsed.permissions

    @property
    def tools(self) -> ToolsConfig:
        return self._parsed.tools

    @property
    def notifications(self) -> NotificationsConfig:
        return self._parsed.notifications

    @property
    def binaries(self) -> _BinariesConfig:
        return _BinariesConfig(AVAILABLE_BINARIES)


# =================================================================================================
# Persisted Config IO, Migration, and Serialization
# =================================================================================================


def get_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def get_agents_dir() -> Path:
    return Path.home() / ".agents"


def _ensure_config_file() -> Path:
    config_dir = get_config_dir()
    config_file = config_dir / "config.toml"

    if not config_file.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(_load_default_config_toml(), encoding="utf-8")

    return config_file


def _record_config_warning(message: str) -> None:
    _config_warnings.append(message)
    print(message, file=sys.stderr)


def consume_config_warnings() -> list[str]:
    warnings = _config_warnings.copy()
    _config_warnings.clear()
    return warnings


def _detect_available_binaries() -> set[str]:
    binaries = {"rg", "fd", "gh"}
    available = set()
    bin_dir = get_config_dir() / "bin"

    for binary in binaries:
        if shutil.which(binary) or (bin_dir / binary).exists():
            available.add(binary)

    return available


def _get_config_version(data: dict[str, Any]) -> int:
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return 0
    version = meta.get("config_version")
    if isinstance(version, int) and version >= 0:
        return version
    return 0


def _migrate_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 1}
    else:
        meta["config_version"] = 1
    return migrated


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 2}
    else:
        meta["config_version"] = 2
    return migrated


def _migrate_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    ui = migrated.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        migrated["ui"] = ui

    ui["theme"] = "gruvbox-dark"
    ui.pop("colors", None)

    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 3}
    else:
        meta["config_version"] = 3
    return migrated


def _migrate_v3_to_v4(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    llm = migrated.get("llm")
    if not isinstance(llm, dict):
        llm = {}
        migrated["llm"] = llm

    auth = llm.get("auth")
    if not isinstance(auth, dict):
        auth = {}
        llm["auth"] = auth

    auth.setdefault("openai_compat", "auto")
    auth.setdefault("anthropic_compat", "auto")

    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 4}
    else:
        meta["config_version"] = 4
    return migrated


def _migrate_v4_to_v5(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    notifications = migrated.get("notifications")
    if not isinstance(notifications, dict):
        notifications = {}
        migrated["notifications"] = notifications

    notifications.setdefault("volume", 0.5)

    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 5}
    else:
        meta["config_version"] = 5
    return migrated


def _migrate_v5_to_v6(data: dict[str, Any]) -> dict[str, Any]:
    migrated = Config._apply_legacy_key_shims(data)
    llm = migrated.get("llm")
    if not isinstance(llm, dict):
        llm = {}
        migrated["llm"] = llm

    system_prompt = llm.get("system_prompt")
    if not isinstance(system_prompt, dict):
        system_prompt = {}
        llm["system_prompt"] = system_prompt

    system_prompt["content"] = _DEFAULT_CONFIG_DATA["llm"]["system_prompt"]["content"]
    system_prompt["git_context"] = _DEFAULT_CONFIG_DATA["llm"]["system_prompt"]["git_context"]

    meta = migrated.get("meta")
    if not isinstance(meta, dict):
        migrated["meta"] = {"config_version": 6}
    else:
        meta["config_version"] = 6
    return migrated


def _migrate_config_data(data: dict[str, Any]) -> tuple[dict[str, Any], int, int, bool]:
    original = deepcopy(data)
    current_version = _get_config_version(original)
    migrated = deepcopy(original)

    while current_version < CURRENT_CONFIG_VERSION:
        if current_version == 0:
            migrated = _migrate_v0_to_v1(migrated)
            current_version = 1
            continue
        if current_version == 1:
            migrated = _migrate_v1_to_v2(migrated)
            current_version = 2
            continue
        if current_version == 2:
            migrated = _migrate_v2_to_v3(migrated)
            current_version = 3
            continue
        if current_version == 3:
            migrated = _migrate_v3_to_v4(migrated)
            current_version = 4
            continue
        if current_version == 4:
            migrated = _migrate_v4_to_v5(migrated)
            current_version = 5
            continue
        if current_version == 5:
            migrated = _migrate_v5_to_v6(migrated)
            current_version = 6
            continue
        break

    migrated_version = _get_config_version(migrated)
    did_migrate = migrated != original
    return migrated, _get_config_version(original), migrated_version, did_migrate


def _toml_escape_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _toml_format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return _toml_escape_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_format_value(item) for item in value) + "]"
    raise TypeError(
        f"Unsupported config value type for TOML serialization: {type(value).__name__}"
    )


def _toml_dump_dict(data: dict[str, Any], table: str | None = None) -> str:
    lines: list[str] = []

    scalar_items = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    dict_items = [(k, v) for k, v in data.items() if isinstance(v, dict)]

    if table is not None:
        lines.append(f"[{table}]")

    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_format_value(value)}")

    if dict_items and lines:
        lines.append("")

    for idx, (key, value) in enumerate(dict_items):
        nested_table = f"{table}.{key}" if table else key
        nested = _toml_dump_dict(value, nested_table)
        if nested:
            lines.append(nested)
        if idx < len(dict_items) - 1:
            lines.append("")

    return "\n".join(lines)


def _serialize_config_toml(data: dict[str, Any]) -> str:
    return _toml_dump_dict(data) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _backup_and_write_migrated_config(config_file: Path, data: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = config_file.with_name(f"{config_file.name}.bak.{timestamp}")
    shutil.copy2(config_file, backup_path)
    _atomic_write_text(config_file, _serialize_config_toml(data))
    return backup_path


# =================================================================================================
# Runtime Environment Capabilities
# TODO: Consider moving runtime capability detection and caching to a dedicated runtime.py module.
# =================================================================================================


AVAILABLE_BINARIES = _detect_available_binaries()


def update_available_binaries() -> None:
    AVAILABLE_BINARIES.clear()
    AVAILABLE_BINARIES.update(_detect_available_binaries())


# =================================================================================================
# Persisted Config Loading and Runtime Cache
# =================================================================================================


def _read_config_data(config_file: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(config_file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        _record_config_warning(
            f"Invalid config at {config_file}: {exc}. Falling back to built-in defaults."
        )
        return {}


def _load_config() -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    try:
        migrated_data, from_version, to_version, did_migrate = _migrate_config_data(data)
        if did_migrate and data:
            try:
                backup = _backup_and_write_migrated_config(config_file, migrated_data)
                _record_config_warning(
                    f"Migrated config at {config_file} from v{from_version} to v{to_version}. "
                    f"Backup saved to {backup}."
                )
            except Exception as exc:
                _record_config_warning(
                    f"Failed to persist migrated config at {config_file}: {exc}. "
                    "Continuing with in-memory migrated config."
                )
        return Config(migrated_data)
    except ValidationError as exc:
        _record_config_warning(
            f"Invalid config values at {config_file}: {exc}. Falling back to built-in defaults."
        )
        return Config({})


def get_config() -> Config:
    """
    Get the current config instance.

    Returns the config from context variable if set, otherwise loads from file.
    The loaded config is cached in the context variable.
    """
    cfg = _config_var.get()
    if cfg is None:
        cfg = _load_config()
        _config_var.set(cfg)
    return cfg


def set_config(config: Config) -> None:
    """Set the config instance (useful for testing)."""
    _config_var.set(config)


def reload_config() -> Config:
    """Reload config from file and update the context variable."""
    cfg = _load_config()
    _config_var.set(cfg)
    return cfg


def _set_config_version(data: dict[str, Any]) -> None:
    meta = data.get("meta")
    if not isinstance(meta, dict):
        data["meta"] = {"config_version": CURRENT_CONFIG_VERSION}
    else:
        meta["config_version"] = CURRENT_CONFIG_VERSION


def set_theme(theme: str) -> Config:
    get_theme(theme)

    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui

    ui["theme"] = theme
    ui.pop("colors", None)
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def set_show_welcome_shortcuts(enabled: bool) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui

    ui["show_welcome_shortcuts"] = enabled
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def set_permissions_mode(mode: PermissionMode) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    perms = data.get("permissions")
    if not isinstance(perms, dict):
        perms = {}
        data["permissions"] = perms

    perms["mode"] = mode
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))

    # ContextVars are copied when Textual starts the agent worker. Mutate the
    # shared config instance so an already-running worker sees the new mode,
    # then reload to keep the caller's context in sync with the persisted file.
    get_config().permissions.mode = mode
    return reload_config()


def set_thinking_lines(lines: ThinkingLinesOption) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui

    ui["thinking_lines"] = lines
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def set_git_context(enabled: bool) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    llm = data.get("llm")
    if not isinstance(llm, dict):
        llm = {}
        data["llm"] = llm

    system_prompt = llm.get("system_prompt")
    if not isinstance(system_prompt, dict):
        system_prompt = {}
        llm["system_prompt"] = system_prompt

    system_prompt["git_context"] = enabled
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def set_colored_tool_badge(enabled: bool) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui

    ui["colored_tool_badge"] = enabled
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def set_notifications_enabled(enabled: bool) -> Config:
    config_file = _ensure_config_file()
    data = _read_config_data(config_file)

    notifications = data.get("notifications")
    if not isinstance(notifications, dict):
        notifications = {}
        data["notifications"] = notifications

    notifications["enabled"] = enabled
    _set_config_version(data)

    _atomic_write_text(config_file, _serialize_config_toml(data))
    return reload_config()


def reset_config() -> None:
    """Reset config to uninitialized state (next get_config() will reload from file)."""
    _config_var.set(None)
    _config_warnings.clear()
