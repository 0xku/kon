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
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

CONFIG_DIR_NAME: str = ".kon"

OnOverflowMode = Literal["continue", "pause"]


def _load_default_config_text() -> str:
    return resources.files("kon.defaults").joinpath("config.toml").read_text(encoding="utf-8")


_DEFAULT_CONFIG_DATA = tomllib.loads(_load_default_config_text())
CURRENT_CONFIG_VERSION = int(_DEFAULT_CONFIG_DATA.get("meta", {}).get("config_version", 1))

_config_var: ContextVar["Config | None"] = ContextVar("kon_config", default=None)
_config_warnings: list[str] = []


class MetaConfig(BaseModel):
    config_version: int = CURRENT_CONFIG_VERSION


class ToolBgConfig(BaseModel):
    pending: str
    success: str
    error: str


class BadgeColorConfig(BaseModel):
    bg: str
    label: str


class ColorsConfig(BaseModel):
    dim: str
    muted: str
    title: str
    spinner: str
    accent: str
    info: str
    markdown_heading: str
    markdown_code: str
    selected: str
    error: str
    notice: str
    diff_added: str
    diff_removed: str
    tool_bg: ToolBgConfig
    badge: BadgeColorConfig


class UIConfig(BaseModel):
    colors: ColorsConfig


class SystemPromptConfig(BaseModel):
    content: str
    git_context: bool = False


class LLMConfig(BaseModel):
    default_provider: str
    default_model: str
    default_base_url: str = ""
    default_thinking_level: str
    system_prompt: SystemPromptConfig
    tool_call_idle_timeout_seconds: float = 60


class CompactionConfig(BaseModel):
    on_overflow: OnOverflowMode = "continue"
    buffer_tokens: int = 20000


class AgentConfig(BaseModel):
    max_turns: int = 500
    default_context_window: int = 200000


class PermissionsConfig(BaseModel):
    mode: Literal["prompt", "auto"] = "prompt"


class ConfigSchema(BaseModel):
    meta: MetaConfig
    llm: LLMConfig
    ui: UIConfig
    compaction: CompactionConfig
    agent: AgentConfig
    permissions: PermissionsConfig


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
    def eza(self) -> bool:
        return "eza" in self._binaries


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

        ui_colors = normalized_data.get("ui", {}).get("colors")
        if isinstance(ui_colors, dict):
            if "badge" not in ui_colors and isinstance(ui_colors.get("compaction"), dict):
                ui_colors["badge"] = deepcopy(ui_colors["compaction"])
            if "notice" not in ui_colors and isinstance(ui_colors.get("warning"), str):
                ui_colors["notice"] = ui_colors["warning"]

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
    def binaries(self) -> _BinariesConfig:
        return _BinariesConfig(AVAILABLE_BINARIES)


def get_config_dir() -> Path:
    return Path.home() / CONFIG_DIR_NAME


def _ensure_config_file() -> Path:
    config_dir = get_config_dir()
    config_file = config_dir / "config.toml"

    if not config_file.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(_load_default_config_text(), encoding="utf-8")

    return config_file


def _record_config_warning(message: str) -> None:
    _config_warnings.append(message)
    print(message, file=sys.stderr)


def consume_config_warnings() -> list[str]:
    warnings = _config_warnings.copy()
    _config_warnings.clear()
    return warnings


def _detect_available_binaries() -> set[str]:
    binaries = {"rg", "fd", "eza"}
    available = set()
    bin_dir = Path.home() / CONFIG_DIR_NAME / "bin"

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


AVAILABLE_BINARIES = _detect_available_binaries()


def update_available_binaries() -> None:
    AVAILABLE_BINARIES.clear()
    AVAILABLE_BINARIES.update(_detect_available_binaries())


def _load_config() -> Config:
    config_file = _ensure_config_file()

    try:
        data = tomllib.loads(config_file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        _record_config_warning(
            f"Invalid config at {config_file}: {exc}. Falling back to built-in defaults."
        )
        data = {}

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


def reset_config() -> None:
    """Reset config to uninitialized state (next get_config() will reload from file)."""
    _config_var.set(None)
    _config_warnings.clear()
