import asyncio
import glob
import os
import shutil
import sys
import time
import tomllib
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar, Literal

from rich.console import Console
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding

from kon import config, consume_config_warnings, update_available_binaries
from kon.tools_manager import ensure_tools

from ..context.skills import load_skills
from ..core.types import StopReason
from ..events import (
    AgentEndEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    RetryEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolApprovalEvent,
    ToolArgsTokenUpdateEvent,
    ToolEndEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    WarningEvent,
)
from ..llm import (
    API_TYPE_TO_PROVIDER_CLASS,
    PROVIDER_API_BY_NAME,
    ApiType,
    BaseProvider,
    ProviderConfig,
    get_max_tokens,
    get_model,
    is_copilot_logged_in,
    is_openai_logged_in,
    resolve_provider_api_type,
)
from ..loop import Agent
from ..permissions import ApprovalResponse
from ..session import Session
from ..tools import DEFAULT_TOOLS, get_tools
from ..update_check import get_newer_pypi_version
from .autocomplete import DEFAULT_COMMANDS, FilePathProvider, SlashCommand, SlashCommandProvider
from .blocks import HandoffLinkBlock, LaunchWarning
from .chat import ChatLog
from .commands import CommandsMixin
from .floating_list import FloatingList
from .input import InputBox
from .selection_mode import SelectionMode
from .session_ui import SessionUIMixin
from .styles import STYLES
from .widgets import InfoBar, QueueDisplay, StatusLine, format_path


def _get_package_name() -> str:
    pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text())
            return data["project"]["name"]
        except Exception:
            pass
    return "kon-coding-agent"


_PYPI_PACKAGE_NAME = _get_package_name()
_CHANGELOG_URL = "https://github.com/0xku/kon/blob/main/CHANGELOG.md"

try:
    VERSION = version(_PYPI_PACKAGE_NAME)
except PackageNotFoundError:
    VERSION = "0.3.0"

_COPILOT_API_TYPES: frozenset[ApiType] = frozenset(
    {ApiType.GITHUB_COPILOT, ApiType.GITHUB_COPILOT_RESPONSES, ApiType.ANTHROPIC_COPILOT}
)

_OPENAI_OAUTH_API_TYPES: frozenset[ApiType] = frozenset({ApiType.OPENAI_CODEX_RESPONSES})

_API_TYPE_BY_PROVIDER: dict[type[BaseProvider], ApiType] = {
    v: k for k, v in API_TYPE_TO_PROVIDER_CLASS.items()
}


def _default_base_url_for_api(api_type: ApiType) -> str | None:
    if api_type == ApiType.OPENAI_COMPLETIONS:
        return os.environ.get("KON_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    return None


class Kon(CommandsMixin, SessionUIMixin, App[None]):
    CSS = STYLES
    TITLE = "kon"
    VERSION = VERSION

    BINDINGS: ClassVar[list] = [
        ("ctrl+c", "handle_ctrl_c", "Clear"),
        Binding("ctrl+d", "handle_ctrl_d", "Delete session", priority=True),
        ("escape", "interrupt_agent", "Interrupt"),
        ("ctrl+t", "toggle_thinking", "Toggle thinking"),
        Binding("shift+tab", "cycle_thinking_level", "Cycle thinking level", priority=True),
    ]

    def __init__(
        self,
        cwd: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        resume_session: str | None = None,
        continue_recent: bool = False,
        thinking_level: str | None = None,
    ):
        super().__init__()
        self.theme = "textual-ansi"
        self._cwd = cwd or os.getcwd()
        self._model = model or config.llm.default_model
        self._model_provider: str | None = (
            provider
            if provider is not None
            else (config.llm.default_provider if model is None else None)
        )
        self._api_key = api_key
        self._base_url = base_url or config.llm.default_base_url or None
        self._resume_session = resume_session
        self._continue_recent = continue_recent
        self._thinking_level = thinking_level or config.llm.default_thinking_level
        self._is_running = False
        self._last_ctrl_c_time = 0.0
        self._last_ctrl_d_time = 0.0
        self._ctrl_c_threshold = 2.0
        self._ctrl_d_threshold = 2.0
        self._ctrl_c_timer = None
        self._ctrl_d_timer = None
        self._cancel_event: asyncio.Event | None = None
        self._interrupt_requested = False
        self._abort_shown = False
        self._current_block_type: str | None = None
        self._approval_future: asyncio.Future[ApprovalResponse] | None = None
        self._approval_tool_id: str | None = None
        self._hide_thinking = False
        self._fd_path: str | None = None
        self._selection_mode: SelectionMode | None = None

        self._pending_queue: deque[tuple[str, str]] = deque(maxlen=QueueDisplay.MAX_QUEUE)
        self._exit_hints: list[str] = []
        self._session_start_time: float | None = None

        self._provider: BaseProvider | None = None
        self._session: Session | None = None
        self._agent: Agent | None = None

        self._pending_update_notice_version: str | None = None
        self._update_notice_shown = False
        self._startup_complete = False
        self._launch_warnings: list[LaunchWarning] = []

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")
        yield QueueDisplay(id="queue-display")
        yield StatusLine(id="status-line")
        yield InputBox(cwd=self._cwd, id="input-box")
        yield FloatingList(window_size=5, label_width=12, id="completion-list")
        yield InfoBar(
            cwd=self._cwd,
            model=self._model,
            thinking_level=self._thinking_level,
            hide_thinking=self._hide_thinking,
            id="info-bar",
        )

    def _sync_slash_commands(self) -> None:
        input_box = self.query_one("#input-box", InputBox)
        commands = DEFAULT_COMMANDS.copy()

        skills = self._agent.context.skills if self._agent else load_skills(self._cwd).skills
        for skill in skills:
            if not skill.register_cmd:
                continue
            cmd_description = skill.cmd_info
            if not cmd_description:
                cmd_description = skill.description[:32]
                if len(skill.description) > 32:
                    cmd_description = f"{cmd_description}..."
            commands.append(
                SlashCommand(name=skill.name, description=cmd_description, is_skill=True)
            )

        input_box.set_commands(commands)

    @staticmethod
    def _build_skill_trigger_message(skill_name: str, description: str, query: str) -> str:
        truncated_description = description[:300]
        if len(description) > 300:
            truncated_description = f"{truncated_description}..."

        parts = [f"[{skill_name}]", truncated_description]
        if query.strip():
            parts.extend(["", "[query]", query.strip()])
        return "\n".join(parts)

    def _create_provider(self, api_type: ApiType, config: ProviderConfig) -> BaseProvider:
        if api_type in _COPILOT_API_TYPES and not is_copilot_logged_in():
            raise ValueError("Not logged in to GitHub Copilot. Use /login to authenticate.")
        if api_type in _OPENAI_OAUTH_API_TYPES and not is_openai_logged_in():
            raise ValueError("Not logged in to OpenAI. Use /login to authenticate.")
        cls = API_TYPE_TO_PROVIDER_CLASS.get(api_type)
        if cls is None:
            raise ValueError(f"Unsupported API type: {api_type.value}")
        return cls(config)

    def _get_provider_api_type(self, provider: BaseProvider) -> ApiType:
        return _API_TYPE_BY_PROVIDER.get(type(provider), ApiType.OPENAI_COMPLETIONS)

    @on(events.TextSelected)
    def _on_text_selected(self) -> None:
        selection = self.screen.get_selected_text()
        if selection:
            self.copy_to_clipboard(selection)

    def on_mount(self) -> None:
        if config.binaries.fd:
            self._fd_path = shutil.which("fd") or shutil.which("fdfind")
        else:
            self._fd_path = None

        input_box = self.query_one("#input-box", InputBox)
        input_box.set_fd_path(self._fd_path)
        input_box.set_commands(DEFAULT_COMMANDS.copy())

        if not self._fd_path:
            self.run_worker(self._collect_file_paths(), exclusive=False)

        self.run_worker(self._ensure_binaries(), exclusive=False)
        self.run_worker(self._check_for_updates(), exclusive=False)

        if self._resume_session:
            try:
                self._session = Session.continue_by_id(self._cwd, self._resume_session)
            except Exception as e:
                self._add_launch_warning(str(e), severity="error")
                chat = self.query_one("#chat-log", ChatLog)
                self._flush_launch_warnings(chat)
                return
            if self._session.entries:
                model_info = self._session.model
                if model_info:
                    provider, self._model, session_base_url = model_info
                    self._model_provider = provider
                    if self._base_url is None and session_base_url:
                        self._base_url = session_base_url
                self._thinking_level = self._session.thinking_level
        elif self._continue_recent:
            try:
                self._session = Session.continue_recent(self._cwd)
            except Exception as e:
                self._add_launch_warning(str(e), severity="error")
                chat = self.query_one("#chat-log", ChatLog)
                self._flush_launch_warnings(chat)
                return
            if self._session.entries:
                model_info = self._session.model
                if model_info:
                    provider, self._model, session_base_url = model_info
                    self._model_provider = provider
                    if self._base_url is None and session_base_url:
                        self._base_url = session_base_url
                self._thinking_level = self._session.thinking_level

        model_info = get_model(self._model, self._model_provider)
        if model_info:
            api_type = model_info.api
            base_url = self._base_url or model_info.base_url
        else:
            try:
                api_type = resolve_provider_api_type(self._model_provider)
            except ValueError as e:
                self._add_launch_warning(str(e), severity="error")
                chat = self.query_one("#chat-log", ChatLog)
                self._flush_launch_warnings(chat)
                return
            base_url = self._base_url or _default_base_url_for_api(api_type)

        provider_config = ProviderConfig(
            api_key=self._api_key,
            base_url=base_url,
            model=self._model,
            max_tokens=get_max_tokens(self._model),
            thinking_level=self._thinking_level,
            provider=self._model_provider,
            session_id=self._session.id if self._session else None,
        )

        provider_error: str | None = None
        try:
            self._provider = self._create_provider(api_type, provider_config)
        except ValueError as e:
            provider_error = str(e)

        if self._provider:
            valid_levels = self._provider.thinking_levels
            if self._thinking_level not in valid_levels:
                self._thinking_level = valid_levels[0] if valid_levels else "medium"
                self._provider.set_thinking_level(self._thinking_level)

        self._session_start_time = time.time()

        if not self._continue_recent and not self._resume_session:
            selected_model = get_model(self._model, self._model_provider)
            model_provider = (
                selected_model.provider
                if selected_model
                else (self._provider.name if self._provider else self._model_provider)
            )
            self._model_provider = model_provider
            self._session = Session.create(
                self._cwd,
                provider=model_provider,
                model_id=self._model,
                thinking_level=self._thinking_level,
            )
            if model_provider:
                self._session.append_model_change(model_provider, self._model, base_url)

        # Create Agent once — it owns context + system prompt (stable across queries
        # for prompt-prefix caching on llama-server and similar engines).
        if self._provider is not None and self._session is not None:
            self._agent = Agent(
                provider=self._provider,
                tools=get_tools(DEFAULT_TOOLS),
                session=self._session,
                cwd=self._cwd,
            )

        self._sync_slash_commands()

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_session_info(VERSION)

        if self._agent:
            chat.add_loaded_resources(
                context_paths=[format_path(f.path) for f in self._agent.context.agents_files],
                skill_paths=[format_path(s.path) for s in self._agent.context.skills],
            )
            for path, message in self._agent.context.skill_warnings:
                self._add_launch_warning(f"Skill warning in {format_path(path)}: {message}")

        if provider_error:
            self._add_launch_warning(provider_error, severity="error")

        for warning in consume_config_warnings():
            self._add_launch_warning(warning)

        self._flush_launch_warnings(chat)

        info_bar = self.query_one("#info-bar", InfoBar)
        if self._session:
            info_bar.set_session_id(self._session.id[:8])
        model_provider = (
            model_info.provider
            if model_info
            else (self._model_provider or (self._provider.name if self._provider else None))
        )
        self._model_provider = model_provider
        info_bar.set_model(self._model, model_provider)
        info_bar.set_thinking_level(self._thinking_level)

        if (
            (self._continue_recent or self._resume_session)
            and self._session
            and self._session.entries
        ):
            self._render_session_entries(self._session)
            input_t, output_t, context_t, cache_read_t, cache_write_t = (
                self._calculate_session_tokens(self._session)
            )
            info_bar.set_tokens(input_t, output_t, context_t, cache_read_t, cache_write_t)
            info_bar.set_file_changes(self._calculate_session_file_changes(self._session))
            chat.add_info_message("Resumed session")

        if self._provider and self._session:
            self._provider.config.session_id = self._session.id

        self._startup_complete = True
        self._show_pending_update_notice_if_idle()
        input_box.focus()

    async def _collect_file_paths(self) -> None:
        """Collect file paths using glob (fallback when fd is unavailable)."""
        patterns = [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.json",
            "**/*.md",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
        ]
        paths = []
        for pattern in patterns:
            for path in glob.glob(os.path.join(self._cwd, pattern), recursive=True):
                rel_path = os.path.relpath(path, self._cwd)
                if not rel_path.startswith(
                    (".git", "node_modules", "__pycache__", ".venv", "venv")
                ):
                    paths.append(rel_path)
        paths = sorted(paths)
        self.query_one("#input-box", InputBox).set_file_paths(paths)

    async def _ensure_binaries(self) -> None:
        paths = await ensure_tools(silent=True)
        update_available_binaries()

        if not self._fd_path and paths.get("fd"):
            self._fd_path = paths["fd"]
            self.query_one("#input-box", InputBox).set_fd_path(self._fd_path)

        if not paths.get("eza") and sys.platform == "darwin":
            self._exit_hints.append(
                "Install eza for more token-efficient directory listings (brew install eza)"
            )

    async def _check_for_updates(self) -> None:
        latest = await get_newer_pypi_version(_PYPI_PACKAGE_NAME, VERSION)
        if latest is None:
            return

        self._pending_update_notice_version = latest
        self.call_later(self._show_pending_update_notice_if_idle)

    def _show_pending_update_notice_if_idle(self) -> None:
        if not self._startup_complete or self._is_running:
            return
        if self._update_notice_shown or self._pending_update_notice_version is None:
            return

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_update_available_message(
            self._pending_update_notice_version, changelog_url=_CHANGELOG_URL
        )
        self._update_notice_shown = True
        self._pending_update_notice_version = None

    def _add_launch_warning(
        self, message: str, *, severity: Literal["warning", "error"] = "warning"
    ) -> None:
        cleaned = message.strip()
        if not cleaned:
            return
        self._launch_warnings.append(LaunchWarning(message=cleaned, severity=severity))

    def _flush_launch_warnings(self, chat: ChatLog) -> None:
        if self._launch_warnings:
            chat.add_launch_warnings(self._launch_warnings)

    # -------------------------------------------------------------------------
    # Completion message handlers
    # -------------------------------------------------------------------------

    @on(InputBox.CompletionUpdate)
    def on_completion_update(self, event: InputBox.CompletionUpdate) -> None:
        if self._selection_mode is not None:
            return

        completion_list = self.query_one("#completion-list", FloatingList)
        if completion_list.is_visible:
            completion_list.update_items(event.items)
        else:
            completion_list.show(event.items)

    @on(InputBox.CompletionHide)
    def on_completion_hide(self, event: InputBox.CompletionHide) -> None:
        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.hide()

        input_box = self.query_one("#input-box", InputBox)
        if self._selection_mode is not None:
            self._selection_mode = None
            input_box.clear()
            input_box.set_autocomplete_enabled(True)
            self._reset_ctrl_d_delete_state()

        input_box.set_completing(False)

    @on(InputBox.CompletionSelect)
    def on_completion_select(self, event: InputBox.CompletionSelect) -> None:
        completion_list = self.query_one("#completion-list", FloatingList)
        input_box = self.query_one("#input-box", InputBox)
        item = completion_list.selected_item

        if not item:
            completion_list.hide()
            input_box.set_completing(False)
            input_box.submit_raw()
            return

        if self._selection_mode is not None:
            completion_list.hide()
            selection_mode = self._selection_mode
            self._selection_mode = None
            input_box.clear()
            input_box.set_autocomplete_enabled(True)
            input_box.set_completing(False)
            self._reset_ctrl_d_delete_state()

            match selection_mode:
                case SelectionMode.SESSION:
                    self.run_worker(self._load_session(item.value.path), exclusive=True)
                case SelectionMode.MODEL:
                    self._select_model(item.value)
                case SelectionMode.LOGIN:
                    self._select_login_provider(item.value)
                case SelectionMode.LOGOUT:
                    self._select_logout_provider(item.value)

            return

        if input_box.is_tab_completing:
            completion_list.hide()
            input_box.apply_tab_path_completion(item)
            return

        provider = input_box.active_provider
        completion_list.hide()

        if isinstance(provider, SlashCommandProvider):
            input_box.apply_slash_command(item)
        elif isinstance(provider, FilePathProvider):
            input_box.apply_file_completion(item)

        input_box.set_completing(False)

    @on(InputBox.SearchUpdate)
    def on_search_update(self, event: InputBox.SearchUpdate) -> None:
        if self._selection_mode is None:
            return
        completion_list = self.query_one("#completion-list", FloatingList)
        completion_list.set_search_query(event.query)

    @on(InputBox.CompletionMove)
    def on_completion_move(self, event: InputBox.CompletionMove) -> None:
        completion_list = self.query_one("#completion-list", FloatingList)
        if event.direction < 0:
            completion_list.move_up()
        else:
            completion_list.move_down()

    # -------------------------------------------------------------------------
    # Key bindings
    # -------------------------------------------------------------------------

    def action_handle_ctrl_c(self) -> None:
        input_box = self.query_one("#input-box", InputBox)
        status = self.query_one("#status-line", StatusLine)

        if input_box.text.strip():
            input_box.clear()
            status.hide_exit_hint()
            self._last_ctrl_c_time = 0.0
            return

        now = time.time()
        if now - self._last_ctrl_c_time < self._ctrl_c_threshold:
            self.exit()
        else:
            self._last_ctrl_c_time = now
            status.show_exit_hint()

            if self._ctrl_c_timer:
                self._ctrl_c_timer.stop()
            self._ctrl_c_timer = self.set_timer(
                self._ctrl_c_threshold, lambda: status.hide_exit_hint()
            )

    def action_handle_ctrl_d(self) -> None:
        if self._selection_mode != SelectionMode.SESSION:
            return

        completion_list = self.query_one("#completion-list", FloatingList)
        if not completion_list.is_visible or completion_list.selected_item is None:
            return

        status = self.query_one("#status-line", StatusLine)
        now = time.time()
        if now - self._last_ctrl_d_time < self._ctrl_d_threshold:
            self._last_ctrl_d_time = 0.0
            if self._ctrl_d_timer:
                self._ctrl_d_timer.stop()
                self._ctrl_d_timer = None
            status.hide_exit_hint()
            self._delete_selected_resume_session()
            return

        self._last_ctrl_d_time = now
        status.show_delete_session_hint()
        if self._ctrl_d_timer:
            self._ctrl_d_timer.stop()
        self._ctrl_d_timer = self.set_timer(
            self._ctrl_d_threshold, lambda: status.hide_exit_hint()
        )

    def action_interrupt_agent(self) -> None:
        if self._is_running:
            self._request_interrupt(show_message=True)

    def _request_interrupt(self, show_message: bool) -> None:
        if not self._is_running:
            return

        self._interrupt_requested = True

        if show_message:
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status("Interrupting...")

        if self._cancel_event:
            self._cancel_event.set()

    def _reset_ctrl_d_delete_state(self) -> None:
        self._last_ctrl_d_time = 0.0
        if self._ctrl_d_timer:
            self._ctrl_d_timer.stop()
            self._ctrl_d_timer = None

        status = self.query_one("#status-line", StatusLine)
        status.hide_exit_hint()

    def action_toggle_thinking(self) -> None:
        self._hide_thinking = not self._hide_thinking
        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)

        info_bar.set_thinking_visibility(self._hide_thinking)

        for block in chat.query(".thinking-block"):
            if self._hide_thinking:
                block.add_class("-hidden")
            else:
                block.remove_class("-hidden")

        status = "hidden" if self._hide_thinking else "visible"
        chat.show_status(f"Thinking blocks {status}")

    def action_cycle_thinking_level(self) -> None:
        if self._provider is None:
            return

        chat = self.query_one("#chat-log", ChatLog)
        info_bar = self.query_one("#info-bar", InfoBar)

        new_level = self._provider.cycle_thinking_level()
        self._thinking_level = new_level

        if self._session:
            self._session.set_thinking_level(new_level)

        info_bar.set_thinking_level(new_level)

        chat.show_status(f"Thinking level: {new_level}")

    @on(HandoffLinkBlock.LinkSelected)
    def on_handoff_link_selected(self, event: HandoffLinkBlock.LinkSelected) -> None:
        if not event.target_session_id:
            return
        event.stop()
        self.run_worker(self._load_session_by_id(event.target_session_id), exclusive=True)

    def _clear_approval_state(self) -> None:
        self._approval_future = None
        if self._approval_tool_id is not None:
            chat = self.query_one("#chat-log", ChatLog)
            chat.hide_tool_approval(self._approval_tool_id)
            self._approval_tool_id = None

    def deny_pending_approval(self) -> bool:
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(ApprovalResponse.DENY)
            self._clear_approval_state()
            return True
        return False

    def on_key(self, event: events.Key) -> None:
        if self._approval_future is None or self._approval_future.done():
            return
        if event.key in ("y", "Y"):
            self._approval_future.set_result(ApprovalResponse.APPROVE)
        elif event.key in ("n", "N"):
            self._approval_future.set_result(ApprovalResponse.DENY)
        else:
            return
        event.prevent_default()
        event.stop()
        self._clear_approval_state()

    @on(InputBox.Submitted)
    def on_input_submitted(self, event: InputBox.Submitted) -> None:
        display_text = event.text.strip()
        if not display_text:
            return

        if display_text.startswith("/") and self._handle_command(display_text):
            return

        query_text = event.query_text.strip()

        skill_prompt: str | None = None
        selected_skill_name = event.selected_skill_name
        highlighted_skill: str | None = None
        if selected_skill_name and self._agent:
            selected_skill = next(
                (
                    skill
                    for skill in self._agent.context.skills
                    if skill.register_cmd and skill.name == selected_skill_name
                ),
                None,
            )
            if selected_skill:
                skill_query = event.selected_skill_query or ""
                skill_prompt = self._build_skill_trigger_message(
                    selected_skill.name, selected_skill.description, skill_query
                )
                display_text = skill_prompt
                query_text = skill_prompt
                highlighted_skill = selected_skill.name

        if self._is_running:
            if len(self._pending_queue) >= QueueDisplay.MAX_QUEUE:
                self.notify("Queue full (max 5)", severity="warning", timeout=2)
                return
            self._pending_queue.append((display_text, query_text))
            self._update_queue_display()
            return

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_user_message(display_text, highlighted_skill=highlighted_skill)

        self._is_running = True
        self.run_worker(self._run_agent(query_text), exclusive=True)

    def _update_queue_display(self) -> None:
        queue_display = self.query_one("#queue-display", QueueDisplay)
        queue_display.update_items([display for display, _ in self._pending_queue])

    async def _run_agent(self, prompt: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-line", StatusLine)
        info_bar = self.query_one("#info-bar", InfoBar)

        if self._provider is None or self._session is None:
            chat.add_info_message("Agent not initialized")
            self._is_running = False
            return

        if self._agent is None:
            self._agent = Agent(
                provider=self._provider,
                tools=get_tools(DEFAULT_TOOLS),
                session=self._session,
                cwd=self._cwd,
            )

        current_prompt = prompt

        while True:
            was_interrupted = False

            self._cancel_event = asyncio.Event()
            self._abort_shown = False
            self._current_block_type = None
            if self._interrupt_requested:
                self._cancel_event.set()

            model_info = get_model(self._model, self._model_provider)
            self._agent.provider = self._provider
            self._agent.session = self._session
            self._agent.tools = get_tools(DEFAULT_TOOLS)
            self._agent.config.context_window = model_info.context_window if model_info else None
            self._agent.config.max_output_tokens = model_info.max_tokens if model_info else None

            status.set_status("working")

            try:
                async for event in self._agent.run(
                    current_prompt, cancel_event=self._cancel_event
                ):
                    match event:
                        case AgentStartEvent():
                            pass

                        case TurnStartEvent():
                            pass

                        case ThinkingStartEvent():
                            if self._current_block_type != "thinking":
                                if self._current_block_type:
                                    chat.end_block()
                                block = chat.start_thinking()
                                if self._hide_thinking:
                                    block.add_class("-hidden")
                                self._current_block_type = "thinking"

                        case ThinkingDeltaEvent(delta=d):
                            await chat.append_to_current(d)

                        case ThinkingEndEvent():
                            pass

                        case TextStartEvent():
                            if self._current_block_type != "content":
                                if self._current_block_type:
                                    chat.end_block()
                                chat.start_content()
                                self._current_block_type = "content"

                        case TextDeltaEvent(delta=d):
                            await chat.append_to_current(d)

                        case TextEndEvent():
                            pass

                        case ToolStartEvent(tool_call_id=id, tool_name=name):
                            if self._current_block_type:
                                chat.end_block()
                            chat.start_tool(name, id, "")
                            self._current_block_type = "tool_call"
                            status.increment_tool_calls()
                            status.set_streaming_tokens(0)  # Reset token count for new tool

                        case ToolArgsTokenUpdateEvent(token_count=tc):
                            status.set_streaming_tokens(tc)

                        case ToolEndEvent(tool_call_id=id, display=display):
                            chat.update_tool_call_msg(id, display)

                        case ToolApprovalEvent(tool_call_id=id, tool_name=name, future=f):
                            chat.show_tool_approval(id)
                            self._approval_future = f
                            self._approval_tool_id = id

                        case ToolResultEvent(tool_call_id=id, result=r, file_changes=fc):
                            self._approval_future = None
                            self._approval_tool_id = None
                            if r:
                                if r.display:
                                    text = r.display
                                    markup = True
                                elif r.content:
                                    text = self._format_tool_result_text(r)
                                    markup = False
                                else:
                                    text = ""
                                    markup = False
                                success = not r.is_error
                                chat.set_tool_result(id, text, success, markup=markup)
                            if fc:
                                info_bar.update_file_changes(fc.path, fc.added, fc.removed)

                        case TurnEndEvent():
                            if event.assistant_message and event.assistant_message.usage:
                                usage = event.assistant_message.usage
                                info_bar.update_tokens(
                                    usage.input_tokens,
                                    usage.output_tokens,
                                    usage.cache_read_tokens,
                                    usage.cache_write_tokens,
                                )

                        case InterruptedEvent():
                            was_interrupted = True
                            if self._current_block_type:
                                chat.end_block()
                                self._current_block_type = None

                        case CompactionStartEvent():
                            if self._current_block_type:
                                chat.end_block()
                                self._current_block_type = None
                            chat.show_spinner_status("Auto-compacting...")

                        case CompactionEndEvent(tokens_before=tb, aborted=ab):
                            if ab:
                                chat.show_status("Compaction failed")
                            else:
                                chat.add_compaction_message(tb)

                        case RetryEvent(attempt=a, total_attempts=t, delay=d, error=e):
                            msg = f"Request failed (attempt {a}/{t}), retrying in {d}s; Error: {e}"
                            chat.add_info_message(msg, error=True)

                        case ErrorEvent(error=e):
                            chat.add_info_message(str(e), error=True)

                        case WarningEvent(warning=w):
                            chat.add_info_message(str(w), warning=True)

                        case AgentEndEvent(stop_reason=reason):
                            if reason == StopReason.INTERRUPTED:
                                was_interrupted = True
                            if self._current_block_type:
                                chat.end_block()
                            self._current_block_type = None

            except Exception as e:
                chat.add_info_message(str(e), error=True)

            if was_interrupted and not self._abort_shown:
                chat.add_aborted_message("Interrupted by user")
                self._abort_shown = True

            self._interrupt_requested = False
            self._cancel_event = None
            self._clear_approval_state()
            status.set_status("idle")

            if was_interrupted:
                self._pending_queue.clear()
                self._update_queue_display()
                break

            if self._pending_queue:
                next_display, next_query = self._pending_queue.popleft()
                self._update_queue_display()
                chat.add_user_message(next_display)
                current_prompt = next_query
                continue

            break

        self._is_running = False
        self._show_pending_update_notice_if_idle()


_LOGO = ["█ K █", "█ O █", "█ N █"]


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    secs = total % 60
    return f"{minutes}m {secs}s"


def _print_exit_message(
    hints: list[str],
    session_id: str | None = None,
    duration_seconds: float | None = None,
    file_changes: dict[str, tuple[int, int]] | None = None,
) -> None:
    colors = config.ui.colors
    console = Console(highlight=False)

    for hint in hints:
        console.print(f"Hint: [bright_black]{hint}[/bright_black]")

    t = colors.dim
    logo_color = "bright_black"
    info_lines: list[str] = []

    if duration_seconds is not None:
        info_lines.append(f"[{t}]Elapsed {_format_duration(duration_seconds)}[/{t}]")

    if file_changes:
        n_files = len(file_changes)
        total_added = sum(a for a, _ in file_changes.values())
        total_removed = sum(r for _, r in file_changes.values())
        info_lines.append(
            f"[{t}]Changed {n_files} file{'s' if n_files != 1 else ''}[/{t}]"
            f" [{colors.diff_added}]+{total_added}[/{colors.diff_added}]"
            f" [{colors.diff_removed}]-{total_removed}[/{colors.diff_removed}]"
        )

    if session_id:
        info_lines.append(f"[{t}]kon -r {session_id}[/{t}]")

    if not info_lines:
        return

    while len(info_lines) < len(_LOGO):
        info_lines.append("")

    console.print()
    for logo_line, info_line in zip(_LOGO, info_lines, strict=False):
        padding = "  " if info_line else ""
        console.print(f"  [{logo_color}]{logo_line}[/{logo_color}]{padding}{info_line}")
    console.print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kon TUI")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument(
        "--provider", "-p", choices=sorted(PROVIDER_API_BY_NAME), help="Provider to use"
    )
    parser.add_argument("--api-key", "-k", help="API key")
    parser.add_argument("--base-url", "-u", help="Base URL for API")
    parser.add_argument(
        "--continue",
        "-c",
        action="store_true",
        dest="continue_recent",
        help="Resume the most recent session",
    )
    parser.add_argument(
        "--resume",
        "-r",
        dest="resume_session",
        help="Resume a specific session by ID (full or unique prefix)",
    )
    parser.add_argument("--version", action="version", version=f"kon {VERSION}")
    args = parser.parse_args()

    app = Kon(
        model=args.model,
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        resume_session=args.resume_session,
        continue_recent=args.continue_recent,
    )
    app.run()

    hints = list(app._exit_hints)
    session_id: str | None = None
    duration: float | None = None
    file_changes: dict[str, tuple[int, int]] | None = None

    if app._session:
        session_id = app._session.id
        file_changes = SessionUIMixin._calculate_session_file_changes(app._session) or None
    if app._session_start_time is not None:
        duration = time.time() - app._session_start_time

    if hints or session_id:
        _print_exit_message(hints, session_id, duration, file_changes)


if __name__ == "__main__":
    main()
