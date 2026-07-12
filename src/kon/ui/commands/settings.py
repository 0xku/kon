"""/settings, /themes, /permissions, /thinking and /notifications commands."""

from __future__ import annotations

from typing import Literal

from kon import (
    config,
    set_colored_tool_badge,
    set_git_context,
    set_notifications_enabled,
    set_permissions_mode,
    set_show_welcome_shortcuts,
    set_theme,
    set_thinking_lines,
)
from kon.config import (
    NOTIFICATION_MODES,
    PERMISSION_MODES,
    THINKING_LINES_OPTIONS,
    NotificationMode,
    PermissionMode,
    ThinkingLinesOption,
)

from ...themes import get_theme_options
from ..chat import ChatLog
from ..floating_list import FloatingList, ListItem
from ..selection_mode import SelectionMode
from ..widgets import InfoBar
from .base import CommandSupport

SettingsSelectionResult = Literal["reopened-picker", "closed"]


class SettingsCommands(CommandSupport):
    def _handle_themes_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)

        requested = args.strip()
        if requested:
            try:
                self._select_theme(requested)
            except ValueError as e:
                chat.add_info_message(str(e), error=True)
            return

        current_theme = config.ui.theme
        items = [
            ListItem(value=theme_id, label=f"{label} ✓" if theme_id == current_theme else label)
            for theme_id, label in get_theme_options()
        ]

        self._show_selection_picker(items, SelectionMode.THEME)

    def _select_theme(self, theme_id: str) -> None:
        set_theme(theme_id)
        self._apply_theme(theme_id)
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info_message(
            f"Theme changed to {theme_id}. Full theme refresh applies when kon is restarted.",
            warning=True,
        )

    def _handle_permissions_command(self, args: str) -> None:
        descriptions: dict[PermissionMode, str] = {
            "prompt": "ask before mutating tool calls",
            "auto": "allow tool calls without approval prompts",
        }
        self._handle_choice_command(
            args,
            name="permission",
            choices=PERMISSION_MODES,
            current=config.permissions.mode,
            descriptions=descriptions,
            selection_mode=SelectionMode.PERMISSIONS,
            select=self._select_permission_mode,
        )

    def _select_permission_mode(self, mode: PermissionMode) -> None:
        set_permissions_mode(mode)
        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.set_permission_mode(mode)
        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Permission mode changed to {mode}")

    def _handle_thinking_command(self, args: str) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        if self._runtime.provider is None:
            chat.add_info_message("Agent not initialized", error=True)
            return

        requested = args.strip()
        if requested:
            if requested in self._runtime.provider.thinking_levels:
                self._select_thinking_level(requested)
            else:
                valid_levels = ", ".join(self._runtime.provider.thinking_levels)
                chat.add_info_message(
                    f"Invalid thinking level: {requested}. Use one of: {valid_levels}", error=True
                )
            return

        items = [
            ListItem(
                value=level, label=f"{level} ✓" if level == self._runtime.thinking_level else level
            )
            for level in self._runtime.provider.thinking_levels
        ]
        self._show_selection_picker(items, SelectionMode.THINKING)

    def _select_thinking_level(self, level: str) -> None:
        if self._runtime.provider is None:
            return

        self._runtime.set_thinking_level(level)
        self._sync_runtime_state()

        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.set_thinking_level(level)

        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Thinking level changed to {level}")

    def _show_thinking_lines_picker(self) -> None:
        descriptions: dict[ThinkingLinesOption, str] = {
            "1": "show 1 line",
            "2": "show 2 lines",
            "3": "show 3 lines",
            "4": "show 4 lines",
            "5": "show 5 lines",
            "none": "no truncation",
        }
        items = self._build_choice_items(
            THINKING_LINES_OPTIONS, config.ui.thinking_lines, descriptions
        )
        self._show_selection_picker(items, SelectionMode.THINKING_LINES)

    def _select_thinking_lines(self, lines: ThinkingLinesOption) -> None:
        set_thinking_lines(lines)
        chat = self.query_one("#chat-log", ChatLog)
        label = (
            "no truncation" if lines == "none" else f"{lines} line{'s' if lines != '1' else ''}"
        )
        chat.show_status(f"Thinking lines changed to {label}")

    def _handle_notifications_command(self, args: str) -> None:
        current: NotificationMode = "on" if config.notifications.enabled else "off"
        descriptions: dict[NotificationMode, str] = {
            "on": "play notification sounds",
            "off": "disable notification sounds",
        }
        self._handle_choice_command(
            args,
            name="notifications",
            choices=NOTIFICATION_MODES,
            current=current,
            descriptions=descriptions,
            selection_mode=SelectionMode.NOTIFICATIONS,
            select=self._select_notifications_mode,
        )

    def _select_notifications_mode(self, mode: NotificationMode) -> None:
        set_notifications_enabled(mode == "on")
        chat = self.query_one("#chat-log", ChatLog)
        chat.show_status(f"Notifications turned {mode}")

    # -------------------------------------------------------------------------
    # Settings (unified panel for themes, permissions, notifications, thinking)
    # -------------------------------------------------------------------------

    def _build_settings_items(self) -> list[ListItem[str]]:
        notification_status = "on" if config.notifications.enabled else "off"
        try:
            thinking_level = self._runtime.thinking_level or "off"
        except Exception:
            thinking_level = "off"

        shortcut_status = "on" if config.ui.show_welcome_shortcuts else "off"
        thinking_lines_status = config.ui.thinking_lines
        colored_badge_status = "on" if config.ui.colored_tool_badge else "off"
        git_context_status = "on" if config.llm.system_prompt.git_context else "off"
        return [
            ListItem(
                value="colored-tool-badge",
                label="colored-tool-badge",
                description=colored_badge_status,
            ),
            ListItem(value="git-context", label="git-context", description=git_context_status),
            ListItem(
                value="notifications", label="notifications", description=notification_status
            ),
            ListItem(value="show-shortcuts", label="show-shortcuts", description=shortcut_status),
            ListItem(
                value="permissions", label="permissions", description=config.permissions.mode
            ),
            ListItem(value="themes", label="themes", description=config.ui.theme),
            ListItem(value="thinking", label="thinking", description=thinking_level),
            ListItem(
                value="thinking-lines", label="thinking-lines", description=thinking_lines_status
            ),
        ]

    def _show_settings_picker(self, selected_value: str | None = None) -> None:
        items = self._build_settings_items()
        self._show_selection_picker(items, SelectionMode.SETTINGS, max_label_width=40)
        self._settings_selected_value = selected_value
        if selected_value is not None:
            completion_list = self.query_one("#completion-list", FloatingList)
            completion_list.select_value(selected_value)

    def _handle_settings_command(self) -> None:
        self._show_settings_picker()

    def _handle_settings_select(self, item_value: str) -> SettingsSelectionResult:
        if item_value == "notifications":
            current_enabled = config.notifications.enabled
            set_notifications_enabled(not current_enabled)
            mode: NotificationMode = "on" if not current_enabled else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Notifications turned {mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "show-shortcuts":
            shortcuts_current = config.ui.show_welcome_shortcuts
            set_show_welcome_shortcuts(not shortcuts_current)
            mode = "on" if not shortcuts_current else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Welcome shortcuts turned {mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "permissions":
            current: PermissionMode = config.permissions.mode
            new_mode: PermissionMode = "auto" if current == "prompt" else "prompt"
            set_permissions_mode(new_mode)
            info_bar = self.query_one("#info-bar", InfoBar)
            info_bar.set_permission_mode(new_mode)
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Permission mode changed to {new_mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "themes":
            self._settings_active = True
            self._handle_themes_command("")
            return "reopened-picker"

        elif item_value == "thinking":
            if self._runtime.provider is None:
                self._handle_thinking_command("")
                return "closed"
            self._settings_active = True
            self._handle_thinking_command("")
            return "reopened-picker"

        elif item_value == "thinking-lines":
            self._settings_active = True
            self._show_thinking_lines_picker()
            return "reopened-picker"

        elif item_value == "colored-tool-badge":
            badge_current = config.ui.colored_tool_badge
            set_colored_tool_badge(not badge_current)
            mode = "on" if not badge_current else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Colored tool badge turned {mode}")
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        elif item_value == "git-context":
            git_current = config.llm.system_prompt.git_context
            set_git_context(not git_current)
            mode = "on" if not git_current else "off"
            chat = self.query_one("#chat-log", ChatLog)
            chat.show_status(f"Git context turned {mode}")
            chat.add_info_message(
                "Git context change applies on new conversations (use /new) or on kon restart.",
                warning=True,
            )
            self._show_settings_picker(selected_value=item_value)
            return "reopened-picker"

        return "closed"
