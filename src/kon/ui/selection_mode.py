from enum import StrEnum


class SelectionMode(StrEnum):
    SESSION = "session"
    MODEL = "model"
    THEME = "theme"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSIONS = "permissions"
    THINKING = "thinking"
    THINKING_LINES = "thinking_lines"
    COLORED_TOOL_BADGE = "colored_tool_badge"
    NOTIFICATIONS = "notifications"
    SETTINGS = "settings"
    TREE = "tree"
    URL = "url"
