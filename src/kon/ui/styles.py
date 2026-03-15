"""TUI styles for kon."""

from kon import config


def get_styles() -> str:
    colors = config.ui.colors

    return f"""
Screen {{
    layout: grid;
    grid-size: 1;
    grid-rows: 1fr auto auto auto auto auto;
}}

#chat-log {{
    height: 100%;
    padding: 0 0 1 0;
    scrollbar-size: 0 0;
    align-vertical: bottom;
}}

/* Thinking block - dim, hidden by default */
.thinking-block {{
    color: {colors.dim};
    padding: 0 1;
    margin-top: 1;
}}

.thinking-block.-hidden {{
    display: none;
    height: 0;
    margin: 0;
}}

#thinking-content {{
    color: {colors.dim};
}}

/* Content block */
.content-block {{
    padding: 0 1;
    margin-top: 1;
}}

/* Ensure text wraps in all blocks */
.thinking-block Label,
.content-block Label,
.user-block Label,
.update-available-block Label,
.launch-warnings-block Label,
.tool-block Label,
.handoff-link-block Label {{
    width: 100%;
}}

/* User message */
.user-block {{
    padding: 0 1;
    margin: 1 0 0 0;
    border-top: solid grey;
    border-bottom: solid grey;
}}

.user-block.skill-trigger-message {{
    background: {colors.badge.bg};
}}

/* Update available message */
.update-available-block {{
    padding: 0 1;
    margin: 1 0 0 0;
    border-top: solid yellow;
    border-bottom: solid yellow;
}}

/* Launch warnings */
.launch-warnings-block {{
    padding: 0 1;
    margin: 1 0 0 0;
    border-top: solid yellow;
    border-bottom: solid yellow;
}}

/* Session info */
.session-info {{
    padding: 1;
}}

/* Tool block */
.tool-block {{
    padding: 1 1;
    margin-top: 1;
}}

.tool-block.-pending {{
    background: {colors.tool_bg.pending};
    color: #c0c0d0;
}}

.tool-block.-approval {{
    background: {colors.tool_bg.pending};
    color: #c0c0d0;
    border-left: thick {colors.accent};
}}

.tool-block.-success {{
    background: {colors.tool_bg.success};
    color: #c0d0c0;
}}

.tool-block.-error {{
    background: {colors.tool_bg.error};
    color: #d0c0c0;
}}

#tool-header {{
    /* Color handled by Rich Text inline styling for the symbol */
}}

/* Compaction message */
.compaction-message {{
    background: {colors.badge.bg};
    padding: 1 1;
    margin-top: 1;
    width: 100%;
}}

/* Handoff link */
.handoff-link-block {{
    background: {colors.badge.bg};
    padding: 1 1;
    margin: 1 0 0 0;
    width: 100%;
}}

/* Aborted message */
.aborted-message {{
    padding: 0 1;
    margin-top: 1;
}}

/* Info message */
.info-message {{
    padding: 0 1;
    margin-top: 1;
}}

/* Loaded resources should not add extra top margin */
.info-message.loaded-resources {{
    margin-top: 0;
}}

/* First info message after session info - no margin */
.info-message:first-of-type {{
    margin-top: 0;
}}

/* Queue display - shown above status line when messages are queued */
#queue-display {{
    height: auto;
    padding: 0 1 1 1;
}}

#queue-display.-hidden {{
    display: none;
}}

#queue-content {{
    color: {colors.dim};
    width: 100%;
}}

/* Status line - kon style with spinner */
.status-line {{
    height: auto;
    min-height: 1;
    padding: 0 1;
    color: $warning;
}}

#status-text {{
    color: {colors.dim};
    width: 1fr;
}}

#exit-hint {{
    color: {colors.dim};
    width: auto;
}}

/* Completion list - between input and info bar */
#completion-list {{
    height: auto;
    padding: 0 1;
}}

/* Info bar - kon style tmux-like bottom bar with two rows */
.info-bar {{
    height: 2;
    color: {colors.dim};
}}

#info-row-1, #info-row-2 {{
    height: 1;
}}

#info-cwd {{
    width: 1fr;
    padding: 0 1;
}}

#info-row2-left {{
    width: 1fr;
    padding: 0 1;
}}

#info-row1-right, #info-row2-right {{
    width: auto;
    padding: 0 1;
    text-align: right;
}}

/* Notifications */
Notification {{
    layer: notification;
}}
"""


STYLES = get_styles()
