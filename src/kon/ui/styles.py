"""TUI styles for kon."""

from kon import config


def get_styles() -> str:
    colors = config.ui.colors

    return f"""
Screen {{
    layout: grid;
    grid-size: 1;
    grid-rows: 1fr auto auto auto auto auto;
    background: {colors.bg};
    color: {colors.fg};
}}

#chat-log {{
    height: 100%;
    padding: 0 0 1 0;
    scrollbar-size: 0 0;
    align-vertical: bottom;
    background: {colors.bg};
    color: {colors.fg};
}}

/* Thinking block - dim italic with left border */
.thinking-block {{
    color: {colors.dim};
    text-style: italic;
    padding: 0 1 0 1;
    margin: 1 0 0 1;
    border-left: thick {colors.border};
}}

.thinking-block.-hidden {{
    display: none;
    height: 0;
    margin: 0;
    border: none;
}}

#thinking-content {{
    color: {colors.dim};
    text-style: italic;
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
    border-top: solid {colors.border};
    border-bottom: solid {colors.border};
    background: {colors.panel_user};
}}

.user-block.skill-trigger-message {{
    background: {colors.panel_user};
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
    padding: 0 1;
    margin-top: 1;
    background: transparent;
}}

.tool-block.-compact {{
    margin-top: 0;
}}

.tool-block.-pending,
.tool-block.-approval,
.tool-block.-success,
.tool-block.-error {{
    background: transparent;
    color: {colors.dim};
    border: none;
}}

#tool-header {{
    color: {colors.dim};
    text-style: none;
}}

#tool-output,
.tool-output {{
    color: {colors.dim};
    padding: 0 0 0 2;
}}

.tool-block.-with-details {{
    padding: 0 1;
}}

#tool-output.-hidden {{
    display: none;
    height: 0;
}}

/* Compaction message */
.compaction-message {{
    background: {colors.panel};
    padding: 1 1;
    margin-top: 1;
    width: 100%;
}}

/* Handoff link */
.handoff-link-block {{
    background: {colors.panel};
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

/* Input area */
#input-box {{
    border-top: solid {colors.border};
    border-bottom: solid {colors.border};
    border-title-color: {colors.dim};
    border-subtitle-color: {colors.dim};
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
