# Kon

[![PyPI](https://img.shields.io/pypi/v/kon-coding-agent)](https://pypi.org/project/kon-coding-agent/)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Kon is a minimal coding agent with a tiny harness: about **210 tokens** for the system prompt and around **1,070 tokens** for tool definitions – so **under 1.3k tokens** before conversation context. This does not include AGENTS.md, skills, or git context (if enabled), which vary per user and project.

The repo has **~126 files** and is easy to understand in a weekend. If you want a truly minimal coding agent with batteries included – something you can understand, fork, and extend quickly – Kon might be interesting.

[Kon](https://bleach.fandom.com/wiki/Kon) is inspired from Bleach, an artificial soul

## Setup

### Prerequisites

Python 3.12+ and [uv](https://github.com/astral-sh/uv).

### Install (recommended)

```bash
uv tool install kon-coding-agent
```

This installs `kon` globally as a CLI tool.

### Install from source (advanced)

```bash
git clone https://github.com/kuutsav/kon
cd kon
uv tool install .
```

> [!WARNING]
> Platform support: macOS and Linux are supported; Windows is not tested yet.

### Run

```bash
kon
```

CLI options:

```text
usage: kon [-h] [--model MODEL]
           [--provider {azure-ai-foundry,github-copilot,openai,openai-codex,openai-responses,zhipu}]
           [--api-key API_KEY] [--base-url BASE_URL] [--continue]
           [--resume RESUME_SESSION]

Kon TUI

options:
  -h, --help            show this help message and exit
  --model, -m MODEL     Model to use
  --provider, -p {azure-ai-foundry,github-copilot,openai,openai-codex,openai-responses,zhipu}
                        Provider to use
  --api-key, -k API_KEY
                        API key
  --base-url, -u BASE_URL
                        Base URL for API
  --continue, -c        Resume the most recent session
  --resume, -r RESUME_SESSION
                        Resume a specific session by ID (full or unique
                        prefix)
```

### Tool binaries

- **[fd](https://github.com/sharkdp/fd)** – required for fast file discovery; Kon auto-downloads it only if it's missing.
- **[ripgrep (rg)](https://github.com/BurntSushi/ripgrep)** – required for fast content search; Kon auto-downloads it only if it's missing.
- **[eza](https://github.com/eza-community/eza)** (optional) – supports `.gitignore`-aware listings and usually emits fewer tokens than `ls`.

## OAuth and API keys

- **GitHub Copilot OAuth**: run `/login` and choose GitHub Copilot.
- **OpenAI OAuth (Codex)**: run `/login` and choose OpenAI. Kon supports callback flow plus manual paste fallback.
- **OpenAI-compatible providers (for example ZhiPu)**: set an API key via environment variable (`OPENAI_API_KEY` or `ZAI_API_KEY`).
- **Azure AI Foundry**: set `AZURE_AI_FOUNDRY_API_KEY` and `AZURE_AI_FOUNDRY_BASE_URL` environment variables.

## Features

### Tools

| Tool   | Purpose |
| ------ | ------- |
| `read` | Read file contents (pagination for large files, image support) |
| `edit` | Surgical find-and-replace edits |
| `write` | Create or overwrite files |
| `bash` | Execute shell commands |
| `grep` | Search file contents with regex |
| `find` | Find files by glob pattern |

### Slash commands

Type `/` at the start of input to see available commands.

| Command | Description |
| ------- | ----------- |
| `/new` | Start a new conversation and reload project context/skills |
| `/resume` | Browse and restore a saved session |
| `/model` | Switch model via interactive picker |
| `/session` | Show session metadata and token stats |
| `/compact` | Compact the current conversation immediately |
| `/export` | Export current session to HTML |
| `/copy` | Copy last assistant response to clipboard |
| `/login` | Authenticate with a provider |
| `/logout` | Log out from a provider |
| `/clear` | Clear current conversation |
| `/help` | Show commands and keybindings |
| `/quit` (`/exit`, `/q`) | Quit Kon |

### `@` file and folder search

Type `@` + query to fuzzy-search files/folders in the current project and insert paths into your prompt.

### Tab path autocomplete

Press **Tab** in the input box to complete paths (`~`, `./`, `../`, absolute paths, quoted paths, etc.).

### Query queueing

If the agent is currently running, you can still submit more prompts. Kon queues them and runs them in order once the current task finishes (up to 5 queued prompts).

### Sessions

Sessions are append-only JSONL files under `~/.kon/sessions/`.

- `/resume` to reopen past sessions
- `/session` for message/token stats
- `/export` for standalone HTML transcripts
- `--continue` / `-c` to continue the most recent session from CLI

### AGENTS.md

Kon loads project guidelines from `AGENTS.md` (or `CLAUDE.md`) files into the system prompt:

1. Global: `~/.kon/AGENTS.md`
2. Ancestor directories from git root (or home) down to current working directory

### Skills

Skills are reusable instruction packs loaded from:

- Project: `.kon/skills/`
- Global: `~/.kon/skills/`

Each skill has a `SKILL.md` file with front matter:

```markdown
---
name: my-skill
description: Brief description of what this skill does
register_cmd: true
cmd_info: Quick action shown in slash menu
---

# My Skill

Detailed instructions for the agent...
```

Front matter fields:

- `name` (optional) – Skill identifier. If omitted, Kon uses the directory name.
- `description` (required) – Used for skill discovery and shown in the prompt context.
- `register_cmd` (optional, default `false`) – If `true`, the skill is available as a slash command (`/my-skill`) in the input menu.
- `cmd_info` (optional) – Short menu hint shown for slash-command skills.

Validation rules:

- `name` must be lowercase `a-z`, `0-9`, and `-`
- `name` must not start/end with `-` or include `--`
- `name` max length: 64 chars
- `description` max length: 1024 chars
- `cmd_info` max length: 32 chars

For skills with scripts, see [Agent Skills Documentation](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).

## Not supported

Some features you might expect in other coding agents are not part of Kon's design philosophy:

- **MCP servers** – Use skills instead; they're simpler and give you full control
- **Sandbox environments** – Kon runs directly on your machine for simplicity; use Docker or VMs if you need isolation
- **Checkpoint restores** – Not currently supported; may be added in the future

## Architecture

```text
LLM Provider
    │
    │ StreamPart (TextPart, ThinkPart, ToolCallStart, ToolCallDelta, ...)
    ▼
Single Turn (turn.py)
    │
    │ StreamEvent (ThinkingStart/Delta/End, TextStart/Delta/End, ToolStart/End, ToolResult, ...)
    ▼
Agentic Loop (loop.py)
    │
    │ Event (AgentStart, TurnStart, TurnEnd, AgentEnd + all StreamEvents)
    ▼
UI (app.py)
```

## Supported Models

Kon works well with local models exposed through an OpenAI-compatible `/v1` API.

### Example using llama-server

To run a local model using llama-server:

```bash
./llama-server -m <models-dir>/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf \
    -n 8192 \
    -c 64000

# Then use Kon with:
kon --model zai-org/glm-4.7-flash \
    --provider openai \
    --base-url http://localhost:8080/v1 \
    --api-key ""
```

`GLM-4.7-Flash-Q4` ran at 80-90 tps on my i7-14700F × 28, 64GB RAM, 24GB VRAM (RTX 3090)

### All Supported Providers

| Model (local=*) | Provider | Thinking | Vision |
| ----- | -------- | -------- | ------ |
| `*zai-org/glm-4.7-flash` | OpenAI Completions | Yes | No |
| `*qwen/qwen3-coder-next` | OpenAI Completions | Yes | No |
| `glm-4.7` | ZhiPu (OpenAI Completions) | Yes | No |
| `glm-5` | ZhiPu (OpenAI Completions) | Yes | No |
| `claude-sonnet-4.6` | GitHub Copilot | Yes | Yes |
| `claude-opus-4.6` | GitHub Copilot | Yes | Yes |
| `gpt-5.3-codex` | GitHub Copilot | Yes | Yes |
| `gpt-5.4` | GitHub Copilot | Yes | Yes |
| `gpt-5.3-codex` | OpenAI Codex Responses | Yes | Yes |
| `gpt-5.4` | OpenAI Codex Responses | Yes | Yes |
| `claude-sonnet-4.6` | Azure AI Foundry | Yes | Yes |
| `claude-opus-4.6` | Azure AI Foundry | Yes | Yes |


## Configuration

Config lives at `~/.kon/config.toml` (auto-created on first run).

Kon also runs automatic config migrations on startup for older schemas. When a migration is applied,
the previous file is backed up to `~/.kon/config.toml.bak.<timestamp>`.

Here’s the full default config with all available options:

```toml
[meta]
config_version = 2

[llm]
default_provider = "openai-codex" # "zhipu", "github-copilot", "openai-codex", "azure-ai-foundry"
default_model = "gpt-5.3-codex"
default_base_url = ""
default_thinking_level = "high"
tool_call_idle_timeout_seconds = 120

[llm.system_prompt]
git_context = false # if true, injects git branch, recent commits, and status into the system prompt
content = """You are an expert coding assistant called `Kon`.
...
"""

[compaction]
on_overflow = "continue" # "continue" or "pause"
buffer_tokens = 20000

[agent]
max_turns = 500
default_context_window = 200000

[permissions]
mode = "prompt" # "prompt" requires approval for mutating tools, "auto" skips approval

[ui.colors]
dim = "#666666"
muted = "#808080"
title = "#d97706"
spinner = "#d97706"
accent = "#8ABEB7"
info = "#ffff00"
markdown_heading = "#f0c674"
markdown_code = "#689d6a"
selected = "#689d6a"
error = "#e06c75"
notice = "#e5c07b"
diff_added = "#b5bd68"
diff_removed = "#cc6666"

[ui.colors.tool_bg]
pending = "#282832"
success = "#283228"
error = "#3c2828"

[ui.colors.badge]
bg = "#2d2838"
label = "#9575cd"
```

## Development setup

For hacking on Kon locally:

```bash
uv sync
uv run kon
uv run ruff format .
uv run pytest
```

## Acknowledgements

- Kon takes significant inspiration from [`pi-mono` coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent), especially in terms of the overall philosophy and UI design.
  - Why not just use pi? Pi is no longer a small project, and I want to be in complete control of my coding agent.
  - I mostly agree with Mario (author of pi), but I have different beliefs on some matters - for example, subagents (especially useful for context gathering in larger repos when paired with semantic search tools).
  - Over time, I also want to give more preference to local LLMs I can run. `glm-4.7-flash` and `qwen-3-coder-next` look promising, so I may make decisions that do not necessarily optimize for SOTA paid models.
- Kon also borrows ideas from [Amp](https://ampcode.com/), Claude Code, and other coding agents.

## LICENSE

MIT
