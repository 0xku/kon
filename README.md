<h1 align="center">Kon</h1>
<p align="center">A minimal terminal coding agent with a small core, strong defaults, and user-owned context.</p>
<p align="center">
  <a href="https://pypi.org/project/kon-coding-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/kon-coding-agent?style=flat-square" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square" /></a>
</p>

<p align="center">
  <img src="docs/images/kon-screenshot.png" alt="Kon terminal UI screenshot" width="490" />
</p>

Kon is a minimal coding agent focused on a tiny core prompt, a small built-in toolset, and project-specific context layered on top only when you want it. The default system prompt stays **under 270 tokens**, and even including the built-in tool descriptions and parameter schemas, the fixed harness stays at about **~1,000 tokens**. The core experience is built around just **6 default tools** plus **2 optional web tools**.

[Kon](https://bleach.fandom.com/wiki/Kon) is named after the artificial soul from *Bleach*.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Why Kon](#why-kon)
  - [Minimal by design](#minimal-by-design)
  - [Configuration](#configuration)
  - [Core tools](#core-tools)
  - [Extra tools](#extra-tools)
- [Interactive TUI](#interactive-tui)
  - [Editor and navigation](#editor-and-navigation)
  - [Slash commands](#slash-commands)
  - [Themes](#themes)
- [Sessions](#sessions)
  - [Resume and continue](#resume-and-continue)
  - [Handoff](#handoff)
  - [Export and copy](#export-and-copy)
  - [Compaction](#compaction)
- [Context Loading](#context-loading)
  - [AGENTS.md](#agentsmd)
  - [Skills](#skills)
- [Providers and Models](#providers-and-models)
  - [OAuth and API keys](#oauth-and-api-keys)
  - [Local models](#local-models)
- [Permissions](#permissions)
- [Tool binaries](#tool-binaries)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Quick Start

### Install

```bash
uv tool install kon-coding-agent
```

### Run

```bash
kon
```

```text
usage: kon [-h] [--model MODEL]
           [--provider {azure-ai-foundry,github-copilot,openai,openai-codex,openai-responses,zhipu}]
           [--api-key API_KEY] [--base-url BASE_URL] [--continue]
           [--resume RESUME_SESSION] [--version]
           [--extra-tools EXTRA_TOOLS]

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
                        Resume a specific session by ID (full or unique prefix)
  --version             show program's version number and exit
  --extra-tools EXTRA_TOOLS
                        Comma-separated extra tools to enable (e.g. web_search,web_fetch)
```

### Common examples

```bash
# choose a model explicitly
kon -p openai-codex -m gpt-5.4

# continue your latest session
kon -c

# resume a specific session by id or unique prefix
kon -r 3f2a8c1b-...

# enable the optional web tools
kon --extra-tools web_search,web_fetch
```

### Install from source

```bash
git clone https://github.com/kuutsav/kon
cd kon
uv tool install .
```

> [!WARNING]
> Kon currently targets macOS and Linux. Windows is not tested yet.

---

## Why Kon

### Minimal by design

Kon tries to stay small in the places that matter most:

- **System prompt under 270 tokens** by default
- **6 core tools** for everyday coding work
- **2 optional extra tools** for web lookup and content extraction
- **Project instructions are externalized** through `AGENTS.md`
- **Heavily configurable** defaults that you can tune around model, prompt, permissions, compaction, tools, and UI
- **Useful features are borrowed selectively** from other agents, like `/handoff` inspired by Amp

That means the default harness stays lightweight, while your actual working behavior can still become rich through project context and configuration.

If you want a coding agent you can read, understand, fork, and adapt without inheriting a giant framework, that is the point of Kon.

### Configuration

Kon stores config at:

```text
~/.kon/config.toml
```

It is created automatically on first run, and old schemas are migrated forward automatically when needed.

Users are recommended to customize this config based on their model, workflow, safety preferences, and UI taste. For the shipped default config with inline comments, see [`src/kon/defaults/config.toml`](src/kon/defaults/config.toml).

Here is the full config shape:

```toml
[meta]
config_version = 4

[llm]
default_provider = "openai-codex"
default_model = "gpt-5.4"
default_base_url = ""
default_thinking_level = "high"
tool_call_idle_timeout_seconds = 180

[llm.auth]
openai_compat = "auto"
anthropic_compat = "auto"

[llm.system_prompt]
git_context = true
content = """You are an expert coding assistant called Kon.
..."""

[compaction]
on_overflow = "continue"
buffer_tokens = 20000

[agent]
max_turns = 500
default_context_window = 200000

[tools]
extra = ["web_search", "web_fetch"]

[ui]
theme = "gruvbox-dark"

[permissions]
mode = "prompt"
```

### Core tools

These are enabled by default:

| Tool | What it does | Why it matters |
| --- | --- | --- |
| `read` | Read file contents with pagination and image support | Keeps file inspection structured and token-efficient |
| `edit` | Exact text replacement | Good for surgical code changes |
| `write` | Create or fully overwrite files | Good for new files or full rewrites |
| `bash` | Run shell commands | For tests, git, builds, scripts, package managers |
| `grep` | Regex search inside files | Faster and cleaner than shelling out for content search |
| `find` | Glob-based file discovery | Fast repo navigation with `.gitignore` awareness |

This is the core experience: small, predictable, and enough for most coding tasks.

### Shell command input

Kon supports direct shell command execution from the input box. Simply prefix your command with `!`:

```bash
!ls -la
!git status
!python -m pytest tests/
```

This provides a quick way to run commands without switching to a separate terminal.

### Extra tools

Kon also ships optional built-in tools you can turn on when needed:

| Tool | Purpose | How to enable |
| --- | --- | --- |
| `web_search` | Search the web with DuckDuckGo | `--extra-tools web_search,web_fetch` or config |
| `web_fetch` | Fetch and extract clean page content | Usually paired with `web_search` |

Enable them from the CLI:

```bash
kon --extra-tools web_search,web_fetch
```

Or in `~/.kon/config.toml`:

```toml
[tools]
extra = ["web_search", "web_fetch"]
```

---

## Interactive TUI

Kon is built around a terminal UI that stays simple but practical.

### Editor and navigation

| Feature | How it works |
| --- | --- |
| File reference | Type `@` to fuzzy-search files and folders in the current project |
| Path completion | Press **Tab** to complete paths like `./`, `../`, `~`, quoted paths, and absolute paths |
| Queued prompts | Press **Enter** while the agent is running to queue a follow-up prompt |
| Steer queue | Press **Alt+Enter** to queue a steer message that is processed before normal queued prompts |
| Queue limit | Up to **5** normal queued prompts and **5** steer messages |
| Model switching | Use `/model` to switch interactively |
| Session browsing | Use `/resume` to browse prior sessions |
| Shell commands | Type `!` to execute shell commands directly from the input box |

### Slash commands

Type `/` at the start of the input box to see available commands.

| Command | Description |
| --- | --- |
| `/new` | Start a new conversation and reload project context |
| `/resume` | Browse and restore a saved session |
| `/model` | Switch model via picker |
| `/session` | Show session file, ids, message counts, and token stats |
| `/compact` | Compact the current conversation immediately |
| `/handoff` | Create a focused handoff into a new session |
| `/themes` | Switch UI themes |
| `/export` | Export current session to standalone HTML |
| `/copy` | Copy the last assistant response to the clipboard |
| `/login` | Authenticate with a supported OAuth provider |
| `/logout` | Remove provider credentials |
| `/clear` | Clear the current conversation |
| `/help` | Show help and keybindings |
| `/<custom>` | Custom skills registered as slash commands, shown in the /cmd popup for manual triggering |
| `/quit` (`/exit`, `/q`) | Quit Kon |

### Themes

Kon includes built-in themes and supports switching from inside the app:

- `gruvbox-dark`
- `gruvbox-light`
- `one-dark`
- `one-light`
- other built-in themes shipped with the app

Set one interactively with `/themes`, or persist it in config:

```toml
[ui]
theme = "gruvbox-dark"
```

---

## Sessions

Kon stores sessions as append-only **JSONL** files in:

```text
~/.kon/sessions/
```

That keeps sessions easy to inspect, archive, and move around.

### Resume and continue

You can restore work from the CLI or inside the TUI:

```bash
kon --continue
kon --resume <session-id>
```

Inside Kon:

- `/resume` opens an interactive session picker
- `/session` shows metadata and token usage
- the session picker also supports deleting old saved sessions

### Handoff

`/handoff <query>` starts a new focused session using a synthesized handoff prompt generated from the current conversation.

This is useful when a thread has grown broad and you want a fresh, narrower working context without losing the original session.

### Export and copy

- `/export` writes a standalone HTML transcript into the current working directory
- `/copy` copies the latest assistant response text to your clipboard

### Compaction

Long sessions eventually fill the context window. Kon supports both:

- **manual compaction** via `/compact`
- **automatic compaction** on overflow

Overflow behavior is configurable:

```toml
[compaction]
on_overflow = "continue" # or "pause"
buffer_tokens = 20000
```

The full session history still remains on disk in the JSONL file.

---

## Context Loading

Kon keeps the built-in harness small by moving project-specific instructions out of the fixed prompt.

### AGENTS.md

Kon loads project guidance from `AGENTS.md` or `CLAUDE.md` files into the system prompt.

Load order:

1. `~/.kon/AGENTS.md`
2. matching ancestor directories from git root (or home) down to the current working directory

Use these files for repo conventions, test commands, code style notes, deployment steps, or anything else you want loaded automatically.

### Skills

Skills are reusable instruction packs discovered from:

- project: `.kon/skills/`
- global: `~/.kon/skills/`

Each skill lives in its own directory with a `SKILL.md` file.

Example:

```markdown
---
name: my-skill
description: Brief description of what this skill does
register_cmd: true  # also registers the skill in the /cmd popup for manual triggering
cmd_info: Quick action shown in slash menu
---

# My Skill

Detailed instructions for the agent...
```

Important fields:

- `name` - skill identifier
- `description` - used for discovery and prompt context
- `register_cmd` - if `true`, exposes the skill as a slash command and includes it in the `/cmd` popup for manual triggering
- `cmd_info` - short help text for the slash menu

Validation highlights:

- lowercase letters, numbers, and `-` only
- no leading/trailing `-`
- no `--`
- max length for `name`: 64 chars

Skills are the main way to add reusable behaviors without bloating the default harness.

---

## Providers and Models

Kon works with hosted models and local models exposed through an OpenAI-compatible `/v1` API.

Built-in provider support includes:

- **GitHub Copilot**
- **OpenAI Codex**
- **OpenAI Responses / OpenAI-compatible endpoints**
- **Azure AI Foundry**
- **ZhiPu**

Use `/model` in the TUI to switch between available configured models.

### OAuth and API keys

Kon supports both OAuth login flows and direct API-key configuration.

- **GitHub Copilot OAuth**: run `/login` and choose GitHub Copilot
- **OpenAI OAuth**: run `/login` and choose OpenAI
- **OpenAI-compatible providers**: use `OPENAI_API_KEY` or provider-specific equivalents
- **Azure AI Foundry**: set `AZURE_AI_FOUNDRY_API_KEY` and `AZURE_AI_FOUNDRY_BASE_URL`

You can also pass credentials directly on launch:

```bash
kon --provider openai --model some-model --api-key "$OPENAI_API_KEY"
```

### Local models

Kon works well with local models served through an OpenAI-compatible endpoint. For one-off launches, you can force unauthenticated local behavior with `--openai-compat-auth none` or `--anthropic-compat-auth none`. To make that persistent across sessions, set `[llm.auth] openai_compat = "auto"|"none"` and/or `anthropic_compat = "auto"|"none"` in `~/.kon/config.toml`.

More notes, tested models, and examples live in [docs/local-models.md](docs/local-models.md).

---

### Permissions

Kon supports two permission modes:

| Mode | Behavior |
| --- | --- |
| `prompt` | Ask before mutating tool calls |
| `auto` | Skip approval prompts |

In `prompt` mode, non-mutating tools are allowed automatically, and some clearly read-only shell commands are also allowed.

```toml
[permissions]
mode = "prompt" # or "auto"
```

### Tool binaries

Kon depends on a few fast CLI tools for file discovery and search:

- **[`fd`](https://github.com/sharkdp/fd)** - required for fast file discovery
- **[`ripgrep`](https://github.com/BurntSushi/ripgrep)** - required for fast content search

If `fd` or `rg` are missing, Kon can download them automatically.

---

## Documentation

- [Local models](docs/local-models.md) - Running Kon with local OpenAI-compatible models, plus tested setups and example commands.

---

## Acknowledgements

- Kon takes significant inspiration from [pi coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent), especially around philosophy and UI direction.
- Kon also borrows ideas from Amp, Claude Code, and other terminal coding agents.

---

## License

MIT
