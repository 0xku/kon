<pre align="center">
░█░█░█▀█░█▀█
░█▀▄░█░█░█░█
░▀░▀░▀▀▀░▀░▀
</pre>
<p align="center">Minimal coding agent harness</p>
<p align="center">
  <a href="https://pypi.org/project/kon-coding-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/kon-coding-agent?style=flat-square" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square" /></a>
</p>

<p align="center">
  <img src="docs/images/kon-screenshot.png" alt="Kon terminal UI screenshot" width="700" />
</p>

Kon is a minimal coding agent focused on a tiny core prompt, a small built-in toolset, and project-specific context layered on top only when you want it. The default system prompt stays **under 270 tokens**, and even including the built-in tool descriptions and parameter schemas, the fixed harness stays at about **~1,000 tokens**. The core experience is built around **6 default tools** plus **2 extra web tools** included by the shipped config.

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
  - [Shell commands](#shell-commands)
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
           [--provider {azure-ai-foundry,deepseek,github-copilot,openai,openai-codex,openai-responses,xai,zhipu}]
           [--prompt [PROMPT]] [--api-key API_KEY] [--base-url BASE_URL]
           [--openai-compat-auth {auto,required,none}]
           [--anthropic-compat-auth {auto,required,none}]
           [--insecure-skip-verify] [--continue] [--resume RESUME_SESSION]
           [--version] [--extra-tools EXTRA_TOOLS]

Kon

options:
  -h, --help            show this help message and exit
  --model, -m MODEL     Model to use
  --provider {azure-ai-foundry,deepseek,github-copilot,openai,openai-codex,openai-responses,xai,zhipu}
                        Provider to use
  --prompt, -p [PROMPT]
                        Run a single prompt non-interactively, then exit (omit
                        the value or pipe stdin to read the prompt from stdin)
  --api-key, -k API_KEY
                        API key
  --base-url, -u BASE_URL
                        Base URL for API
  --openai-compat-auth {auto,required,none}
                        Auth mode for OpenAI-compatible endpoints
  --anthropic-compat-auth {auto,required,none}
                        Auth mode for Anthropic-compatible endpoints
  --insecure-skip-verify
                        Skip TLS verification (e.g. self-signed certs on local
                        providers)
  --continue, -c        Resume the most recent session
  --resume, -r RESUME_SESSION
                        Resume a specific session by ID (full or unique
                        prefix)
  --version             show program's version number and exit
  --extra-tools EXTRA_TOOLS
                        Comma-separated extra tools to enable (e.g.
                        web_search,web_fetch)
```

### Common examples

```bash
# choose a provider and model explicitly
kon --provider openai-codex -m gpt-5.6-sol

# continue your latest session
kon -c

# resume a specific session by id or unique prefix
kon -r 3f2a8c1b-...

# enable extra web tools for this run
kon --extra-tools web_search,web_fetch
```

**Once inside Kon, you can:**

```bash
# Run shell commands directly
!ls -la
!git status

# Run commands and get LLM analysis
!!grep -r "TODO" src/
!!find . -name "*.py" | head -20
```

### Non-interactive

Run a single prompt headlessly with `-p`/`--prompt`, then exit:

```bash
# pass the prompt inline
kon -p "fix the failing test"

# read the prompt from stdin
cat task.md | kon -p

# capture the final response
kon -p "summarize this module" > out.txt
```

In this mode tools run **auto-approved** (no confirmation prompts). The final assistant response is printed to stdout on a clean finish; errors go to stderr. Exit codes: `0` completed, `1` error, `2` startup error (empty prompt or provider/init failure), `3` hit the max-turn limit. Session flags (`-c`/`--continue`, `-r`/`--resume`) aren't supported in this mode.

### Install from source

```bash
git clone https://github.com/kuutsav/kon
cd kon
uv tool install .
```

> [!WARNING]
> Kon currently targets macOS and Linux. Windows is not tested yet.

### Development and testing

Run the test suite through `uv` so the dev dependencies are active — `pytest-asyncio` comes from the dev group, and a bare `pytest` without it fails every async test:

```bash
git clone https://github.com/kuutsav/kon
cd kon
uv sync --dev
uv run python -m pytest
```

**Windows note:** if you see many failures like `async def functions are not natively supported`, you are running a pytest without `pytest-asyncio` — re-run with `uv run python -m pytest` as above. Remaining Windows-specific failures are worth reporting as issues.

---

## Why Kon

### Minimal by design

Kon tries to stay small in the places that matter most:

- **System prompt under 270 tokens** by default
- **6 core tools** for everyday coding work
- **2 extra web tools** for web lookup and content extraction in the shipped config
- **Project instructions are externalized** through `AGENTS.md`
- **Heavily configurable** defaults that you can tune around model, prompt, permissions, compaction, tools, and UI
- **Useful features are borrowed selectively** from other agents, like `/handoff` inspired by Amp

That means the default harness stays lightweight, while your actual working behavior can still become rich through project context and configuration.

If you want a coding agent you can read, understand, fork, and adapt without inheriting a giant framework, that is the point of Kon.

### Configuration

Kon stores config at:

```text
~/.config/kon/config.toml
```

It is created automatically on first run, and old schemas are migrated forward automatically when needed.

Users are recommended to customize this config based on their model, workflow, safety preferences, and UI taste. For the shipped default config with inline comments, see [`src/kon/defaults/config.toml`](src/kon/defaults/config.toml).

Here is the full config shape:

```toml
[meta]
config_version = 6

[llm]
default_provider = "openai-codex" # "openai", "zhipu", "deepseek", "xai", "github-copilot", "openai-codex", "azure-ai-foundry"
default_model = "gpt-5.6-sol"
default_base_url = ""             # override the provider endpoint (e.g. a local server)
default_thinking_level = "low"    # "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra"
tool_call_idle_timeout_seconds = 180
request_timeout_seconds = 600

[llm.auth]
# Policy for OpenAI-/Anthropic-compatible endpoints.
# "auto" injects a placeholder key for local endpoints; "none" always does; "required" demands a real key.
openai_compat = "auto"
anthropic_compat = "auto"

[llm.tls]
insecure_skip_verify = false      # allow self-signed certs on local providers

[llm.system_prompt]
git_context = true
content = """You are an expert coding assistant called Kon.
..."""

[compaction]
on_overflow = "continue"          # "continue" keeps going after compaction; "pause" stops
buffer_tokens = 20000             # compact this many tokens before the context window fills

[agent]
max_turns = 500
default_context_window = 200000
turn_cooldown_seconds = 0         # pause this long after each LLM turn before the next one; 0 disables

[tools]
extra = ["web_search", "web_fetch"] # extra built-in tools beyond the core 6

[ui]
theme = "gruvbox-dark"
collapse_thinking = true          # collapse finalized thinking blocks to a summary
thinking_lines = "1"              # lines shown when collapsed; "none" = no truncation
colored_tool_badge = true         # colorize the tool icon/name badge on success
show_welcome_shortcuts = true     # show keyboard shortcuts on launch
hidden_models = []                # hide entries from the /model picker

[permissions]
mode = "prompt"                   # "prompt" asks before mutating actions; "auto" skips prompts

[notifications]
enabled = true                    # play audio when a task finishes, errors, or awaits approval
volume = 0.5                      # 0.0 (muted) to 1.0 (full)
```

The `ui.hidden_models` list trims the `/model` picker. Use a provider name (`"github-copilot"`) to hide every model from that provider, or `"provider:model"` (`"github-copilot:gpt-5.6-sol"`) to hide a single model. Hidden models stay usable via config defaults or session resume — they're just removed from the picker.

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

### Extra tools

Kon also ships extra built-in tools beyond the 6 core tools. The shipped config enables these web tools by default:

| Tool | Purpose | Config |
| --- | --- | --- |
| `web_search` | Search the web with DuckDuckGo | `extra = ["web_search", "web_fetch"]` |
| `web_fetch` | Fetch and extract clean page content | Usually paired with `web_search` |

You can also add extra tools for a single run from the CLI:

```bash
kon --extra-tools web_search,web_fetch
```

To disable web tools, remove them from `~/.config/kon/config.toml`:

```toml
[tools]
extra = []
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
| `/permissions` | Switch permission mode |
| `/thinking` | Switch thinking level for the current session |
| `/notifications` | Toggle notification sounds |
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

- `ayu`
- `catppuccin-frappe`
- `catppuccin-latte`
- `catppuccin-macchiato`
- `catppuccin-mocha`
- `dracula`
- `everforest`
- `flexoki`
- `github-dark`
- `github-light`
- `gruvbox-dark`
- `gruvbox-light`
- `kanagawa`
- `kanagawa-dragon`
- `monokai`
- `nightowl`
- `nord`
- `one-dark`
- `one-light`
- `osaka-jade`
- `palenight`
- `rosepine`
- `solarized-dark`
- `solarized-light`
- `tokyo-day`
- `tokyo-night`

Set one interactively with `/themes`, or persist it in config:

```toml
[ui]
theme = "gruvbox-dark"
```

### Shell commands

Kon supports direct shell command execution from the input box using two prefixes:

| Prefix | Behavior |
| --- | --- |
| `!command` | Run the command and show the result in chat |
| `!!command` | Run the command, show the result, and send the output to the LLM for follow-up |

**Examples:**

```bash
!ls -la              # List files in current directory
!git status          # Show git status
!python -m pytest tests/ -v  # Run tests

!!grep -r "TODO" src/    # Search for TODOs and analyze results
!!find . -name "*.py" | head -20  # Find Python files and get LLM insights
```

---

## Sessions

Kon stores sessions as append-only **JSONL** files in:

```text
~/.config/kon/sessions/
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

1. `~/.config/kon/AGENTS.md`
2. matching `AGENTS.md` or `CLAUDE.md` files in ancestor directories from git root (or home) down to the current working directory

Use these files for repo conventions, test commands, code style notes, deployment steps, or anything else you want loaded automatically.

### Skills

Skills are reusable instruction packs discovered from:

- project: `.agents/skills/`
- global: `~/.agents/skills/`

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
- `register_cmd` - if `true`, exposes the skill as a slash command and includes it in the `/cmd` popup for manual triggering; use `only` to register it as a slash command without including it in the system prompt
- `cmd_info` - short help text for the slash menu

> [!NOTE]
> Installed skills do **not** appear as slash commands by default — without `register_cmd` they are only described to the model, which invokes them on its own when relevant. Add `register_cmd: true` to a skill's frontmatter to trigger it manually as `/<skill-name>`.

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
- **DeepSeek**
- **xAI Grok**
- **ZhiPu**

Use `/model` in the TUI to switch between available configured models.

### OAuth and API keys

Kon supports both OAuth login flows and direct API-key configuration.

- **GitHub Copilot OAuth**: run `/login` and choose GitHub Copilot
- **OpenAI OAuth**: run `/login` and choose OpenAI
- **xAI OAuth**: run `/login` and choose xAI to sign in with SuperGrok or X Premium
- **OpenAI-compatible providers**: use `OPENAI_API_KEY` or provider-specific equivalents
  - OpenAI/default: `OPENAI_API_KEY` only
  - DeepSeek: `DEEPSEEK_API_KEY` first, then `OPENAI_API_KEY`
  - ZhiPu/ZAI: `ZAI_API_KEY` first, then `OPENAI_API_KEY`
- **Azure AI Foundry**: set `AZURE_AI_FOUNDRY_API_KEY` and `AZURE_AI_FOUNDRY_BASE_URL`

You can also pass API-key credentials directly on launch:

```bash
kon --provider openai --model some-model --api-key "$OPENAI_API_KEY"
kon --provider deepseek --model deepseek-v4-flash
```

After signing in to xAI via `/login`, launch Grok with:

```bash
kon --provider xai --model grok-4.5
```

### Local models

Kon works well with local models served through an OpenAI-compatible endpoint. For one-off launches, you can force unauthenticated local behavior with `--openai-compat-auth none` or `--anthropic-compat-auth none`. To make that persistent across sessions, set `[llm.auth] openai_compat = "auto"|"none"` and/or `anthropic_compat = "auto"|"none"` in `~/.config/kon/config.toml`.

More notes, tested models, and examples live in [docs/local-models.md](docs/local-models.md).

---

### Permissions

Kon supports two permission modes:

| Mode | Behavior |
| --- | --- |
| `prompt` | Ask before mutating tool calls |
| `auto` | Skip approval prompts |

In `prompt` mode, non-mutating tools are allowed automatically, and some clearly read-only shell commands are also allowed. This includes web tools: `web_search` and `web_fetch` are read-only, so they run **without an approval prompt** even in `prompt` mode. If you don't want the agent reaching the network at all, remove them from the `[tools] extra` config.

Use `/permissions` to switch modes for the current session and persist the change to config.

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

## Acknowledgements

- Kon takes significant inspiration from [pi coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent), especially around philosophy and UI direction.
- Kon also borrows ideas from Amp, Claude Code, and other terminal coding agents.

---

## License

MIT
