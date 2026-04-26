# E2E Test Coverage Review

This note captures the current tmux E2E coverage, when it was last materially updated, and the strongest additional E2E tests to consider based on features added afterward.

## Current E2E test location

The current tmux E2E harness lives in the repo-local Kon skill:

- `.kon/skills/kon-tmux-test/SKILL.md`
- `.kon/skills/kon-tmux-test/run-e2e-tests.sh`

The main script launches Kon in a detached tmux session, drives keyboard input with `tmux send-keys`, captures pane output to `/tmp/kon-test-*.txt`, and relies on the reviewing agent/user to evaluate those captured outputs.

## Last material update

Git history for `.kon/skills/kon-tmux-test` shows the last material test update was:

```text
ae88aae 2026-03-18 21:18:04 +0530 test: make tmux e2e tab completion cases repo-deterministic
```

The only earlier tracked change for that skill path was the project rename:

```text
37b9a34 2026-02-22 12:15:57 +0530 Rename project from dot to kon
```

## Existing coverage

`run-e2e-tests.sh` currently covers:

| Area | Current checks |
| --- | --- |
| Slash trigger | Type `/`, verify slash command list appears |
| File search trigger | Type `@pyproject`, verify file picker appears |
| Model picker | Run `/model`, verify model selector appears |
| New conversation | Run `/new`, verify new conversation message |
| Tab path completion | Unique match, multiple alternatives, nested unique file, selecting from list |
| Tool execution | LLM-dependent write/edit/list/calculate prompt, verified through filesystem state |
| Session info | Run `/session`, verify session details/stats appear |
| Resume picker | Run `/resume`, verify saved sessions appear |

The coverage is strongest for basic UI triggers and tab completion. It is weaker for newer runtime controls, persistence, approval flows, export/compaction/handoff behavior, and CLI resume paths.

## Current documentation drift

`.kon/skills/kon-tmux-test/SKILL.md` is partially stale compared with `run-e2e-tests.sh`.

Examples:

- The skill doc still describes tab tests such as `sr` completing to `src/` and `~/De` showing home-directory alternatives.
- The actual script now tests repo-deterministic paths such as `pypr`, `src/kon/ui/s`, and `src/kon/ui/widg`.
- The listed output file names still include older descriptions such as `kon-test-7-tab-home.txt`, while the script now uses that filename for the nested `widgets.py` case.

Before adding many tests, update the skill doc so future evaluations match the script.

## Major features added after the last E2E update

Notable user-facing/runtime features added after `ae88aae` include:

- theme-based input border variants by thinking level
- optional web tools: `web_search` and `web_fetch`, enabled by `--extra-tools` or config
- tool approval previews and approval popup styling changes
- built-in and custom skill slash commands, including bundled `/init`
- queued prompts and Alt+Enter steer queue
- improved directory listings and tool display icons
- thinking collapse config
- session persistence fixes: system prompt persistence, provider rebuild on session load, thinking-level persistence in session header
- theme and model picker indicator improvements
- more built-in themes
- local/OpenAI-compatible auth mode flags
- configurable request timeout for API calls
- resume popup width fixes and width-aware popup/queue rendering
- rewritten standalone HTML session export
- manual `/compact` and automatic compaction improvements
- LaTeX math rendering in markdown output
- terminal bell/audio notifications and `/notifications` runtime controls
- permission popup/UI changes and info bar permission mode display
- runtime slash controls for `/permissions`, `/thinking`, and `/notifications`
- standardized thinking levels: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`

## Harness prerequisite: isolate HOME/config

Before adding tests that mutate runtime settings, update the harness to isolate the user environment.

The current script uses the real user config and sessions under `~/.kon`. That is risky for tests that exercise:

- `/themes`
- `/permissions`
- `/notifications`
- `--continue`
- `--resume`
- session deletion from `/resume`

Recommended harness change:

1. Create a temporary home, for example `/tmp/kon-e2e-home`.
2. Launch Kon with `HOME=/tmp/kon-e2e-home`.
3. Seed any required config and session fixtures under that temp home.
4. Clean up the temp home after the run, or keep it on failure for debugging.

This should be treated as a prerequisite for most new persistence-oriented E2E tests.

## Strongly recommended additions

### P0: Runtime mode controls and info bar

These directly cover the newest branch work and are mostly deterministic.

| Test | What to verify |
| --- | --- |
| Slash menu update | `/` menu includes newer commands: `/themes`, `/permissions`, `/thinking`, `/notifications`, `/init`, `/compact`, `/handoff`, `/export`, `/copy`, `/login`, `/logout` |
| `/permissions` picker | Picker opens, shows `prompt` and `auto`, current mode is checked |
| `/permissions auto` and `/permissions prompt` | Info bar updates to `✓✓ auto` / `⏸ prompt`, status says saved, temp config persists the selected mode |
| Shift+Tab permission cycling | Press Shift+Tab and verify permission mode toggles in the info bar and config |
| `/thinking` picker | Picker opens and lists `none`, `minimal`, `low`, `medium`, `high`, `xhigh` with current checkmark |
| `/thinking <level>` | Info bar model/thinking area updates, input border class behavior is indirectly visible through stable UI render, session records a thinking-level change |
| Ctrl+Shift+T thinking cycling | Press Ctrl+Shift+T and verify level cycles in the info bar |
| `/notifications` picker | Picker opens, shows `on` and `off`, current mode is checked |
| `/notifications on/off` | Status says saved and temp config `notifications.enabled` changes |
| Info bar row2 regression | Permission mode stays on row2-left; model/provider/thinking stays on row2-right after permission and thinking changes |

### P1: Session and persistence flows

These cover important features that have changed since the E2E script was last updated.

| Test | What to verify |
| --- | --- |
| `--continue` | Seed or create a saved session, launch `uv run kon --continue`, verify session restores and UI shows resumed content/status |
| `--resume <unique-prefix>` | Launch with a unique session ID prefix and verify the matching session loads |
| `/resume` delete flow | Open `/resume`, press Ctrl+D once for the delete hint, press Ctrl+D again, verify the selected session file is deleted or list updates |
| Session persistence after `/thinking` | Change thinking level, persist a turn, restart/resume, verify restored thinking level appears in info bar |
| Provider/model metadata on resume | Seed a session with model-change metadata and verify `/session`/info bar reflect it after resume |

### P1: Approval and tool-result UI

These are high-value because they exercise agent/tool/UI integration. They are more LLM-dependent, so keep prompts short and verify filesystem state instead of natural-language output.

| Test | What to verify |
| --- | --- |
| Mutating tool approval deny | In `prompt` mode, request a file write, capture approval popup preview, send `n`, verify file was not created |
| Mutating tool approval allow | Repeat with `y`, verify file exists and approval popup clears |
| File changes info bar | After write/edit tool calls, verify info bar shows changed file totals |
| Exit summary file changes | Exit after file changes and verify terminal summary includes changed-file totals |
| Tool display previews | For write/edit/bash/web tools where applicable, verify the captured pane includes concise preview/summary text and expected icons |

### P1: Export, copy, compact, and handoff commands

Several of these can be covered with deterministic negative-state tests, even before adding LLM-dependent positive paths.

| Test | What to verify |
| --- | --- |
| `/export` empty session | New empty session reports that there are no messages to export |
| `/export` after a real turn | HTML file is created in cwd and contains standalone transcript markers/content |
| `/copy` empty session | Shows `No agent messages to copy yet` |
| `/compact` empty session | Shows `No conversation to compact` |
| `/compact` while running | If practical, submit during a running request and verify `Cannot compact while agent is running` |
| `/handoff` without args | Shows usage message |
| `/handoff <query>` with no conversation | Shows `No conversation to handoff` |
| `/handoff` while running | If practical, verify `Cannot handoff while agent is running` |

### P1: Skills and slash command integration

| Test | What to verify |
| --- | --- |
| Built-in `/init` appears | Slash menu includes `/init` with its guided setup description |
| `/init` selection behavior | Selecting `/init` inserts/submits the skill trigger path correctly without breaking normal slash-command handling |
| Custom registered skill | Seed a temp project `.kon/skills/<name>/SKILL.md` with `register_cmd: true`, launch Kon, verify command appears in slash menu |
| Skill preview in resume list | If a skill-triggered session is persisted, `/resume` preview shows `/skill-name ...` instead of raw internal prompt scaffolding |

### P1: Queue and steer behavior

| Test | What to verify |
| --- | --- |
| Normal queue | While the agent is running, press Enter with a follow-up prompt and verify queue display appears |
| Steer queue | While running, submit with Alt+Enter and verify `[steer]` appears |
| Steer priority | Queue one normal prompt and one steer prompt; verify steer item is displayed before normal queued items |
| Queue limits | Fill more than five normal or steer items and verify warning appears |
| Interrupt clears queues | Queue items, press Escape during running request, verify queue display clears |

### P2: UI/input polish and optional tools

| Test | What to verify |
| --- | --- |
| `/themes` picker | Picker opens, shows built-in themes, current theme is checked |
| `/themes <id>` | Status/info message says theme changed and temp config persists the theme |
| Invalid `/themes` | Shows a useful invalid-theme error |
| `/model` picker update | Existing test should assert current checkmark and provider/no-vision labels, not just that a list appears |
| `/login` picker | Shows GitHub Copilot and OpenAI options without needing real auth |
| `/logout` with no creds | Shows `No providers logged in` |
| Large paste marker | Paste large multiline/long content and verify `[paste #N +x lines]` or `[paste #N y chars]` marker appears |
| Multiline input | Shift+Enter inserts newline; Enter submits |
| Optional web tools launch warning | Launch with unknown `--extra-tools` and verify warning; launch with `web_search,web_fetch` and verify no unknown-tool warning |
| LaTeX rendering on seeded resume | Seed assistant markdown with math, resume session, verify rendered Unicode/math output appears |

## Suggested implementation order

1. Isolate `HOME` and test config/session directories in the tmux harness.
2. Update `.kon/skills/kon-tmux-test/SKILL.md` to match the current script.
3. Add P0 runtime-mode and info-bar tests.
4. Add deterministic command negative-state tests for `/compact`, `/handoff`, `/copy`, and `/export`.
5. Add session resume/continue/delete tests using seeded fixtures.
6. Add approval-flow tests once the harness can reliably answer `y`/`n` to tool approval prompts.
7. Add queue/steer and larger LLM-dependent tests last, because they are more timing-sensitive.

## Evaluation notes

Prefer deterministic checks where possible:

- Verify config files under temp `HOME` for persisted runtime settings.
- Verify session JSONL contents for thinking/model/session persistence.
- Verify filesystem state for write/edit/export tests.
- Use captured pane output for picker visibility and status/info messages.
- Avoid asserting exact AI prose unless the prompt and model response are tightly controlled.

The existing E2E philosophy is still sound: shell scripts should drive and capture the app, while humans/agents evaluate the captured output. The main gap is breadth: the script predates much of the runtime-mode, session, export, notification, skill, and queue functionality now present in Kon.
