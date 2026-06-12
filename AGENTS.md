# Agent Guidelines

## Code Style

- Don't add trivial docstrings. Only add docstrings when explaining complex functionality.
- This project uses `uv`. Run `uv run ruff format .` after editing or creating any files.
- If generating and running a Python script, use `uv run python` instead of `python`.

## Testing

- Use `uv run python -m pytest` for testing in general; after edits/writes
- If the user asks for e2e tests then run the kon-tmux e2e test if available

## Skills

- Kon supports registering a skill as a slash command by setting `register_cmd: true` in the SKILL.md frontmatter. If a user asks for a "registered" skill, include this field.

## Committing code

- If the user tells you to commit code, look at all the changes and create multiple commits if needed based on logical groupings
- Follow commit message conventions: `docs:`, `feat:`, `fix:`, `build:`, etc. for the commit prefix

## Pushing

- If the user asks you to push code, run these first before doing so: `uv run ruff format .`, `uv run ruff check .`, `uv run python -m pyright .` and `uv run python -m pytest` in parallel (same tool call)
- Only if these all pass without issues should you push otherwise report the warnings/errors back to user and ask for next steps