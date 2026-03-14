# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added automatic user config migration with schema versioning and backup creation (`~/.kon/config.toml.bak.<timestamp>`).
- Added migration tests for legacy config key upgrades and no-op behavior on current schema versions.
- Added provider error-format tests to ensure empty upstream exceptions render with readable fallback text.

### Fixed

- Fixed blank error notifications (`✗` with no message) by normalizing empty provider/UI error messages.

## 0.2.6 - 2026-03-14

### New Features

- Added slash-triggered skill workflow in the TUI.

### Added

- Added clearer UI highlighting for informational/system notices.
- Added regression tests for compaction behavior and update notice behavior.
- Added a direct changelog link in update-available notices.

### Changed

- Improved tool output presentation, including truncated output labels and bash previews.
- Refined loaded resource headers in chat (`[Context]` and `[Skills]`) for better scanning.
- Renamed warning color usage to `notice` in UI color configuration for consistency.
- Simplified update notifications to always show the repository changelog URL.
- Updated README skills docs with `register_cmd` and `cmd_info` front matter fields and validation rules.

### Fixed

- Fixed compaction usage accounting by backtracking token usage correctly.
- Fixed markdown heading rendering by sanitizing inline code ticks.
- Fixed skill-trigger prompt formatting edge cases in UI messages.
- Removed italic styling from thinking blocks in both TUI and exported transcripts.

## 0.2.5 - 2026-03-14

- Added update-available notice in TUI.
- Improved configuration and context loading behavior.
- Added tests for update notice behavior.
