---
name: kon-release-publish
description: Tag, publish to PyPI, and create GitHub release for Kon with validation and rollback-safe steps
---

# Kon Release + PyPI Publish

Use this skill when the user asks to cut a new Kon version, tag it, publish to PyPI, and/or create a GitHub release.

## Inputs to confirm

- Target version (example: `0.2.1`)
- Base range for notes (usually previous tag, example: `v0.2.0..HEAD`)
- Whether to push `main`
- Whether to publish to PyPI now
- Whether to create GitHub release now

## Versioning requirement

- Kon's update check only supports strict numeric `MAJOR.MINOR.PATCH` versions such as `0.2.7` and `0.3.0`.
- Do **not** cut releases with PyPI/PEP 440 prerelease or suffix forms like `0.3.0rc1`, `0.3.0b1`, `0.3.0.post1`, or `0.3.0.dev1` unless the update-check logic is updated first.
- Even though PyPI commonly allows those formats, Kon releases should continue following plain `X.Y.Z` so update detection stays correct.

## Files to bump

- `pyproject.toml` → `[project].version`
- `src/kon/version.py` → fallback `VERSION = "..."`
- `uv.lock` → local package version block

## Release workflow

1. **Preflight**
   - `git status --short --branch` must be clean (or confirm with user)
   - `git tag --list` and `git log --oneline <prev_tag>..HEAD` to summarize changes

2. **Update CHANGELOG.md**
   - Replace the `## [Unreleased]` section's `- No changes yet.` with a new versioned heading: `## <version> - YYYY-MM-DD`
   - Use `git log --oneline <prev_tag>..HEAD` to categorize changes into `### Added`, `### Changed`, `### Fixed` sections
   - Credit external contributors with `- @username`
   - Commit message: `docs: update changelog for <version>`

3. **Version bump**
   - Update version in all 3 files above

4. **Quality gates**
   - `uv run ruff format .`
   - `uv run ruff check .`
   - `uv run pyright .`
   - `uv run pytest`

5. **Commit**
   - Commit message: `build: bump version to <version>`

6. **Tag**
   - Annotated tag: `git tag -a v<version> -m "v<version> ..."`
   - Include concise "changes since previous tag" bullets

7. **Push**
   - `git push origin main`
   - `git push origin v<version>`

8. **Build + verify artifacts**
   - `rm -rf dist && uv build`
   - `uv run python -m twine check dist/*`

9. **Publish to PyPI**
   - Prefer token file if present (example `~/.pypi-token`):
   - `TWINE_USERNAME=__token__ TWINE_PASSWORD="$(< ~/.pypi-token)" uv run python -m twine upload dist/*`
   - Verify:
     - `https://pypi.org/project/kon-coding-agent/<version>/`
     - `https://pypi.org/pypi/kon-coding-agent/json` reports latest version

10. **Create GitHub release**
   - If token exists at `~/.github-token`, call Releases API:
   - `POST /repos/<owner>/<repo>/releases` with:
     - `tag_name: v<version>`
     - `target_commitish: main`
     - `name: v<version>`
     - `generate_release_notes: true`
   - If 403 occurs, report missing token scopes/permissions (`contents:write` required)

## Important notes

- **Tagging and GitHub release are separate**:
  - Tag = git ref in repository
  - Release = GitHub object attached to a tag (notes/assets)
- You can do either independently, but most projects do both together for user-facing releases.
- If PyPI publish succeeds but GitHub release fails, do **not** retag/re-publish. Just fix auth and create the release for the existing tag.

## Output checklist to report

- Changelog updated for `<version>`
- Version bumped in all files
- Checks passed
- Commit hash
- Tag created and pushed
- PyPI upload URL
- GitHub release URL (or exact error + remediation)
