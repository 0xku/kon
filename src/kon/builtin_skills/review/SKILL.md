---
name: review
description: Review code changes and return prioritized, actionable findings
register_cmd: true
cmd_info: review code changes
---

Review the requested code changes as if you are reviewing another engineer's PR.

User-provided target or constraints (honor these):
$ARGUMENTS

## Target selection

- If no target is provided, review the current code changes: staged, unstaged, and untracked files.
- If the user provides a base branch, review the changes that would merge into that branch. Find the merge base and inspect the diff from that commit.
- If the user provides a commit SHA, review only the changes introduced by that commit.
- If the user provides a PR number, PR URL, or text like `PR#68 feat/headless-mode "feat: add non-interactive prompt mode"`, assume the GitHub CLI is available. Use `gh pr view` to inspect PR metadata, resolve the base/head branches, fetch as needed, compute the merge base, and review the PR diff. Do not checkout a different branch unless the user explicitly asks or confirms.
- For PRs, do not assume the head branch exists on `origin`; it may live on a contributor fork. Prefer `gh pr checkout --detach <number>` only if checkout is acceptable, or fetch the head repository/ref reported by `gh pr view --json headRepository,headRefName` into a temporary remote/ref before diffing.
- For a PR scenario such as `/review PR#68 feat/headless-mode "feat: add non-interactive prompt mode"`, use the PR number/title/branch as hints, verify them with `gh pr view 68`, then review the code changes relative to the PR base branch.

## Review rubric

Only report issues the original author would likely fix if they knew about them.

Flag a finding only when:
- it meaningfully affects correctness, security, performance, reliability, or maintainability;
- it is discrete and actionable;
- it appears introduced by the reviewed change;
- you can identify the affected code path or user scenario;
- it is not merely a style preference, nit, or intentional behavior change.

Do not stop at the first issue. Return all qualifying findings. If there are no findings worth fixing, say so clearly.

## Investigation guidance

Use repository tools to inspect the actual diff and surrounding code. Prefer precise commands such as:
- `git status --short`
- `git diff --staged`
- `git diff`
- `git diff <merge-base>...HEAD`
- `git show <sha>`
- `gh pr view <number> --json number,title,body,baseRefName,headRefName,url,author`

Read nearby implementation and tests when needed to prove whether a suspected issue is real. Avoid speculative findings.

## Finding format

For each finding, include:
- priority tag in the title: `[P0]`, `[P1]`, `[P2]`, or `[P3]`;
- concise title;
- file path and line range;
- one short paragraph explaining why this is a bug and when it matters;
- confidence score if useful.

Keep line ranges as short as possible and make sure they overlap the reviewed diff when possible.

End with an overall verdict:
- `patch is correct` if no blocking/real bugs were found;
- `patch is incorrect` if one or more findings indicate the patch can break behavior or needs correction.

Do not fix the code unless the user asks after the review.
