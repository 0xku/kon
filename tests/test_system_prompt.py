from kon import Config, reset_config, set_config
from kon.context import Context
from kon.loop import build_system_prompt


def test_system_prompt_includes_guidelines():
    set_config(Config({}))
    try:
        prompt = build_system_prompt("/tmp", Context("/tmp"))
    finally:
        reset_config()

    assert "Use grep to search file contents" in prompt
    assert "Use find to search for files by name/glob" in prompt
    assert "Use read to view files" in prompt
    assert "Use edit for precise changes" in prompt
    assert "Use write only for new files or complete rewrites" in prompt
    assert "Use bash for terminal operations" in prompt
    assert "Kon session logs are JSONL files in ~/.kon/sessions" in prompt


def test_system_prompt_includes_cwd():
    prompt = build_system_prompt("/test/dir", Context("/test/dir"))
    assert "/test/dir" in prompt


def test_system_prompt_excludes_git_context_by_default():
    set_config(Config({}))
    try:
        prompt = build_system_prompt("/tmp", Context("/tmp"))
    finally:
        reset_config()

    assert "<git-status>" not in prompt


def test_system_prompt_includes_git_context_when_enabled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    set_config(Config({"llm": {"system_prompt": {"git_context": True}}}))
    try:
        prompt = build_system_prompt(str(repo), Context(str(repo)))
    finally:
        reset_config()

    # Non-git directory should still omit the section
    assert "<git-status>" not in prompt
