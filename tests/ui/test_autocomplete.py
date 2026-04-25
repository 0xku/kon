import shutil
from pathlib import Path

import pytest

from kon.ui.autocomplete import (
    DEFAULT_COMMANDS,
    CompletionResult,
    FilePathProvider,
    ListItem,
    SlashCommand,
    SlashCommandProvider,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "src" / "utils.py").write_text("# utils")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("# test")
    (tmp_path / "README.md").write_text("# readme")
    (tmp_path / "pyproject.toml").write_text("[tool]")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "src" / "models").mkdir()
    (tmp_path / "src" / "models" / "user.py").write_text("# user")
    return tmp_path


@pytest.fixture
def provider(temp_dir: Path) -> FilePathProvider:
    paths = [
        "data.json",
        "pyproject.toml",
        "README.md",
        "src/",
        "src/main.py",
        "src/utils.py",
        "src/models/",
        "src/models/user.py",
        "tests/",
        "tests/test_main.py",
    ]
    provider = FilePathProvider(cwd=str(temp_dir))
    provider.set_paths(paths)
    return provider


@pytest.fixture
def fd_available():
    return bool(shutil.which("fd") or shutil.which("fdfind"))


# -------------------------------------------------------------------------
# Behavior tests (work regardless of fd/fallback implementation)
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,cursor_col,should_trigger",
    [
        ("@", 1, True),
        ("@src", 4, True),
        ("some text @", 11, True),
        ("some text @src", 15, True),
        ("@src/", 5, True),
        ("@", 0, False),
    ],
)
def test_should_trigger(
    provider: FilePathProvider, text: str, cursor_col: int, should_trigger: bool
):
    assert provider.should_trigger(text, cursor_col) == should_trigger


def test_should_not_trigger(provider: FilePathProvider):
    assert not provider.should_trigger("file@test", 8)
    assert not provider.should_trigger("test@test@test", 14)
    assert not provider.should_trigger("hello world", 11)
    assert not provider.should_trigger("", 0)


def test_empty_query(provider: FilePathProvider):
    result = provider.get_suggestions("@", 1)

    assert result is not None
    assert isinstance(result, CompletionResult)
    assert result.prefix == "@"
    assert result.replace_start == 0
    assert len(result.items) > 0

    for item in result.items[:5]:
        assert isinstance(item, ListItem)
        assert isinstance(item.value, str)
        assert isinstance(item.label, str)
        assert isinstance(item.description, str)


def test_suggestions_with_query(provider: FilePathProvider):
    result = provider.get_suggestions("@src", 4)

    assert result is not None
    assert result.prefix == "@src"
    assert result.replace_start == 0

    labels = [item.label for item in result.items]
    assert any("src" in label for label in labels)


def test_fuzzy_filtering(provider: FilePathProvider):
    result = provider.get_suggestions("@main", 5)

    assert result is not None
    labels = [item.label for item in result.items]
    assert "main.py" in labels


def test_no_results(provider: FilePathProvider):
    result = provider.get_suggestions("@nonexistent_file_xyz", 23)
    assert result is None


def test_apply_completion_file(provider: FilePathProvider):
    item = ListItem(value="src/main.py", label="main.py", description="src")
    text = "@main"
    cursor_col = 5
    prefix = "@main"

    new_text, new_cursor = provider.apply_completion(text, cursor_col, item, prefix)

    assert new_text == "@src/main.py "
    assert new_cursor == len("@src/main.py ")


def test_apply_completion_directory(provider: FilePathProvider):
    item = ListItem(value="src/", label="src/", description=".")
    text = "@src"
    cursor_col = 4
    prefix = "@src"

    new_text, new_cursor = provider.apply_completion(text, cursor_col, item, prefix)

    assert new_text == "@src/"
    assert new_cursor == len("@src/")


def test_fuzzy_matcher_scoring(provider: FilePathProvider):
    result_src = provider.get_suggestions("@src", 4)
    result_test = provider.get_suggestions("@test", 5)

    assert result_src is not None
    assert result_test is not None

    src_labels = [item.label for item in result_src.items]
    test_labels = [item.label for item in result_test.items]

    assert any("src" in label for label in src_labels)
    assert any("test" in label for label in test_labels)


def test_trigger_chars(provider: FilePathProvider):
    assert provider.trigger_chars == {"@"}


def test_limit_results(provider: FilePathProvider):
    result = provider.get_suggestions("@", 1)

    assert result is not None
    assert len(result.items) <= 20


def test_nested_paths(provider: FilePathProvider):
    result = provider.get_suggestions("@user", 5)

    assert result is not None
    labels = [item.label for item in result.items]
    assert any("user" in label for label in labels)


def test_empty_text(provider: FilePathProvider):
    assert not provider.should_trigger("", 0)


def test_cursor_position_variations(provider: FilePathProvider):
    text = "test @file"
    assert provider.should_trigger(text, 6)
    assert provider.should_trigger(text, 10)
    assert not provider.should_trigger(text, 4)


def test_special_characters_in_query(provider: FilePathProvider):
    result = provider.get_suggestions("@test-", 6)
    assert result is None or isinstance(result, CompletionResult)


def test_set_cwd(provider: FilePathProvider):
    new_cwd = "/tmp"
    provider.set_cwd(new_cwd)
    assert provider._cwd == new_cwd


def test_large_number_of_cached_paths(tmp_path: Path):
    for i in range(100):
        (tmp_path / f"file_{i}.py").write_text(f"# file {i}")

    paths = [f"file_{i}.py" for i in range(100)]
    provider = FilePathProvider(cwd=str(tmp_path))
    provider.set_paths(paths)

    result = provider.get_suggestions("@", 1)
    assert result is not None
    assert len(result.items) <= 20

    result = provider.get_suggestions("@file_5", 8)
    assert result is not None
    assert len(result.items) > 0


def test_unicode_in_paths(provider: FilePathProvider):
    provider._cached_paths.append("émoji_file.py")

    result = provider.get_suggestions("@émoji", 6)
    assert result is None or isinstance(result, CompletionResult)


def test_case_insensitive_matching(provider: FilePathProvider):
    result_lower = provider.get_suggestions("@main", 5)
    result_upper = provider.get_suggestions("@MAIN", 5)
    result_mixed = provider.get_suggestions("@Main", 5)

    assert result_lower is not None
    assert result_upper is not None
    assert result_mixed is not None

    labels_lower = [item.label for item in result_lower.items]
    labels_upper = [item.label for item in result_upper.items]
    labels_mixed = [item.label for item in result_mixed.items]

    assert "main.py" in labels_lower
    assert "main.py" in labels_upper
    assert "main.py" in labels_mixed


def test_completion_result_structure(provider: FilePathProvider):
    result = provider.get_suggestions("@", 1)

    assert result is not None
    assert hasattr(result, "items")
    assert hasattr(result, "prefix")
    assert hasattr(result, "replace_start")

    assert isinstance(result.items, list)
    assert isinstance(result.prefix, str)
    assert isinstance(result.replace_start, int)
    assert result.replace_start >= 0


# -------------------------------------------------------------------------
# Implementation verification (minimal, only what's worth checking)
# -------------------------------------------------------------------------


def test_fd_used_when_available(temp_dir: Path, fd_available: bool):
    if not fd_available:
        pytest.skip("fd binary not available")

    fd_path = shutil.which("fd") or shutil.which("fdfind")
    provider = FilePathProvider(cwd=str(temp_dir), fd_path=fd_path)

    result = provider.get_suggestions("@main", 5)
    assert result is not None
    labels = [item.label for item in result.items]
    assert any("main" in label.lower() for label in labels)


def test_fallback_works_without_fd(provider: FilePathProvider):
    provider.set_fd_path(None)
    result = provider.get_suggestions("@main", 5)
    assert result is not None
    labels = [item.label for item in result.items]
    assert "main.py" in labels


def test_default_slash_commands_include_copy_compact_and_handoff():
    names = {cmd.name for cmd in DEFAULT_COMMANDS}
    assert "copy" in names
    assert "compact" in names
    assert "handoff" in names
    assert "permissions" in names
    assert "thinking" in names
    assert "notifications" in names


def test_slash_provider_triggers_mid_input_for_skills_only():
    provider = SlashCommandProvider(
        [SlashCommand("compact", "Compact"), SlashCommand("custom-skill", "Skill", is_skill=True)]
    )

    text = "please run /cus"
    cursor_col = len(text)

    assert provider.should_trigger(text, cursor_col)
    result = provider.get_suggestions(text, cursor_col)
    assert result is not None
    assert result.prefix == "/cus"
    assert result.replace_start == text.rfind("/")
    labels = [item.label for item in result.items]
    assert labels == ["/custom-skill"]


def test_slash_provider_mid_input_does_not_trigger_without_skills():
    provider = SlashCommandProvider([SlashCommand("compact", "Compact")])

    text = "please run /co"
    cursor_col = len(text)

    assert not provider.should_trigger(text, cursor_col)
    assert provider.get_suggestions(text, cursor_col) is None


def test_slash_provider_start_shows_full_menu():
    provider = SlashCommandProvider(
        [SlashCommand("compact", "Compact"), SlashCommand("custom-skill", "Skill", is_skill=True)]
    )

    text = "/"
    cursor_col = len(text)

    assert provider.should_trigger(text, cursor_col)
    result = provider.get_suggestions(text, cursor_col)
    assert result is not None
    labels = [item.label for item in result.items]
    assert "/compact" in labels
    assert "/custom-skill" in labels


def test_slash_provider_does_not_trigger_inside_word():
    provider = SlashCommandProvider([SlashCommand("compact", "Compact")])

    text = "path/compact"
    cursor_col = len(text)

    assert not provider.should_trigger(text, cursor_col)
    assert provider.get_suggestions(text, cursor_col) is None
