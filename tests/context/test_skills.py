from kon.context.skills import (
    Skill,
    _load_skill_from_dir,
    _parse_frontmatter,
    _validate_skill,
    formatted_skills,
    load_builtin_cmd_skills,
    load_skills,
    merge_registered_skills,
    render_skill_prompt,
    strip_frontmatter,
)


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        content = """---
name: my-skill
description: A test skill
---
# Content here
"""
        result = _parse_frontmatter(content)

        assert result["name"] == "my-skill"
        assert result["description"] == "A test skill"

    def test_no_frontmatter(self):
        content = "# Just markdown\nNo frontmatter here"
        result = _parse_frontmatter(content)

        assert result == {}

    def test_missing_closing_delimiter(self):
        content = """---
name: my-skill
description: broken
# No closing ---
"""
        result = _parse_frontmatter(content)

        assert result == {}

    def test_quoted_values_double(self):
        content = """---
name: "quoted-name"
description: "A quoted description"
---
"""
        result = _parse_frontmatter(content)

        assert result["name"] == "quoted-name"
        assert result["description"] == "A quoted description"

    def test_quoted_values_single(self):
        content = """---
name: 'single-quoted'
description: 'Another description'
register_cmd: 'true'
cmd_info: 'slash hint'
---
"""
        result = _parse_frontmatter(content)

        assert result["name"] == "single-quoted"
        assert result["description"] == "Another description"
        assert result["register_cmd"] == "true"
        assert result["cmd_info"] == "slash hint"

    def test_empty_values(self):
        content = """---
name:
description:
---
"""
        result = _parse_frontmatter(content)

        assert result["name"] == ""
        assert result["description"] == ""

    def test_comments_ignored(self):
        content = """---
# This is a comment
name: my-skill
# Another comment
description: test
---
"""
        result = _parse_frontmatter(content)

        assert result["name"] == "my-skill"
        assert result["description"] == "test"
        assert "#" not in result.get("name", "")

    def test_colon_in_value(self):
        content = """---
name: my-skill
description: This has: a colon in it
---
"""
        result = _parse_frontmatter(content)

        assert result["description"] == "This has: a colon in it"

    def test_whitespace_handling(self):
        content = """---
  name:   spaced-skill\x20\x20
  description:   Lots of spaces\x20\x20\x20
---
"""
        result = _parse_frontmatter(content)

        assert result["name"] == "spaced-skill"
        assert result["description"] == "Lots of spaces"


class TestValidateSkill:
    def test_valid_skill(self):
        warnings = _validate_skill(
            "my-skill", "A valid description", "my-skill", "/path/SKILL.md", cmd_info="menu"
        )

        assert warnings == []

    def test_name_mismatch(self):
        warnings = _validate_skill("skill-name", "Description", "different-dir", "/path/SKILL.md")

        assert len(warnings) == 1
        assert "does not match directory" in warnings[0].message

    def test_name_too_long(self):
        long_name = "a" * 65
        warnings = _validate_skill(long_name, "Description", long_name, "/path/SKILL.md")

        assert any("exceeds 64 characters" in w.message for w in warnings)

    def test_name_uppercase_invalid(self):
        warnings = _validate_skill("MySkill", "Description", "MySkill", "/path/SKILL.md")

        assert any("lowercase" in w.message for w in warnings)

    def test_name_special_chars_invalid(self):
        warnings = _validate_skill("my_skill", "Description", "my_skill", "/path/SKILL.md")

        assert any("lowercase a-z, 0-9, hyphens only" in w.message for w in warnings)

    def test_name_starts_with_hyphen(self):
        warnings = _validate_skill("-skill", "Description", "-skill", "/path/SKILL.md")

        assert any("start or end with hyphen" in w.message for w in warnings)

    def test_name_ends_with_hyphen(self):
        warnings = _validate_skill("skill-", "Description", "skill-", "/path/SKILL.md")

        assert any("start or end with hyphen" in w.message for w in warnings)

    def test_name_consecutive_hyphens(self):
        warnings = _validate_skill("my--skill", "Description", "my--skill", "/path/SKILL.md")

        assert any("consecutive hyphens" in w.message for w in warnings)

    def test_empty_description(self):
        warnings = _validate_skill("my-skill", "", "my-skill", "/path/SKILL.md")

        assert any("description is required" in w.message for w in warnings)

    def test_whitespace_only_description(self):
        warnings = _validate_skill("my-skill", "   ", "my-skill", "/path/SKILL.md")

        assert any("description is required" in w.message for w in warnings)

    def test_description_too_long(self):
        long_desc = "a" * 1025
        warnings = _validate_skill("my-skill", long_desc, "my-skill", "/path/SKILL.md")

        assert any("exceeds 1024 characters" in w.message for w in warnings)

    def test_multiple_errors(self):
        warnings = _validate_skill("MY--SKILL-", "", "wrong-dir", "/path/SKILL.md")

        assert len(warnings) >= 3

    def test_cmd_info_too_long(self):
        warnings = _validate_skill(
            "my-skill", "Description", "my-skill", "/path/SKILL.md", cmd_info="x" * 33
        )

        assert any("cmd_info exceeds 32 characters" in w.message for w in warnings)


class TestLoadSkillFromDir:
    def test_load_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: A test skill
register_cmd: true
cmd_info: Quick publish helper
---
# Skill content
""")

        skill, warnings = _load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "A test skill"
        assert skill.register_cmd is True
        assert skill.cmd_info == "Quick publish helper"
        assert warnings == []

    def test_no_skill_file(self, tmp_path):
        skill_dir = tmp_path / "empty-dir"
        skill_dir.mkdir()

        skill, warnings = _load_skill_from_dir(skill_dir)

        assert skill is None
        assert warnings == []

    def test_skill_without_description_returns_none(self, tmp_path):
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: bad-skill
---
# No description
""")

        skill, warnings = _load_skill_from_dir(skill_dir)

        assert skill is None
        assert any("description is required" in w.message for w in warnings)

    def test_uses_dir_name_as_fallback(self, tmp_path):
        skill_dir = tmp_path / "fallback-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
description: Uses directory name
---
""")

        skill, _warnings = _load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.name == "fallback-skill"
        assert skill.register_cmd is False
        assert skill.cmd_info == ""

    def test_register_cmd_parses_truthy_strings(self, tmp_path):
        skill_dir = tmp_path / "cmd-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: cmd-skill
description: Slash skill
register_cmd: yes
---
""")

        skill, warnings = _load_skill_from_dir(skill_dir)

        assert skill is not None
        assert skill.register_cmd is True
        assert warnings == []


class TestLoadSkills:
    def test_loads_local_and_global_unique_skills(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        local_skills_dir = repo / ".kon" / "skills"
        local_skill_dir = local_skills_dir / "local-skill"
        local_skill_dir.mkdir(parents=True)
        (local_skill_dir / "SKILL.md").write_text("""---
name: local-skill
description: Local skill
---
""")

        global_dir = tmp_path / "global"
        global_skills_dir = global_dir / "skills"
        global_skill_dir = global_skills_dir / "global-skill"
        global_skill_dir.mkdir(parents=True)
        (global_skill_dir / "SKILL.md").write_text("""---
name: global-skill
description: Global skill
---
""")

        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: global_dir)

        result = load_skills(str(repo))

        assert {s.name for s in result.skills} == {"local-skill", "global-skill"}
        assert result.warnings == []

    def test_local_overrides_global_name_collision(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        local_skill_dir = repo / ".kon" / "skills" / "shared-skill"
        local_skill_dir.mkdir(parents=True)
        (local_skill_dir / "SKILL.md").write_text("""---
name: shared-skill
description: Local version
---
""")

        global_dir = tmp_path / "global"
        global_skill_dir = global_dir / "skills" / "shared-skill"
        global_skill_dir.mkdir(parents=True)
        (global_skill_dir / "SKILL.md").write_text("""---
name: shared-skill
description: Global version
---
""")

        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: global_dir)
        monkeypatch.setenv("HOME", str(tmp_path))

        result = load_skills(str(repo))

        assert len(result.skills) == 1
        assert result.skills[0].name == "shared-skill"
        assert result.skills[0].path == str(local_skill_dir / "SKILL.md")
        collision = next(w for w in result.warnings if "name collision" in w.message)
        expected = (
            'name collision: "shared-skill" already loaded '
            "from ~/repo/.kon/skills/shared-skill/SKILL.md"
        )
        assert collision.message == expected

    def test_skips_duplicate_global_dir_when_cwd_is_home(self, tmp_path, monkeypatch):
        home_dir = tmp_path / "home"
        skill_dir = home_dir / ".kon" / "skills" / "noc"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("""---
name: noc
description: Planning-only mode
---
""")

        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: home_dir / ".kon")

        result = load_skills(str(home_dir))

        assert len(result.skills) == 1
        assert result.skills[0].name == "noc"
        assert result.skills[0].path == str(skill_dir / "SKILL.md")
        assert result.warnings == []

    def test_empty_when_no_skill_directories(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        global_dir = tmp_path / "global"

        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: global_dir)

        result = load_skills(str(repo))

        assert result.skills == []
        assert result.warnings == []

    def test_invalid_skill_excluded_and_warning_returned(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        invalid_skill_dir = repo / ".kon" / "skills" / "invalid-skill"
        invalid_skill_dir.mkdir(parents=True)
        (invalid_skill_dir / "SKILL.md").write_text("""---
name: invalid-skill
---
# missing description
""")

        global_dir = tmp_path / "global"
        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: global_dir)

        result = load_skills(str(repo))

        assert result.skills == []
        assert any("description is required" in w.message for w in result.warnings)

    def test_uses_directory_name_when_name_missing_in_frontmatter(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        skill_dir = repo / ".kon" / "skills" / "fallback-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("""---
description: Uses directory fallback
---
""")

        global_dir = tmp_path / "global"
        monkeypatch.setattr("kon.context.skills.get_config_dir", lambda: global_dir)

        result = load_skills(str(repo))

        assert len(result.skills) == 1
        assert result.skills[0].name == "fallback-name"
        assert result.warnings == []


class TestBuiltinCommandSkills:
    def test_loads_builtin_init_skill(self):
        result = load_builtin_cmd_skills()

        skill = next((s for s in result.skills if s.name == "init"), None)
        assert skill is not None
        assert skill.register_cmd is True
        assert skill.cmd_info == "Guided AGENTS.md setup"
        assert skill.bundled is True
        assert skill.path.endswith("kon/builtin_skills/init/SKILL.md")
        assert result.warnings == []


class TestBundledSkillPromptRendering:
    def test_strip_frontmatter(self):
        content = """---
name: init
description: Example
---

Body here
"""

        assert strip_frontmatter(content) == "Body here"

    def test_render_skill_prompt_substitutes_arguments(self):
        skill = next((s for s in load_builtin_cmd_skills().skills if s.name == "init"), None)

        assert skill is not None
        prompt = render_skill_prompt(skill, "focus on testing")

        assert "Create or update `AGENTS.md` for this repository." in prompt
        assert "focus on testing" in prompt
        assert "$ARGUMENTS" not in prompt


class TestMergeRegisteredSkills:
    def test_primary_skills_take_precedence(self):
        primary = [
            Skill(name="init", description="Primary", path="/primary/SKILL.md", register_cmd=True)
        ]
        secondary = [
            Skill(
                name="init", description="Secondary", path="/secondary/SKILL.md", register_cmd=True
            ),
            Skill(name="other", description="Other", path="/other/SKILL.md", register_cmd=True),
        ]

        merged = merge_registered_skills(primary, secondary)

        assert [skill.name for skill in merged] == ["init", "other"]
        assert merged[0].path == "/primary/SKILL.md"
        assert merged[1].path == "/other/SKILL.md"


class TestFormatSkillsForPrompt:
    def test_empty_skills(self):
        result = formatted_skills([])

        assert result == ""

    def test_single_skill(self):
        skills = [Skill(name="test-skill", description="A test skill", path="/path/to/SKILL.md")]

        result = formatted_skills(skills)

        assert "# Skills" in result
        assert "<available_skills>" in result
        assert "<name>test-skill</name>" in result
        assert "<description>A test skill</description>" in result
        assert "<location>/path/to/SKILL.md</location>" in result
        assert "</available_skills>" in result

    def test_escapes_xml_chars(self):
        skills = [
            Skill(
                name="test-skill", description='Uses <angle> & "quotes"', path="/path/to/SKILL.md"
            )
        ]

        result = formatted_skills(skills)

        assert "&lt;angle&gt;" in result
        assert "&amp;" in result
        assert "&quot;quotes&quot;" in result

    def test_multiple_skills(self):
        skills = [
            Skill(name="skill-a", description="First", path="/a/SKILL.md"),
            Skill(name="skill-b", description="Second", path="/b/SKILL.md"),
        ]

        result = formatted_skills(skills)

        assert "<name>skill-a</name>" in result
        assert "<name>skill-b</name>" in result
