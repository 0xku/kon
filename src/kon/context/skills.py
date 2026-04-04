"""
Skills discovery and loading.

Skills are directories containing a SKILL.md file with frontmatter.
They provide specialized instructions that the model can read on-demand.

Discovery locations:
1. Global: ~/.kon/skills/
2. Project: <cwd>/.kon/skills/
"""

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .. import CONFIG_DIR_NAME, get_config_dir
from ._xml import escape_xml

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_CMD_INFO_LENGTH = 32


def shorten_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


@dataclass
class Skill:
    path: str
    name: str
    description: str
    register_cmd: bool = False
    cmd_info: str = ""
    bundled: bool = False


@dataclass
class SkillWarning:
    path: str
    message: str


@dataclass
class LoadSkillsResult:
    skills: list[Skill]
    warnings: list[SkillWarning]


def _parse_frontmatter(content: str) -> dict[str, Any]:
    if not content.startswith("---"):
        return {}

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}

    frontmatter_text = content[3 : end_match.start() + 3]

    result: dict[str, Any] = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            result[key] = value

    return result


def _validate_skill(
    name: str, description: str, parent_dir_name: str, file_path: str, cmd_info: str = ""
) -> list[SkillWarning]:
    warnings: list[SkillWarning] = []

    if name != parent_dir_name:
        warnings.append(
            SkillWarning(file_path, f'name "{name}" does not match directory "{parent_dir_name}"')
        )

    if len(name) > MAX_NAME_LENGTH:
        warnings.append(SkillWarning(file_path, f"name exceeds {MAX_NAME_LENGTH} characters"))

    if not re.match(r"^[a-z0-9-]+$", name):
        warnings.append(SkillWarning(file_path, "name must be lowercase a-z, 0-9, hyphens only"))

    if name.startswith("-") or name.endswith("-"):
        warnings.append(SkillWarning(file_path, "name must not start or end with hyphen"))

    if "--" in name:
        warnings.append(SkillWarning(file_path, "name must not contain consecutive hyphens"))

    if not description or not description.strip():
        warnings.append(SkillWarning(file_path, "description is required"))

    if len(description) > MAX_DESCRIPTION_LENGTH:
        warnings.append(
            SkillWarning(file_path, f"description exceeds {MAX_DESCRIPTION_LENGTH} characters")
        )

    if len(cmd_info) > MAX_CMD_INFO_LENGTH:
        warnings.append(
            SkillWarning(file_path, f"cmd_info exceeds {MAX_CMD_INFO_LENGTH} characters")
        )

    return warnings


def _load_skill_from_dir(skill_dir: Path) -> tuple[Skill | None, list[SkillWarning]]:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        return None, []

    warnings: list[SkillWarning] = []
    file_path = str(skill_file)

    try:
        content = skill_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)

        parent_dir_name = skill_dir.name
        name = frontmatter.get("name") or parent_dir_name
        description = frontmatter.get("description", "")
        register_cmd = _parse_bool(frontmatter.get("register_cmd"))
        cmd_info = str(frontmatter.get("cmd_info", "")).strip()

        warnings = _validate_skill(
            name, description, parent_dir_name, file_path, cmd_info=cmd_info
        )

        if not description or not description.strip():
            return None, warnings

        skill = Skill(
            name=name,
            description=description,
            path=file_path,
            register_cmd=register_cmd,
            cmd_info=cmd_info,
        )
        return skill, warnings

    except Exception as e:
        return None, [SkillWarning(file_path, str(e))]


def _load_skills_from_dir(directory: Path) -> LoadSkillsResult:
    skills: list[Skill] = []
    warnings: list[SkillWarning] = []

    if not directory.exists():
        return LoadSkillsResult(skills=skills, warnings=warnings)

    try:
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue

            skill, skill_warnings = _load_skill_from_dir(entry)
            warnings.extend(skill_warnings)
            if skill:
                skills.append(skill)

    except Exception:
        pass

    return LoadSkillsResult(skills=skills, warnings=warnings)


def load_skills(cwd: str | None = None) -> LoadSkillsResult:
    """
    Load skills from global and project locations.

    Discovery:
    1. <cwd>/.kon/skills/ - each subdirectory with SKILL.md is a skill
    2. ~/.kon/skills/ - each subdirectory with SKILL.md is a skill

    Local skills take precedence over global skills with the same name.
    """
    resolved_cwd = Path(cwd) if cwd else Path.cwd()
    resolved_cwd = resolved_cwd.resolve()

    skill_map: dict[str, Skill] = {}
    all_warnings: list[SkillWarning] = []

    def add_skills(result: LoadSkillsResult) -> None:
        all_warnings.extend(result.warnings)
        for skill in result.skills:
            if skill.name in skill_map:
                all_warnings.append(
                    SkillWarning(
                        skill.path,
                        f'name collision: "{skill.name}" already loaded '
                        f"from {shorten_path(skill_map[skill.name].path)}",
                    )
                )
            else:
                skill_map[skill.name] = skill

    local_skills_dir = (resolved_cwd / CONFIG_DIR_NAME / "skills").resolve(strict=False)
    global_skills_dir = (get_config_dir() / "skills").resolve(strict=False)

    add_skills(_load_skills_from_dir(local_skills_dir))
    if global_skills_dir != local_skills_dir:
        add_skills(_load_skills_from_dir(global_skills_dir))

    return LoadSkillsResult(skills=list(skill_map.values()), warnings=all_warnings)


def load_builtin_cmd_skills() -> LoadSkillsResult:
    try:
        builtin_resource = resources.files("kon").joinpath("builtin_skills")
        with resources.as_file(builtin_resource) as builtin_root:
            result = _load_skills_from_dir(builtin_root)
    except Exception:
        return LoadSkillsResult(skills=[], warnings=[])
    return LoadSkillsResult(
        skills=[
            Skill(
                path=skill.path,
                name=skill.name,
                description=skill.description,
                register_cmd=skill.register_cmd,
                cmd_info=skill.cmd_info,
                bundled=True,
            )
            for skill in result.skills
        ],
        warnings=result.warnings,
    )


def strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content.strip()
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return content.strip()
    return content[end_match.end() + 3 :].strip()


def render_skill_prompt(skill: Skill, query: str) -> str:
    try:
        content = Path(skill.path).read_text(encoding="utf-8")
    except Exception:
        return _build_fallback_skill_prompt(skill.description, query)
    template = strip_frontmatter(content)
    rendered = template.replace("$ARGUMENTS", query)
    return rendered.strip()


def _build_fallback_skill_prompt(description: str, query: str) -> str:
    query = query.strip()
    if not query:
        return description
    return f"{description}\n\n{query}"


def merge_registered_skills(primary: list[Skill], secondary: list[Skill]) -> list[Skill]:
    seen = {skill.name for skill in primary}
    merged = list(primary)
    for skill in secondary:
        if skill.name in seen:
            continue
        merged.append(skill)
        seen.add(skill.name)
    return merged


def formatted_skills(skills: list[Skill]) -> str:
    if not skills:
        return ""

    lines = [
        "# Skills",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "If a skill is manually triggered via slash command, its description is "
        "already included in the user message, so you usually don't need to read "
        "the skill file unless you need additional detail.",
        "",
        "<available_skills>",
    ]

    for skill in skills:
        lines.append("<skill>")
        lines.append(f"<name>{escape_xml(skill.name)}</name>")
        lines.append(f"<description>{escape_xml(skill.description)}</description>")
        lines.append(f"<location>{escape_xml(skill.path)}</location>")
        lines.append("</skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)
