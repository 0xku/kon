import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_package_name() -> str:
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text())
            return data["project"]["name"]
        except Exception:
            pass
    return "kon-coding-agent"


PACKAGE_NAME = _get_package_name()

try:
    VERSION = version(PACKAGE_NAME)
except PackageNotFoundError:
    VERSION = "0.4.0"
