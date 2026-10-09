from kon import config

from ..core.types import ToolDefinition
from .base import BaseTool
from .bash import BashTool
from .edit import EditTool
from .find import FindTool
from .grep import GrepTool
from .read import ReadTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
from .write import WriteTool

__all__ = [
    "BaseTool",
    "BashTool",
    "EditTool",
    "FindTool",
    "GrepTool",
    "ReadTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
    "get_tool",
    "get_tool_definitions",
    "get_tools",
    "resolve_tools",
    "tools_by_name",
]

all_tools = [
    ReadTool(),
    EditTool(),
    WriteTool(),
    BashTool(),
    GrepTool(),
    FindTool(),
    WebSearchTool(),
    WebFetchTool(),
]

tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in all_tools}


def get_tools(names: list[str]) -> list[BaseTool]:
    return [tool for tool in all_tools if tool.name in names]


def resolve_tools(cli_extra: list[str] | None = None) -> tuple[list[BaseTool], list[str]]:
    """Return the tools exposed to the agent plus any unknown configured names."""
    names = list(dict.fromkeys(config.tools.enabled + config.tools.extra + (cli_extra or [])))
    return get_tools(names), [name for name in names if name not in tools_by_name]


def get_tool(tool_name: str) -> BaseTool | None:
    return tools_by_name.get(tool_name)


def get_tool_definitions(tools: list[BaseTool]) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.params.model_json_schema(),
        )
        for tool in tools
    ]
