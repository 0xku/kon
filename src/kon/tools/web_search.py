import asyncio

from pydantic import BaseModel, Field

from kon import config

from ..core.types import ToolResult
from .base import BaseTool


class WebSearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(
        description="Number of results to return (default 3)", default=3, ge=1, le=10
    )


class WebSearchTool(BaseTool):
    name = "web_search"
    mutating = False
    params = WebSearchParams
    description = (
        "Search the web using DuckDuckGo. "
        "Returns titles, URLs, and snippets for each result. "
        "Use web_fetch to read full page content from a result URL."
    )

    def format_call(self, params: WebSearchParams) -> str:
        accent = config.ui.colors.accent
        query = params.query.replace('"', '\\"')
        return f'[{accent}]"{query}"[/{accent}]'

    async def execute(
        self, params: WebSearchParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(
                success=False, display="[red]ddgs package not installed (pip install ddgs)[/red]"
            )

        def _search() -> list[dict]:
            return list(DDGS().text(params.query, max_results=params.max_results))

        try:
            work = asyncio.create_task(asyncio.to_thread(_search))
            if cancel_event:
                cancel = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [work, cancel], return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                if cancel in done:
                    return ToolResult(success=False, result="Search aborted")
            else:
                await work
            results = work.result()
        except Exception as e:
            return ToolResult(success=False, display=f"[red]Search failed: {e}[/red]")

        if not results:
            return ToolResult(
                success=True, result="No results found", display="[dim]No results found[/dim]"
            )

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '(no title)')}")
            lines.append(f"   {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')}")
            lines.append("")

        result_text = "\n".join(lines).strip()
        display = f"[dim]{len(results)} results[/dim]"
        return ToolResult(success=True, result=result_text, display=display)
