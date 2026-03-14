import asyncio
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..core.types import ToolResult


class BaseTool[T: BaseModel](ABC):
    name: str
    params: type[T]
    description: str
    mutating: bool = True

    @abstractmethod
    async def execute(
        self, params: T, cancel_event: asyncio.Event | None = None
    ) -> ToolResult: ...

    def format_call(self, params: T) -> str:
        data = params.model_dump(exclude_none=True)
        if not data:
            return ""
        parts = [f"{k}={v}" for k, v in data.items()]
        return " / ".join(parts)
