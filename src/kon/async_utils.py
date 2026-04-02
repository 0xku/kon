import asyncio
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")


class OperationCancelledError(Exception):
    pass


async def cancel_and_await(task: asyncio.Future[Any]) -> None:
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def await_or_cancel(work: asyncio.Future[T], cancel_event: asyncio.Event | None) -> T:
    if not cancel_event:
        return await work

    if cancel_event.is_set():
        await cancel_and_await(work)
        raise OperationCancelledError

    cancel = asyncio.create_task(cancel_event.wait())
    try:
        done, pending = await asyncio.wait({work, cancel}, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            await cancel_and_await(task)

        if cancel in done and cancel_event.is_set():
            if not work.done():
                await cancel_and_await(work)
            raise OperationCancelledError

        return work.result()
    finally:
        await cancel_and_await(cancel)
