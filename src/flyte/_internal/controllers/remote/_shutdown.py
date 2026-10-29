"""Cleanup for event loops owned exclusively by controller threads."""

import asyncio


def drain_and_close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run cancellation finalizers before releasing an owned event loop."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            results = loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            for task, result in zip(pending, results):
                if isinstance(result, Exception):
                    loop.call_exception_handler(
                        {"message": "Error during controller shutdown", "exception": result, "task": task}
                    )
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()
