from __future__ import annotations

import contextvars
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Dict, Iterator

from flyte._context import ctx

# Context variable to store the input context
# This stores both global context (from with_runcontext) and local context (from context managers)
_input_context_var: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar("input_context", default={})


def get_input_context() -> Dict[str, str]:
    """
    Get the current input context. This can be used within a task to retrieve
    context metadata that was passed to the action.

    Context will automatically propagate to sub-actions.

    Example:
    ```python
    import flyte

    env = flyte.TaskEnvironment(name="...")

    @env.task
    def t1():
        # context can be retrieved with `get_input_context`
        ctx = flyte.get_input_context()
        print(ctx)  # {'project': '...', 'entity': '...'}
    ```

    :return: Dictionary of context key-value pairs
    """
    # First check if we're in a task context and have input_context set there
    task_ctx = ctx()
    if task_ctx and task_ctx.input_context:
        return task_ctx.input_context.copy()

    # Otherwise, check the context variable (for context manager usage)
    return _input_context_var.get().copy()


@asynccontextmanager
async def input_context(**context: str) -> AsyncIterator[None]:
    """
    Async context manager to set input context for tasks spawned within this block.

    Example:
    ```python
    import flyte

    env = flyte.TaskEnvironment(name="...")

    @env.task
    async def t1():
        ctx = flyte.get_input_context()
        print(ctx)

    @env.task
    async def main():
        # context can be passed via a context manager
        async with flyte.input_context(project="my-project"):
            await t1()  # will have {'project': 'my-project'} as context
    ```

    :param context: Key-value pairs to set as input context
    """
    # Start with current context (includes global context if set)
    current = _input_context_var.get().copy()

    # Merge with code-provided context (code values override existing values)
    current.update(context)

    # Set the new context
    token = _input_context_var.set(current)
    try:
        yield
    finally:
        _input_context_var.reset(token)


@contextmanager
def input_context_sync(**context: str) -> Iterator[None]:
    """
    Synchronous context manager to set input context for tasks spawned within this block.

    Example:
    ```python
    import flyte

    env = flyte.TaskEnvironment(name="...")

    @env.task
    def t1():
        ctx = flyte.get_input_context()
        print(ctx)

    @env.task
    def main():
        # context can be passed via a context manager
        with flyte.input_context_sync(project="my-project"):
            t1()  # will have {'project': 'my-project'} as context
    ```

    :param context: Key-value pairs to set as input context
    """
    # Start with current context (includes global context if set)
    current = _input_context_var.get().copy()

    # Merge with code-provided context (code values override existing values)
    current.update(context)

    # Set the new context
    token = _input_context_var.set(current)
    try:
        yield
    finally:
        _input_context_var.reset(token)
