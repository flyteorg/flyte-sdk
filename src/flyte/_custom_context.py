from __future__ import annotations

from contextlib import contextmanager

from flyte._context import ctx

from ._context import internal_ctx


def get_custom_context() -> dict[str, str]:
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
        # context can be retrieved with `get_custom_context`
        ctx = flyte.get_custom_context()
        print(ctx)  # {'project': '...', 'entity': '...'}
    ```

    :return: Dictionary of context key-value pairs
    """
    tctx = ctx()
    if tctx is None or tctx.custom_context is None:
        return {}
    return tctx.custom_context


@contextmanager
def custom_context(**context: str):
    """
    Synchronous context manager to set input context for tasks spawned within this block.

    Example:
    ```python
    import flyte

    env = flyte.TaskEnvironment(name="...")

    @env.task
    def t1():
        ctx = flyte.get_custom_context()
        print(ctx)

    @env.task
    def main():
        # context can be passed via a context manager
        with flyte.custom_context(project="my-project"):
            t1()  # will have {'project': 'my-project'} as context
    ```

    :param context: Key-value pairs to set as input context
    """
    ctx = internal_ctx()
    if ctx.data.task_context is None:
        yield
        return

    tctx = ctx.data.task_context
    new_tctx = tctx.replace(custom_context={**tctx.custom_context, **context})

    with ctx.replace_task_context(new_tctx):
        yield
        # Exit the context and restore the previous context
