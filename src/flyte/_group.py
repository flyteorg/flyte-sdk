from contextlib import contextmanager

from ._context import internal_ctx
from .models import GroupData


@contextmanager
def group(name: str):
    """
    Create a new group with the given name. The method is intended to be used as a context manager.

    Example:
    ```python
    @task
    async def my_task():
        ...
        with group("my_group"):
            t1(x,y)  # tasks in this block will be grouped under "my_group"
        ...
    ```

    :param name: The name of the group
    """
    ctx = internal_ctx()
    if ctx.data.task_context is None:
        yield
        return
    tctx = ctx.data.task_context
    new_tctx = tctx.replace(group_data=GroupData(name))
    with ctx.replace_task_context(new_tctx):
        yield
        # Exit the context and restore the previous context
