from contextlib import contextmanager
from typing import Tuple


@contextmanager
def auth_metadata(*kv: Tuple[str, str]):
    """
    This context manager allows you to pass contextualized auth metadata downstream to the Flyte authentication system.

    This is only useful if flyte.init_passthrough() has been called.

    Example:
    ```python

    flyte.init_passthrough("my-endpoint")

    ...

    with auth_metadata((key1, value1), (key2, value2)):
        ...
    ```

    Args:
        *kv: Tuple of auth metadata key/value pairs.
    """
    if not kv:
        raise ValueError("No auth metadata provided.")
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    with ctx.new_metadata(kv):
        yield


def get_auth_metadata() -> Tuple[Tuple[str, str], ...]:
    """
    Returns auth metadata as tuple.
    """
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    return ctx.data.metadata or ()
