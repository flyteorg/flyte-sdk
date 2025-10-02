from __future__ import annotations

import contextvars
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, ParamSpec, TypeVar

from flyte._logging import logger
from flyte.models import GroupData, RawDataPath, TaskContext

if TYPE_CHECKING:
    from flyte.report import Report

P = ParamSpec("P")  # capture the function's parameters
R = TypeVar("R")  # return type


@dataclass(frozen=True, kw_only=True)
class ContextData:
    """
    A ContextData cannot be created without an execution. Even for local execution's there should be an execution ID

    :param: action The action ID of the current execution. This is always set, within a run.
    :param: group_data If nested in a group the current group information
    :param: task_context The context of the current task execution, this is what is available to the user, it is set
        when the task is executed through `run` methods. If the Task is executed as regular python methods, this
        will be None.
    """

    group_data: Optional[GroupData] = None
    task_context: Optional[TaskContext] = None
    raw_data_path: Optional[RawDataPath] = None

    def replace(self, **kwargs) -> ContextData:
        return replace(self, **kwargs)


class Context:
    """
    A context class to hold the current execution context.
    This is not coroutine safe, it assumes that the context is set in a single thread.
    You should use the `contextual_run` function to run a function in a new context tree.

    A context tree is defined as a tree of contexts, where under the root, all coroutines that were started in
    this context tree can access the context mutations, but no coroutine, created outside of the context tree can access
    the context mutations.
    """

    def __init__(self, data: ContextData):
        if data is None:
            raise ValueError("Cannot create a new context without contextdata.")
        self._data = data
        self._id = id(self)  # Immutable unique identifier
        self._token = None  # Context variable token to restore the previous context

    @property
    def data(self) -> ContextData:
        """Viewable data."""
        return self._data

    @property
    def raw_data(self) -> RawDataPath:
        """
        Get the raw data prefix for the current context first by looking up the task context, then the raw data path
        """
        if self.data and self.data.task_context and self.data.task_context.raw_data_path:
            return self.data.task_context.raw_data_path
        if self.data and self.data.raw_data_path:
            return self.data.raw_data_path
        raise ValueError("Raw data path has not been set in the context.")

    @property
    def id(self) -> int:
        """Viewable ID."""
        return self._id

    def replace_task_context(self, tctx: TaskContext) -> Context:
        """
        Replace the task context in the current context.
        """
        return Context(self.data.replace(task_context=tctx))

    def new_raw_data_path(self, raw_data_path: RawDataPath) -> Context:
        """
        Return a copy of the context with the given raw data path object
        """
        return Context(self.data.replace(raw_data_path=raw_data_path))

    def get_report(self) -> Optional[Report]:
        """
        Returns a report if within a task context, else a None
        :return:
        """
        if self.data.task_context:
            return self.data.task_context.report
        return None

    def is_task_context(self) -> bool:
        """
        Returns true if the context is a task context
        :return: bool
        """
        return self.data.task_context is not None

    def __enter__(self):
        """Enter the context, setting it as the current context."""
        self._token = root_context_var.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring the previous context."""
        try:
            root_context_var.reset(self._token)
        except Exception as e:
            logger.warn(f"Failed to reset context: {e}")
            raise e

    async def __aenter__(self):
        """Async version of context entry."""
        self._token = root_context_var.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async version of context exit."""
        root_context_var.reset(self._token)

    def __repr__(self):
        return f"{self.data}"

    def __str__(self):
        return self.__repr__()


# Global context variable to hold the current context
root_context_var = contextvars.ContextVar("root", default=Context(data=ContextData()))


def ctx() -> Optional[TaskContext]:
    """
    Returns flyte.models.TaskContext if within a task context, else None
    Note: Only use this in task code and not module level.
    """
    return internal_ctx().data.task_context


def internal_ctx() -> Context:
    """Retrieve the current context from the context variable."""
    return root_context_var.get()


async def contextual_run(func: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs) -> R:
    """
    Run a function with a new context subtree.
    """
    _ctx = contextvars.copy_context()
    return await _ctx.run(func, *args, **kwargs)
