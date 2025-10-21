import typing
from dataclasses import dataclass
from typing import Generic, Literal, Type

import rich.repr

from flyte.syncify import syncify

EventScope = Literal["task", "run", "action"]

EventType = typing.TypeVar("EventType", bool, int, float, str)


@rich.repr.auto
@dataclass
class _Event(Generic[EventType]):
    """
    An event that can be awaited in a Run. Events can be used to pause Run until an external signal is received.

    Examples:

    ```python
    import flyte

    env = flyte.TaskEnvironment(name="events")

    @env.task
    async def my_task() -> Optional[int]:
        event = await flyte.new_event(name="my_event", scope="run", prompt="Is it ok to continue?", data_type=bool)
        result = await event.wait()
        if result:
            return 42
        else:
            return None
    ```
    """

    name: str
    # TODO restrict scope to action only right now
    scope: EventScope = "run"
    # TODO Support prompt as html
    prompt: str = "Approve?"
    data_type: Type[EventType] = bool  # type: ignore[assignment]
    description: str = ""

    def __post_init__(self):
        valid_types = (bool, int, float, str)
        if self.data_type not in valid_types:
            raise TypeError(f"Invalid data_type {self.data_type}. Must be one of {valid_types}.")

    @syncify
    async def wait(self) -> EventType:
        """
        Await the event to be signaled.

        :return: The payload associated with the event when it is signaled.
        """
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        if ctx.is_task_context():
            # If we are in a task context, that implies we are executing a Run.
            # In this scenario, we should submit the task to the controller.
            # We will also check if we are not initialized, It is not expected to be not initialized
            from ._internal.controllers import get_controller

            controller = get_controller()
            result = await controller.wait_for_event(self)
            return result
        else:
            raise RuntimeError("Events can only be awaited within a task context.")


@syncify
async def new_event(
    name: str,
    /,
    scope: EventScope = "run",
    prompt: str = "Approve?",
    data_type: Type[EventType] = bool,  # type: ignore[assignment]
    description: str = "",
) -> _Event:
    """
    Create an event that can be awaited in a workflow. Events can be used to pause workflow execution until
    an external signal is received.

    :param name: Name of the event
    :param scope: Scope of the event - "task", "run", or "action"
    :param prompt: Prompt message for the event
    :param data_type: Data type of the event payload
    :param description: Description of the event
    :return: An instance of _Event representing the created event
    """
    event = _Event(name=name, scope=scope, prompt=prompt, data_type=data_type, description=description)
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    if ctx.is_task_context():
        # If we are in a task context, that implies we are executing a Run.
        # In this scenario, we should submit the task to the controller.
        # We will also check if we are not initialized, It is not expected to be not initialized
        from ._internal.controllers import get_controller

        controller = get_controller()
        await controller.register_event(event)
    else:
        pass
    return event
