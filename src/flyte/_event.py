import typing
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generic, Literal, Optional, Type, Union

import rich.repr

from flyte.syncify import syncify

PromptType = Literal["text", "markdown"]

EventType = typing.TypeVar("EventType", bool, int, float, str)


@dataclass
class EventWebhook:
    """Webhook configuration for an event notification.

    When specified, the backend will POST to the given URL when the event is created.
    The ``payload`` dict may contain the template variable ``{callback_uri}`` in any
    string value — the backend replaces it with the actual URI that can be used to
    signal the event.

    Example::

        EventWebhook(
            url="https://example.com/hook",
            payload={"callback": "{callback_uri}", "event": "approval_needed"},
        )
    """

    url: str
    payload: Optional[Dict[str, Any]] = None


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
        event = await flyte.new_event(name="my_event", prompt="Is it ok to continue?", data_type=bool)
        result = await event.wait()
        if result:
            return 42
        else:
            return None
    ```
    """

    name: str
    prompt: str = "Approve?"
    prompt_type: PromptType = "text"
    data_type: Type[EventType] = bool  # type: ignore[assignment]
    description: str = ""
    timeout: Union[timedelta, int, float, None] = None
    webhook: Optional[EventWebhook] = None

    def __post_init__(self):
        valid_types = (bool, int, float, str)
        if self.data_type not in valid_types:
            raise TypeError(f"Invalid data_type {self.data_type}. Must be one of {valid_types}.")
        if self.timeout is not None:
            if isinstance(self.timeout, timedelta):
                self._timeout_seconds = self.timeout.total_seconds()
            else:
                self._timeout_seconds = float(self.timeout)
            if self._timeout_seconds <= 0:
                raise ValueError("timeout must be positive")
        else:
            self._timeout_seconds = None

    @syncify
    async def wait(self) -> EventType:
        """
        Await the event to be signaled.

        Blocks until the condition is resolved by the backend. The return value is
        converted to the ``data_type`` specified at creation time:

        - ``bool`` → ``True`` / ``False``
        - ``int`` → Python ``int``
        - ``float`` → Python ``float``
        - ``str`` → Python ``str``

        **Protocol:** When running remotely, the event is backed by a *condition action*.
        The backend delivers the result as a ``Literal`` (protobuf scalar/primitive)
        directly in the ``ActionUpdate`` — no ``output_uri`` is used for conditions.

        :return: The typed payload associated with the event when it is signaled.
        :raises flyte.errors.EventTimedoutError: If the event is not signaled within the
            specified ``timeout``.
        :raises flyte.errors.EventFailedError: If the condition fails during execution.
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
    prompt: str = "Approve?",
    prompt_type: PromptType = "text",
    data_type: Type[EventType] = bool,  # type: ignore[assignment]
    description: str = "",
    timeout: Union[timedelta, int, float, None] = None,
    webhook: Optional[EventWebhook] = None,
) -> _Event:
    """
    Create an event that can be awaited in a workflow. Events can be used to pause workflow execution until
    an external signal is received.

    **Condition protocol (remote execution):**

    When running inside a task, ``new_event`` registers a *condition action* with the
    backend. Calling ``event.wait()`` blocks until the condition is resolved. The backend
    delivers the result as an inline ``Literal`` (protobuf scalar/primitive) in the
    ``ActionUpdate`` stream — no ``output_uri`` is involved for conditions.

    - On success, ``wait()`` returns the value converted to ``data_type``
      (``True``/``False`` for bool, Python ``int``/``float``/``str`` for the others).
    - If the condition times out, ``wait()`` raises ``flyte.errors.EventTimedoutError``.
    - If the condition fails, ``wait()`` raises ``flyte.errors.EventFailedError``.

    :param name: Name of the event
    :param prompt: Prompt message for the event
    :param data_type: Data type of the event payload — one of ``bool``, ``int``, ``float``, ``str``
    :param prompt_type: Type of prompt rendering - "text" or "markdown"
    :param description: Description of the event
    :param timeout: Optional timeout as a timedelta or number of seconds. If the event is not signaled
        within this duration, ``wait()`` will raise ``flyte.errors.EventTimedoutError``.
    :param webhook: Optional webhook configuration. When provided, the backend will POST to the
        given URL with the specified payload. The payload may use ``{callback_uri}`` as a template
        variable — the backend replaces it with the URI that can be used to signal the event.
    :return: An instance of _Event representing the created event
    """
    event = _Event(
        name=name,
        prompt=prompt,
        prompt_type=prompt_type,
        data_type=data_type,
        description=description,
        timeout=timeout,
        webhook=webhook,
    )
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
