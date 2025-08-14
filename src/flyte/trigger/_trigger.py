from dataclasses import dataclass
from typing import Any, Dict

from ._schedule import Cron

AutomationType = Cron  # Type alias for automation, currently only supports Cron


@dataclass(frozen=True)
class Trigger:
    """
    This class defines specification of a Trigger, that can be associated with any Flyte V2 task.
    The trigger then is deployed to the Flyte Platform.

    Triggers can be used to run tasks on a schedule, in response to events, or based on other conditions.
    The `Trigger` class encapsulates the metadata and configuration needed to define a trigger.

    You can associate the same Trigger object with multiple tasks.

    Example usage:
    ```python
    from flyte.trigger import Trigger
    my_trigger = Trigger(
        name="my_trigger",
        description="A trigger that runs every hour",
    )
    ```
    """

    name: str
    automation: AutomationType
    description: str = ""
    inputs: Dict[str, Any] | None = None
    env: Dict[str, str] | None = None
    interruptable: bool | None = None


def new(
    name: str,
    automation: AutomationType,
    *,
    description: str = "",
    env: Dict[str, str] | None = None,
    interruptable: bool | None = None,
) -> Trigger:
    """
    Factory function to create a new Trigger instance with default values.

    :param name: (str) The name of the trigger.
    :param automation: (AutomationType) The automation type, currently only supports Cron.
    :param description: (str) A description of the trigger, default is an empty string.
    :param inputs: (Dict[str, Any]) Optional inputs for the trigger, default is None. If provided, will replace the
       values for inputs to these defaults.
    :param env: (Dict[str, str]) Optional environment variables for the trigger, default is None. If provided, will
        replace the environment variables set in the config of the task.
    :param interruptable: (bool) Whether the trigger is interruptable, default is None. If provided,
     it overrides whatever is set in the config of the task.

    :return Trigger: A new Trigger instance with default name and description.
    """
    return Trigger(name=name, automation=automation, description=description, env=env, interruptable=interruptable)
