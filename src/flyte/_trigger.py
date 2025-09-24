from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Union

import rich.repr


class _trigger_time:
    """
    This class represents the actual time of Trigger, which can be bound to any task input.
    """


TriggerTime = _trigger_time()


@rich.repr.auto
@dataclass(frozen=True)
class Cron:
    """
    This class defines a Cron automation that can be associated with a Trigger in Flyte.
    Example usage:
    ```python
    from flyte.trigger import Trigger, Cron
    my_trigger = Trigger(
        name="my_cron_trigger",
        automation=Cron("0 * * * *"),  # Runs every hour
        description="A trigger that runs every hour",
    )
    ```
    """

    expression: str

    def __str__(self):
        return f"Cron Trigger: {self.expression}"


@rich.repr.auto
@dataclass(frozen=True)
class FixedRate:
    """
    This class defines a FixedRate automation that can be associated with a Trigger in Flyte.
    Example usage:
    ```python
    from flyte.trigger import Trigger, FixedRate
    my_trigger = Trigger(
        name="my_fixed_rate_trigger",
        automation=FixedRate(60),  # Runs every hour
        description="A trigger that runs every hour",
    )
    ```
    """

    interval_minutes: int
    start_time: datetime | None = None

    def __str__(self):
        return f"FixedRate Trigger: every {self.interval_minutes} minutes"


@rich.repr.auto
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

    :param name: (str) The name of the trigger.
    :param automation: (AutomationType) The automation type, currently only supports Cron.
    :param description: (str) A description of the trigger, default is an empty string.
    :param auto_activate: (bool) Whether the trigger should be automatically activated, default is True.
    :param inputs: (Dict[str, Any]) Optional inputs for the trigger, default is None. If provided, will replace the
       values for inputs to these defaults.
    :param env_vars: (Dict[str, str]) Optional environment variables for the trigger, default is None. If provided, will
        replace the environment variables set in the config of the task.
    :param interruptible: (bool) Whether the trigger is interruptible, default is None. If provided,
     it overrides whatever is set in the config of the task.
    :param overwrite_cache: (bool) Whether to overwrite the cache, default is False.
    :param queue: (str) Optional queue to run the trigger in, default is None.
    :param labels: (Mapping[str, str]) Optional labels to attach to the trigger, default is None.
    :param annotations: (Mapping[str, str]) Optional annotations to attach to the trigger, default is None.
    """

    name: str
    automation: Union[Cron, FixedRate]
    description: str = ""
    auto_activate: bool = True
    inputs: Dict[str, Any] | None = None
    env_vars: Dict[str, str] | None = None
    interruptible: bool | None = None
    overwrite_cache: bool = False
    queue: str | None = None
    labels: Mapping[str, str] | None = None
    annotations: Mapping[str, str] | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Trigger name cannot be empty")
        if self.automation is None:
            raise ValueError("Automation cannot be None")

    @classmethod
    def daily(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "daily",
        description: str = "A trigger that runs daily at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs daily at midnight.

        Args:
            trigger_time_input_key (str): The input key for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "daily".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs daily at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 * * *"),  # Cron expression for daily at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def hourly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "hourly",
        description: str = "A trigger that runs every hour",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs every hour.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "hourly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs every hour, on the hour.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 * * * *"),  # Cron expression for every hour
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def minutely(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "minutely",
        description: str = "A trigger that runs every minute",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs every minute.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "every_minute".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs every minute.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("* * * * *"),  # Cron expression for every minute
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def weekly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "weekly",
        description: str = "A trigger that runs weekly on Sundays at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs weekly on Sundays at midnight.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "weekly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs weekly on Sundays at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 * * 0"),  # Cron expression for every Sunday at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )

    @classmethod
    def monthly(
        cls,
        trigger_time_input_key: str = "trigger_time",
        *,
        name: str = "monthly",
        description: str = "A trigger that runs monthly on the 1st at midnight",
        auto_activate: bool = True,
        inputs: Dict[str, Any] | None = None,
        env_vars: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        overwrite_cache: bool = False,
        queue: str | None = None,
        labels: Mapping[str, str] | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> Trigger:
        """
        Creates a Cron trigger that runs monthly on the 1st at midnight.

        Args:
            trigger_time_input_key (str): The input parameter for the trigger time, default is "trigger_time".
            name (str): The name of the trigger, default is "monthly".
            description (str): A description of the trigger.
            auto_activate (bool): Whether the trigger should be automatically activated.
            inputs (Dict[str, Any] | None): Optional inputs for the trigger.
            env_vars (Dict[str, str] | None): Optional environment variables.
            interruptible (bool | None): Whether the trigger is interruptible.
            overwrite_cache (bool): Whether to overwrite the cache.
            queue (str | None): Optional queue to run the trigger in.
            labels (Mapping[str, str] | None): Optional labels to attach to the trigger.
            annotations (Mapping[str, str] | None): Optional annotations to attach to the trigger.

        Returns:
            Trigger: A trigger that runs monthly on the 1st at midnight.
        """
        final_inputs = {trigger_time_input_key: TriggerTime}
        if inputs:
            final_inputs.update(inputs)

        return cls(
            name=name,
            automation=Cron("0 0 1 * *"),  # Cron expression for monthly on the 1st at midnight
            description=description,
            auto_activate=auto_activate,
            inputs=final_inputs,
            env_vars=env_vars,
            interruptible=interruptible,
            overwrite_cache=overwrite_cache,
            queue=queue,
            labels=labels,
            annotations=annotations,
        )
