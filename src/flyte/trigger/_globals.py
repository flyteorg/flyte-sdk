from ._defs import TriggerTime
from ._schedule import Cron
from ._trigger import Trigger


def daily(trigger_time_input_key: str = "trigger_time") -> Trigger:
    """
    Creates a Cron trigger that runs daily at midnight.

    Args:
        trigger_time_input_key (str): The input key for the trigger time, default is "trigger_time".

    Returns:
        Trigger: A trigger that runs daily at midnight.
    """
    return Trigger(
        name="daily",
        automation=Cron("0 0 * * *"),  # Cron expression for daily at midnight
        description="A trigger that runs daily at midnight",
        inputs={trigger_time_input_key: TriggerTime},
    )


def hourly(trigger_time_input_key: str = "trigger_time") -> Trigger:
    """
    Creates a Cron trigger that runs every hour.

    :param trigger_time_input_key: (str) The input parameter for the trigger time, default is "trigger_time".
    :return: Trigger A trigger that runs every hour, on the hour.
    """
    return Trigger(
        name="hourly",
        automation=Cron("0 * * * *"),  # Cron expression for every hour
        description="A trigger that runs every hour",
        inputs={trigger_time_input_key: TriggerTime},
    )


def every_minute(trigger_time_input_key: str = "trigger_time") -> Trigger:
    """
    Creates a Cron trigger that runs every minute.

    :param trigger_time_input_key: (str) The input parameter for the trigger time, default is "trigger_time".
    :return: Trigger A trigger that runs every minute.
    """
    return Trigger(
        name="every_minute",
        automation=Cron("* * * * *"),  # Cron expression for every minute
        description="A trigger that runs every minute",
        inputs={trigger_time_input_key: TriggerTime},
    )


def weekly(trigger_time_input_key: str = "trigger_time") -> Trigger:
    """
    Creates a Cron trigger that runs weekly on Sundays at midnight.

    :param trigger_time_input_key: (str) The input parameter for the trigger time, default is "trigger_time".
    :return: Trigger A trigger that runs weekly on Sundays at midnight.
    """
    return Trigger(
        name="weekly",
        automation=Cron("0 0 * * 0"),  # Cron expression for every Sunday at midnight
        description="A trigger that runs weekly on Sundays at midnight",
        inputs={trigger_time_input_key: TriggerTime},
    )
