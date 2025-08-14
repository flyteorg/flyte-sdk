from __future__ import annotations

from dataclasses import dataclass

import rich.repr


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


cron_daily = Cron("0 0 * * *")
cron_hourly = Cron("0 * * * *")  # Cron expression for every
cron_minute = Cron("* * * * *")  # Cron expression for every minute
cron_weekly = Cron("0 0 * * 0")  # Cron expression for every Sunday at midnight
