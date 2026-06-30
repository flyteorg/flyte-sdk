from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, Union

from flyte._logging import logger


@dataclass
class ReusePolicy:
    """
    Configure a task environment for container reuse across multiple task invocations.

    When environment creation is expensive relative to task runtime, reusable containers
    keep a pool of warm containers ready, avoiding cold-start overhead. The Python process
    may be reused by subsequent task invocations.

    Total concurrent capacity is `max_replicas * concurrency`. For example,
    `ReusePolicy(replicas=(1, 3), concurrency=2)` supports up to 6 concurrent tasks.

    Caution: The environment is shared across invocations — manage memory and resources carefully.

    Example:

    ```python
    env = flyte.TaskEnvironment(
        name="fast_env",
        reusable=flyte.ReusePolicy(replicas=(1, 3), concurrency=2),
    )
    ```

    :param replicas: Number of container replicas to maintain.

        - `int`: Fixed replica count, always running (e.g., `replicas=3`).
        - `tuple(min, max)`: Auto-scaling range (e.g., `replicas=(1, 5)`).
          Scales between min and max based on demand.

        Default is `2`. A minimum of 2 replicas is recommended to avoid starvation
        when the parent task occupies one replica.

    :param idle_ttl: Environment-level idle timeout — shuts down **all** replicas when the
        entire environment has been idle for this duration. Specified as seconds (`int`)
        or `timedelta`. Minimum 30 seconds. Default is 30 seconds.
    :param concurrency: Maximum concurrent tasks per replica. Values greater than 1 are
        only supported for `async` tasks. Default is `1`.
    :param scaledown_ttl: Per-replica scale-down delay — minimum time to wait before
        removing an **individual** idle replica. Prevents rapid scale-down when tasks
        arrive in bursts. Specified as seconds (`int`) or `timedelta`. Default is
        30 seconds.

        Note the distinction: `idle_ttl` controls when the whole environment shuts down;
        `scaledown_ttl` controls when individual replicas are removed during auto-scaling.
    """

    replicas: Union[int, Tuple[int, int]] = 2
    idle_ttl: Union[int, timedelta] = 30  # seconds
    concurrency: int = 1
    scaledown_ttl: Union[int, timedelta] = 30  # seconds

    def __post_init__(self):
        if self.replicas is None:
            raise ValueError("replicas cannot be None")
        if isinstance(self.replicas, int):
            self.replicas = (self.replicas, self.replicas)
        elif not isinstance(self.replicas, tuple):
            raise ValueError("replicas must be an int or a tuple of two ints")
        elif len(self.replicas) != 2:
            raise ValueError("replicas must be an int or a tuple of two ints")

        if isinstance(self.idle_ttl, int):
            self.idle_ttl = timedelta(seconds=int(self.idle_ttl))
        elif not isinstance(self.idle_ttl, timedelta):
            raise ValueError("idle_ttl must be an int (seconds) or a timedelta")
        if self.idle_ttl.total_seconds() < 30:
            raise ValueError("idle_ttl must be at least 30 seconds")

        if self.replicas[1] == 1 and self.concurrency == 1:
            logger.warning(
                "It is recommended to use a minimum of 2 replicas, to avoid starvation. "
                "Starvation can occur if a task is running and no other replicas are available to handle new tasks."
                "Options, increase concurrency, increase replicas or turn-off reuse for the parent task, "
                "that runs child tasks."
            )

        if isinstance(self.scaledown_ttl, int):
            self.scaledown_ttl = timedelta(seconds=int(self.scaledown_ttl))
        elif not isinstance(self.scaledown_ttl, timedelta):
            raise ValueError("scaledown_ttl must be an int (seconds) or a timedelta")
        if self.scaledown_ttl.total_seconds() < 30:
            raise ValueError("scaledown_ttl must be at least 30 seconds")

    @property
    def min_replicas(self) -> int:
        """
        Returns the minimum number of replicas.
        """
        return self.replicas[0] if isinstance(self.replicas, tuple) else self.replicas

    def get_scaledown_ttl(self) -> timedelta | None:
        """
        Returns the scaledown TTL as a timedelta. If scaledown_ttl is not set, returns None.
        """
        if self.scaledown_ttl is None:
            return None
        if isinstance(self.scaledown_ttl, timedelta):
            return self.scaledown_ttl
        return timedelta(seconds=int(self.scaledown_ttl))

    @property
    def max_replicas(self) -> int:
        """
        Returns the maximum number of replicas.
        """
        return self.replicas[1] if isinstance(self.replicas, tuple) else self.replicas
