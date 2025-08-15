from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, Union

from flyte._logging import logger


@dataclass
class ReusePolicy:
    """
    ReusePolicy can be used to configure a task to reuse the environment. This is useful when the environment creation
    is expensive and the runtime of the task is short. The environment will be reused for the next invocation of the
    task, even the python process maybe be reused by subsequent task invocations. A good mental model is to think of
    the environment as a container that is reused for multiple tasks, more like a long-running service.

    Caution: It is important to note that the environment is shared, so managing memory and resources is important.

    :param replicas: Either a single int representing number of replicas or a tuple of two ints representing
     the min and max.
    :param idle_ttl: The maximum idle duration for an environment, specified as either seconds (int) or a
        timedelta, after which all replicas in the environment are shutdown.
        If not set, the default is configured in the backend (can be as low as 90s).
        When a replica remains idle — meaning no tasks are running — for this duration, it will be automatically
        terminated, also referred to as environment idle timeout.
    :param concurrency: The maximum number of tasks that can run concurrently in one instance of the environment.
          Concurrency of greater than 1 is only supported for `async` tasks.
    :param scaledown_ttl: The minimum time to wait before scaling down each replica, specified as either seconds (int)
        or a timedelta. This is useful to prevent rapid scaling down of replicas when tasks are running
        frequently. If not set, the default is configured in the backend.
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
    def ttl(self) -> timedelta | None:
        """
        Returns the idle TTL as a timedelta. If idle_ttl is not set, returns the global default.
        """
        if self.idle_ttl is None:
            return None
        if isinstance(self.idle_ttl, timedelta):
            return self.idle_ttl
        return timedelta(seconds=self.idle_ttl)

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
