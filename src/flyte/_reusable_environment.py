from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple, Union

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
    :param idle_ttl: The maximum idle duration for an environment replica, specified as either seconds (int) or a
        timedelta. If not set, the environment's global default will be used.
        When a replica remains idle — meaning no tasks are running — for this duration, it will be automatically
        terminated.
    :param concurrency: The maximum number of tasks that can run concurrently in one instance of the environment.
          Concurrency of greater than 1 is only supported only for `async` tasks.
    :param reuse_salt: Optional string used to control environment reuse.
        If set, the environment will be reused even if the code bundle changes.
        To force a new environment, either set this to `None` or change its value.

        Example:
            reuse_salt = "v1"  # Environment is reused
            reuse_salt = "v2"  # Forces environment recreation
    """

    replicas: Union[int, Tuple[int, int]] = 2
    idle_ttl: Optional[Union[int, timedelta]] = None
    reuse_salt: str | None = None
    concurrency: int = 1

    def __post_init__(self):
        if self.replicas is None:
            raise ValueError("replicas cannot be None")
        if isinstance(self.replicas, int):
            self.replicas = (self.replicas, self.replicas)
        elif not isinstance(self.replicas, tuple):
            raise ValueError("replicas must be an int or a tuple of two ints")
        elif len(self.replicas) != 2:
            raise ValueError("replicas must be an int or a tuple of two ints")

        if self.idle_ttl:
            if isinstance(self.idle_ttl, int):
                self.idle_ttl = timedelta(seconds=int(self.idle_ttl))
            elif not isinstance(self.idle_ttl, timedelta):
                raise ValueError("idle_ttl must be an int (seconds) or a timedelta")

        if self.replicas[1] == 1 and self.concurrency == 1:
            logger.warning(
                "It is recommended to use a minimum of 2 replicas, to avoid starvation. "
                "Starvation can occur if a task is running and no other replicas are available to handle new tasks."
                "Options, increase concurrency, increase replicas or turn-off reuse for the parent task, "
                "that runs child tasks."
            )

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
    def max_replicas(self) -> int:
        """
        Returns the maximum number of replicas.
        """
        return self.replicas[1] if isinstance(self.replicas, tuple) else self.replicas
