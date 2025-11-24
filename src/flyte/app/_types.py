from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple, Union

import rich.repr

INVALID_APP_PORTS = [8012, 8022, 8112, 9090, 9091]


@rich.repr.auto
@dataclass(frozen=True)
class Port:
    port: int
    name: Optional[str] = None

    def __post_init__(self):
        if self.port in INVALID_APP_PORTS:
            invalid_ports = ", ".join(str(p) for p in INVALID_APP_PORTS)
            msg = f"port {self.port} is not allowed. Please do not use ports: {invalid_ports}"
            raise ValueError(msg)


@rich.repr.auto
@dataclass(frozen=True)
class Link:
    path: str
    title: str
    is_relative: bool = False


@rich.repr.auto
@dataclass
class Scaling:
    @dataclass(frozen=True)
    class Concurrency:
        """
        Use this to specify the concurrency metric for autoscaling, i.e. the number of concurrent requests at a replica
         at which to scale up.
        """

        val: int

        def __post_init__(self):
            if self.val < 1:
                raise ValueError("Concurrency must be greater than or equal to 1")

    @dataclass
    class RequestRate:
        """
        Use this to specify the request rate metric for autoscaling, i.e. the number of requests per second at a replica
         at which to scale up.
        """

        val: int

        def __post_init__(self):
            if self.val < 1:
                raise ValueError("Request rate must be greater than or equal to 1")

    replicas: Union[int, Tuple[int, int]] = (1, 1)
    metric: Optional[Union[Concurrency, RequestRate]] = None
    scaledown_after: int | timedelta | None = None

    def __post_init__(self):
        if isinstance(self.replicas, int):
            if self.replicas < 0:
                raise ValueError("replicas must be greater than or equal to 0")
            self.replicas = (self.replicas, self.replicas)
        elif isinstance(self.replicas, tuple):
            if len(self.replicas) != 2:
                raise ValueError("replicas tuple must be of length 2")
            min_replicas, max_replicas = self.replicas
            if min_replicas < 0:
                raise ValueError("min_replicas must be greater than or equal to 0")
            if max_replicas < 1 or max_replicas < min_replicas:
                raise ValueError("max_replicas must be greater than or equal to 1 and min_replicas")
        else:
            raise TypeError("replicas must be an int or a tuple of two ints")

        if self.metric:
            if not isinstance(self.metric, (Scaling.Concurrency, Scaling.RequestRate)):
                raise TypeError("metric must be an instance of Scaling.Concurrency or Scaling.RequestRate")

        if self.scaledown_after:
            if isinstance(self.scaledown_after, int):
                self.scaledown_after = timedelta(seconds=self.scaledown_after)
            elif not isinstance(self.scaledown_after, timedelta):
                raise TypeError("scaledown_after must be an int or a timedelta")

    def get_replicas(self) -> Tuple[int, int]:
        if isinstance(self.replicas, int):
            return self.replicas, self.replicas
        return self.replicas


@rich.repr.auto
@dataclass
class Domain:
    # SubDomain config
    subdomain: Optional[str] = None
    custom_domain: Optional[str] = None
