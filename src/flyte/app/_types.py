import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from flyte.app._app_environment import AppEnvironment
    from flyte.models import SerializationContext

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
    """Custom links to add to the app"""

    path: str
    title: str
    is_relative: bool = False


@rich.repr.auto
@dataclass
class Scaling:
    """
    Controls replica count and autoscaling behavior for app environments.

    Common scaling patterns:

    - **Scale-to-zero** (default): `Scaling(replicas=(0, 1))` — no replicas when idle,
      scales to 1 on demand.
    - **Always-on**: `Scaling(replicas=(1, 1))` — exactly 1 replica at all times.
    - **Burstable**: `Scaling(replicas=(1, 5))` — 1 replica minimum, scales up to 5.
    - **High-availability**: `Scaling(replicas=(2, 10))` — at least 2 replicas always running.
    - **Fixed size**: `Scaling(replicas=3)` — exactly 3 replicas.

    Args:
        replicas: Number of replicas. An `int` for fixed count, or a `(min, max)`
            tuple for autoscaling. Default `(0, 1)`.
        metric: Autoscaling metric — `Scaling.Concurrency(val)` (scale when concurrent
            requests per replica exceeds `val`) or `Scaling.RequestRate(val)` (scale when
            requests per second per replica exceeds `val`). Default `None`.
        scaledown_after: Time to wait after the last request before scaling down.
            Seconds (`int`) or `timedelta`. Default `None` (platform default).
    """

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

    """Number of replicas to run. Can be a single int or a tuple of two ints representing the min and max replicas."""
    replicas: Union[int, Tuple[int, int]] = (0, 1)

    """Metric to use for autoscaling. Can be a concurrency or request rate."""
    metric: Optional[Union[Concurrency, RequestRate]] = None

    """Time to wait after the last request before scaling down. Can be a number of seconds or a timedelta."""
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


_MAX_REQUEST_TIMEOUT = timedelta(hours=1)


@rich.repr.auto
@dataclass
class Timeouts:
    """Timeout configuration for the application.

    Attributes:
        request: Timeout for requests to the application. Can be an int
            (seconds) or timedelta. Must not exceed 1 hour.
    """

    request: int | timedelta | None = None

    def __post_init__(self):
        if self.request is None:
            return
        if isinstance(self.request, int):
            self.request = timedelta(seconds=self.request)
        elif not isinstance(self.request, timedelta):
            raise TypeError(f"Expected request to be of type int or timedelta, got {type(self.request)}")
        if self.request < timedelta(0):
            raise ValueError("request timeout must be non-negative")
        if self.request > _MAX_REQUEST_TIMEOUT:
            raise ValueError("request timeout must not exceed 1 hour (3600 seconds)")


_PROJECT_DOMAIN_HASH_LEN = 8


@rich.repr.auto
@dataclass(frozen=True)
class Subdomain:
    """
    A subdomain that is resolved at deploy time, when the deployment project and domain are known.

    Use `Subdomain.from_app_name` for the built-in naming schemes:

    - `project_domain_suffix="hash"`: the subdomain is `{app_name}-{hash}`, where the hash is computed
      from `{project}-{domain}`. This keeps subdomains short and stable per project/domain.
    - `project_domain_suffix="default"`: the subdomain is `{app_name}-{project}-{domain}`.

    Use `Subdomain.from_function` for full control: the function receives the `AppEnvironment` and the
    deployment `SerializationContext` (project, domain, org, version, ...) and returns the subdomain.

    The final subdomain string is produced by `resolve()` during serialization.
    """

    app_name: Optional[str] = None
    project_domain_suffix: Literal["hash", "default"] = "hash"
    function: Optional[Callable[["AppEnvironment", "SerializationContext"], str]] = None

    def __post_init__(self):
        if (self.app_name is None) == (self.function is None):
            raise ValueError("exactly one of app_name or function must be set")
        if self.project_domain_suffix not in ("hash", "default"):
            raise ValueError(f"project_domain_suffix must be 'hash' or 'default', got {self.project_domain_suffix!r}")

    @classmethod
    def from_app_name(cls, app_name: str, project_domain_suffix: Literal["hash", "default"] = "hash") -> "Subdomain":
        """
        Create a subdomain for an app whose final value depends on the deployment project and domain.

        Args:
            app_name: Name of the app.
            project_domain_suffix: `"hash"` for `{app_name}-{hash-of-project-domain}`, or `"default"`
                for `{app_name}-{project}-{domain}`.
        """
        return cls(app_name=app_name, project_domain_suffix=project_domain_suffix)

    @classmethod
    def from_function(cls, function: Callable[["AppEnvironment", "SerializationContext"], str]) -> "Subdomain":
        """
        Create a subdomain computed by a user-provided function at deploy time.

        Args:
            function: Called with the `AppEnvironment` being deployed and the deployment
                `SerializationContext`; returns the subdomain string.
        """
        return cls(function=function)

    def resolve(self, app_env: "AppEnvironment", serialization_context: "SerializationContext") -> str:
        """Resolve to the final subdomain string for the given app environment and deployment context."""
        if self.function is not None:
            subdomain = self.function(app_env, serialization_context)
            if not isinstance(subdomain, str) or not subdomain:
                raise ValueError(
                    f"subdomain function for app {app_env.name!r} must return a non-empty str, got {subdomain!r}"
                )
            return subdomain

        project, domain = serialization_context.project, serialization_context.domain
        if not project or not domain:
            raise ValueError(
                f"project and domain are required to resolve subdomain for app {self.app_name!r}, "
                f"got project={project!r}, domain={domain!r}"
            )
        if self.project_domain_suffix == "hash":
            suffix = hashlib.sha256(f"{project}-{domain}".encode()).hexdigest()[:_PROJECT_DOMAIN_HASH_LEN]
            return f"{self.app_name}-{suffix}"
        return f"{self.app_name}-{project}-{domain}"


@rich.repr.auto
@dataclass
class Domain:
    # SubDomain config

    """Subdomain to use for the domain. Either a literal string, or a `Subdomain` resolved against the
    deployment project and domain. If not set, the default subdomain will be used."""

    subdomain: Optional[Union[str, Subdomain]] = None

    """Custom domain to use for the domain. If not set, the default custom domain will be used."""
    custom_domain: Optional[str] = None
