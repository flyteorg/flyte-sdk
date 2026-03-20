import hashlib
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import rich.repr
from typing_extensions import Literal, ParamSpec, TypeVar, get_args

# if TYPE_CHECKING:
from flyte._image import Image
from flyte.models import CodeBundle

P = ParamSpec("P")
FuncOut = TypeVar("FuncOut")

CacheBehavior = Literal["auto", "override", "disable"]


@dataclass
class VersionParameters(Generic[P, FuncOut]):
    """
    Parameters used for cache version hash generation.

    :param func: The function to generate a version for. This is a required parameter but can be any callable
    :type func: Callable[P, FuncOut]
    :param image: The container image to generate a version for. This can be a string representing the
        image name or an Image object.
    :type image: Optional[Union[str, Image]]
    """

    func: Callable[P, FuncOut] | None
    image: Optional[Union[str, Image]] = None
    code_bundle: Optional[CodeBundle] = None


@runtime_checkable
class CachePolicy(Protocol):
    """
    Protocol for custom cache version strategies.

    Implement `get_version(salt, params) -> str` to define how cache versions
    are computed. The default implementation is `FunctionBodyPolicy`, which
    hashes the function source code.

    Example custom policy:

    ```python
    class GitHashPolicy:
        def get_version(self, salt: str, params: VersionParameters) -> str:
            import subprocess
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            return hashlib.sha256(f"{salt}{git_hash}".encode()).hexdigest()
    ```
    """

    def get_version(self, salt: str, params: VersionParameters) -> str: ...


@rich.repr.auto
@dataclass
class Cache:
    """
    Cache configuration for a task.

    Three cache behaviors are available:

    - `"auto"` — Cache version is computed automatically from cache policies
      (default: `FunctionBodyPolicy`, which hashes the function source code).
      Any change to the function body invalidates the cache.
    - `"override"` — You provide an explicit `version_override` string.
      Cache is only invalidated when you change the version.
    - `"disable"` — Caching is disabled; task always re-executes.

    Set via `TaskEnvironment(cache=...)`, `@env.task(cache=...)`, or
    `task.override(cache=...)`.

    :param behavior: Cache behavior — `"auto"`, `"override"`, or `"disable"`.
    :param version_override: Explicit cache version string. Only used when
        `behavior="override"`.
    :param serialize: If `True`, concurrent executions with identical inputs will
        be serialized — only one runs and the rest wait for and reuse the cached result.
        Default `False`.
    :param ignored_inputs: Input parameter names to exclude from the cache key.
        Useful when some inputs (e.g., timestamps) shouldn't affect caching.
    :param salt: Additional salt for cache key generation. Use to create separate
        cache namespaces (e.g., `salt="v2"` to invalidate all existing caches).
    :param policies: Cache policies for version generation. Defaults to
        `[FunctionBodyPolicy()]` when `behavior="auto"`. Provide a custom
        `CachePolicy` implementation for alternative versioning strategies.
    """

    behavior: CacheBehavior
    version_override: Optional[str] = None
    serialize: bool = False
    ignored_inputs: Union[Tuple[str, ...], str] = field(default_factory=tuple)
    salt: str = ""
    policies: Optional[Union[List[CachePolicy], CachePolicy]] = None

    def __post_init__(self):
        if self.behavior not in get_args(CacheBehavior):
            raise ValueError(f"Invalid cache behavior: {self.behavior}. Must be one of ['auto', 'override', 'disable']")

        # Still setup _ignore_inputs when cache is disabled to prevent _ignored_inputs attribute not found error
        if isinstance(self.ignored_inputs, str):
            self._ignored_inputs = (self.ignored_inputs,)
        else:
            self._ignored_inputs = self.ignored_inputs

        if self.behavior == "disable":
            return

        # Normalize policies so that self._policies is always a list
        if self.policies is None:
            from flyte._cache.defaults import get_default_policies

            self.policies = get_default_policies()
        elif isinstance(self.policies, CachePolicy):
            self.policies = [self.policies]

        if self.version_override is None and not self.policies:
            raise ValueError("If version is not defined then at least one cache policy needs to be set")

    def is_enabled(self) -> bool:
        """
        Check if the cache policy is enabled.
        """
        return self.behavior in ["auto", "override"]

    def get_ignored_inputs(self) -> Tuple[str, ...]:
        return self._ignored_inputs

    def get_version(self, params: Optional[VersionParameters] = None) -> str:
        if not self.is_enabled():
            return ""

        if self.version_override is not None:
            return self.version_override

        if params is None:
            raise ValueError("Version parameters must be provided when version_override is not set.")

        if params.code_bundle is not None:
            if params.code_bundle.pkl is not None:
                return params.code_bundle.computed_version

        task_hash = ""
        if self.policies is None:
            raise ValueError("Cache policies are not set.")
        policies = self.policies if isinstance(self.policies, list) else [self.policies]
        for policy in policies:
            try:
                task_hash += policy.get_version(self.salt, params)
            except Exception as e:
                raise ValueError(f"Failed to generate version for cache policy {policy}.") from e

        hash_obj = hashlib.sha256(task_hash.encode())
        return hash_obj.hexdigest()


CacheRequest = CacheBehavior | Cache


def cache_from_request(cache: CacheRequest) -> Cache:
    """
    Coerce user input into a cache object.
    """
    if isinstance(cache, Cache):
        return cache
    return Cache(behavior=cache)
