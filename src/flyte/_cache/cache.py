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
    def get_version(self, salt: str, params: VersionParameters) -> str: ...


@rich.repr.auto
@dataclass
class Cache:
    """
    Cache configuration for a task.
    :param behavior: The behavior of the cache. Can be "auto", "override" or "disable".
    :param version_override: The version of the cache. If not provided, the version will be
     generated based on the cache policies
    :type version_override: Optional[str]
    :param serialize: Boolean that indicates if identical (ie. same inputs) instances of this task should be executed in
          serial when caching is enabled. This means that given multiple concurrent executions over identical inputs,
          only a single instance executes and the rest wait to reuse the cached results.
    :type serialize: bool
    :param ignored_inputs: A tuple of input names to ignore when generating the version hash.
    :type ignored_inputs: Union[Tuple[str, ...], str]
    :param salt: A salt used in the hash generation.
    :type salt: str
    :param policies: A list of cache policies to generate the version hash.
    :type policies: Optional[Union[List[CachePolicy], CachePolicy]]
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
