from typing import Any, Protocol, TypeVar, runtime_checkable

from typing_extensions import ParamSpec

from ._metadata import Metadata

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
P = ParamSpec("P")


@runtime_checkable
class Artifact(Protocol[T_co]):
    """Protocol for objects wrapped with Flyte metadata."""

    _flyte_metadata: Metadata

    def get_flyte_metadata(self) -> Metadata:
        """Get the Flyte metadata associated with this artifact."""
        ...


class ArtifactWrapper:
    """Zero-copy wrapper that preserves the original object interface."""

    __slots__ = ("_flyte_metadata", "_obj")

    def __init__(self, obj: T_co, metadata: Metadata) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_flyte_metadata", metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_obj", "_flyte_metadata"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._obj, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_obj", "_flyte_metadata"):
            raise AttributeError(f"Cannot delete {name}")
        delattr(self._obj, name)

    def __getattribute__(self, name: str) -> Any:
        """Override attribute access to make __class__ return the wrapped object's type."""
        if name == "__class__":
            return type(object.__getattribute__(self, "_obj"))
        elif name in (
            "_obj",
            "_flyte_metadata",
            "get_flyte_metadata",
            "__call__",
            "__repr__",
            "__str__",
            "__bool__",
            "__len__",
            "__iter__",
            "__getitem__",
            "__setitem__",
            "__contains__",
        ):
            return object.__getattribute__(self, name)
        else:
            return getattr(object.__getattribute__(self, "_obj"), name)

    def get_flyte_metadata(self) -> Metadata:
        """Get a copy of the Flyte metadata."""
        import copy

        return copy.deepcopy(self._flyte_metadata)

    # Forward common special methods for better compatibility
    def __str__(self) -> str:
        return str(self._obj)

    def __repr__(self) -> str:
        return f"Artifact[{type(self._obj).__name__}]({self._obj})"

    def __bool__(self) -> bool:
        return bool(self._obj)

    def __len__(self) -> int:
        return len(self._obj)

    def __iter__(self):
        return iter(self._obj)

    def __call__(self, *args: Any, **kwargs: Any):
        return self._obj(*args, **kwargs)

    def __getitem__(self, key):
        return self._obj[key]

    def __setitem__(self, key, value):
        self._obj[key] = value

    def __contains__(self, item):
        return item in self._obj


def new(obj: T, metadata: Metadata) -> T:
    """
    Wrap an object with Flyte metadata while preserving its type interface.

    Args:
        obj: The object to wrap
        metadata: Metadata to associate with the object

    Returns:
        A zero-copy wrapper that behaves exactly like the original object
        but carries additional Flyte metadata accessible via get_flyte_metadata()
    """
    wrapper = ArtifactWrapper(obj, metadata)
    return wrapper  # type: ignore[return-value]
