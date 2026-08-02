from typing import Any, Protocol, TypeVar, cast, runtime_checkable

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


# Primitive values are not allowed as artifacts: an artifact is an addressable,
# typically offloaded asset (a file, directory, dataframe, or structured model),
# not a scalar. Rejecting primitives here keeps the artifact catalog meaningful.
_PRIMITIVE_TYPES = (str, int, float, bool, bytes, complex)


def ensure_artifactable(obj: Any) -> None:
    """
    Validate that a value is allowed to be an artifact. Raises TypeError for
    primitives (str, int, float, bool, bytes, complex) and None.
    """
    if obj is None or isinstance(obj, _PRIMITIVE_TYPES):
        raise TypeError(
            f"values of type {type(obj).__name__!r} cannot be artifacts; use a File, Dir, "
            "DataFrame, dataclass, or pydantic model instead"
        )


def raise_if_nested_wrapper(obj: Any, _depth: int = 0) -> None:
    """
    Raise TypeError if an ArtifactWrapper is nested anywhere inside obj.

    Artifacts must be top-level task outputs: a wrapper nested inside a model
    or container would either fail or silently serialize its inner value and
    drop the artifact metadata (the wrapper masquerades as the wrapped type).
    Walks containers, dataclasses, and pydantic models to a bounded depth.
    """
    if _depth > 10 or obj is None or isinstance(obj, _PRIMITIVE_TYPES):
        return
    if type(obj) is ArtifactWrapper:
        if _depth == 0:
            # Top-level wrappers are handled (unwrapped + stamped) by the caller.
            return raise_if_nested_wrapper(object.__getattribute__(obj, "_obj"), _depth + 1)
        raise TypeError(
            "artifacts cannot be nested inside models or containers; return the "
            "wrapped value as a top-level task output instead"
        )
    if isinstance(obj, dict):
        for v in obj.values():
            raise_if_nested_wrapper(v, _depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            raise_if_nested_wrapper(v, _depth + 1)
    else:
        import dataclasses

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                raise_if_nested_wrapper(getattr(obj, f.name), _depth + 1)
        else:
            model_fields = getattr(type(obj), "model_fields", None)
            if model_fields:  # pydantic BaseModel
                for field_name in model_fields:
                    raise_if_nested_wrapper(getattr(obj, field_name, None), _depth + 1)


def new(obj: T, metadata: Metadata) -> T:
    """
    Wrap an object with Flyte metadata while preserving its type interface.

    Primitive values (str, int, float, bool, bytes) are rejected: artifacts are
    addressable assets such as files, directories, dataframes, or structured
    models. Artifacts must be returned directly from a task (top-level output);
    nesting a wrapped value inside another model is not supported and fails at
    serialization time.

    Args:
        obj: The object to wrap
        metadata: Metadata to associate with the object

    Returns:
        A zero-copy wrapper that behaves exactly like the original object
        but carries additional Flyte metadata accessible via get_flyte_metadata()
    """
    ensure_artifactable(obj)
    wrapper = ArtifactWrapper(obj, metadata)
    return cast(T, wrapper)
