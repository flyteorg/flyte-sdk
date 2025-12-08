from __future__ import annotations

import inspect
import typing
from enum import Enum
from typing import Dict, Generator, Literal, Tuple, Type, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from flyte._logging import logger

LITERAL_ENUM = "LiteralEnum"


def default_output_name(index: int = 0) -> str:
    return f"o{index}"


def output_name_generator(length: int) -> Generator[str, None, None]:
    for x in range(length):
        yield default_output_name(x)


def extract_return_annotation(return_annotation: Union[Type, Tuple, None]) -> Dict[str, Type]:
    """
    The input to this function should be sig.return_annotation where sig = inspect.signature(some_func)
    The purpose of this function is to sort out whether a function is returning one thing, or multiple things, and to
    name the outputs accordingly, either by using our default name function, or from a typing.NamedTuple.

        # Option 1
        nt1 = typing.NamedTuple("NT1", x_str=str, y_int=int)
        def t(a: int, b: str) -> nt1: ...

        # Option 2
        def t(a: int, b: str) -> typing.NamedTuple("NT1", x_str=str, y_int=int): ...

        # Option 3
        def t(a: int, b: str) -> typing.Tuple[int, str]: ...

        # Option 4
        def t(a: int, b: str) -> (int, str): ...

        # Option 5
        def t(a: int, b: str) -> str: ...

        # Option 6
        def t(a: int, b: str) -> None: ...

        # Options 7/8
        def t(a: int, b: str) -> List[int]: ...
        def t(a: int, b: str) -> Dict[str, int]: ...

    Note that Options 1 and 2 are identical, just syntactic sugar. In the NamedTuple case, we'll use the names in the
    definition. In all other cases, we'll automatically generate output names, indexed starting at 0.
    """
    if isinstance(return_annotation, str):
        raise TypeError("String return annotations are not supported.")

    # Handle Option 6
    # We can think about whether we should add a default output name with type None in the future.
    if return_annotation in (None, type(None), inspect.Signature.empty):
        return {}

    # This statement results in true for typing.Namedtuple, single and void return types, so this
    # handles Options 1, 2. Even though NamedTuple for us is multi-valued, it's a single value for Python
    if hasattr(return_annotation, "__bases__") and (
        isinstance(return_annotation, type) or isinstance(return_annotation, TypeVar)
    ):
        # isinstance / issubclass does not work for Namedtuple.
        # Options 1 and 2
        bases = return_annotation.__bases__  # type: ignore
        if len(bases) == 1 and bases[0] is tuple and hasattr(return_annotation, "_fields"):
            # Task returns named tuple
            return dict(get_type_hints(cast(Type, return_annotation), include_extras=True))

    if hasattr(return_annotation, "__origin__") and return_annotation.__origin__ is tuple:  # type: ignore
        # Handle option 3
        # Task returns unnamed typing.Tuple
        if len(return_annotation.__args__) == 1:  # type: ignore
            raise TypeError("Tuples should be used to indicate multiple return values, found only one return variable.")
        ra = get_args(return_annotation)
        annotations = {}
        for i, r in enumerate(ra):
            if r is Ellipsis:
                raise TypeError("Variable length tuples are not supported as return types.")
            if get_origin(r) is Literal:
                annotations[default_output_name(i)] = literal_to_enum(cast(Type, r))
            else:
                annotations[default_output_name(i)] = r
        return annotations

    elif isinstance(return_annotation, tuple):
        if len(return_annotation) == 1:
            raise TypeError("Please don't use a tuple if you're just returning one thing.")
        return dict(zip(list(output_name_generator(len(return_annotation))), return_annotation))

    else:
        # Handle all other single return types
        # Task returns unnamed native tuple
        if get_origin(return_annotation) is Literal:
            return {default_output_name(): literal_to_enum(cast(Type, return_annotation))}
        return {default_output_name(): cast(Type, return_annotation)}


def literal_to_enum(literal_type: Type) -> Type[Enum | typing.Any]:
    """Convert a Literal[...] into Union[str, Enum]."""

    if get_origin(literal_type) is not Literal:
        raise TypeError(f"{literal_type} is not a Literal")

    values = get_args(literal_type)
    if not all(isinstance(v, str) for v in values):
        logger.warning(f"Literal type {literal_type} contains non-string values, using Any instead of Enum")
        return typing.Any
    # Deduplicate & keep order
    enum_dict = {str(v).upper(): v for v in values}

    # Dynamically create an Enum
    literal_enum = Enum(LITERAL_ENUM, enum_dict)  # type: ignore

    return literal_enum  # type: ignore
