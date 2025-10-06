import inspect
from typing import Any, Dict, Iterable, Tuple, Type, cast

from flyteidl2.core import interface_pb2, literals_pb2
from flyteidl2.task import common_pb2

from flyte.models import NativeInterface


def guess_interface(
    interface: interface_pb2.TypedInterface, default_inputs: Iterable[common_pb2.NamedParameter] | None = None
) -> NativeInterface:
    """
    Returns the interface of the task with guessed types, as types may not be present in current env.
    """
    import flyte.types

    if interface is None:
        return NativeInterface({}, {})

    default_input_literals: Dict[str, literals_pb2.Literal] = {}
    if default_inputs is not None:
        for param in default_inputs:
            if param.parameter.HasField("default"):
                default_input_literals[param.name] = param.parameter.default

    guessed_inputs: Dict[str, Tuple[Type[Any], Any] | Any] = {}
    if interface.inputs is not None and len(interface.inputs.variables) > 0:
        input_types = flyte.types.TypeEngine.guess_python_types(cast(dict, interface.inputs.variables))
        for name, t in input_types.items():
            if name not in default_input_literals:
                guessed_inputs[name] = (t, inspect.Parameter.empty)
            else:
                guessed_inputs[name] = (t, NativeInterface.has_default)

    guessed_outputs: Dict[str, Type[Any]] = {}
    if interface.outputs is not None and len(interface.outputs.variables) > 0:
        guessed_outputs = flyte.types.TypeEngine.guess_python_types(cast(dict, interface.outputs.variables))

    return NativeInterface.from_types(guessed_inputs, guessed_outputs, default_input_literals)
