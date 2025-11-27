from typing import Dict, Optional, TypeVar

from flyteidl2.core import interface_pb2

from flyte.models import NativeInterface
from flyte.types._type_engine import TypeEngine

T = TypeVar("T")


def transform_variable_map(
    variable_map: Dict[str, type],
) -> Dict[str, interface_pb2.Variable]:
    """
    Given a map of str (names of inputs for instance) to their Python native types, return a map of the name to a
    Flyte Variable object with that type.
    """
    res = {}
    if variable_map:
        for k, v in variable_map.items():
            res[k] = transform_type(v)
    return res


def transform_native_to_typed_interface(
    interface: Optional[NativeInterface],
) -> Optional[interface_pb2.TypedInterface]:
    """
    Transform the given simple python native interface to FlyteIDL's interface
    """
    if interface is None:
        return None

    inputs_map = transform_variable_map(interface.get_input_types())
    outputs_map = transform_variable_map(interface.outputs)
    return interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(variables=inputs_map), outputs=interface_pb2.VariableMap(variables=outputs_map)
    )


def transform_type(x: type) -> interface_pb2.Variable:
    # add artifact handling eventually
    return interface_pb2.Variable(
        type=TypeEngine.to_literal_type(x),
    )
