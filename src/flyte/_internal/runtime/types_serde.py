from typing import Dict, List, Mapping, Optional, TypeVar

from flyteidl2.core import interface_pb2

from flyte.models import NativeInterface
from flyte.types._type_engine import TypeEngine

T = TypeVar("T")

# The interface sorts inputs by name, so a task's declared input order is kept in its
# template's config, comma-separated, for callers that bind positional arguments.
INPUT_ORDER_CONFIG_KEY = "input_order"


def declared_input_order(config: Mapping[str, str]) -> Optional[List[str]]:
    """The task's inputs in the order its signature declares them, if its template recorded it."""
    order = config.get(INPUT_ORDER_CONFIG_KEY)
    if order is None:
        return None
    return order.split(",") if order else []


def transform_variable_map(
    variable_map: Dict[str, type],
) -> List[interface_pb2.VariableEntry]:
    """
    Given a map of str (names of inputs for instance) to their Python native types, return a list of
    VariableEntry objects with that type, sorted by key for consistency.
    """
    res = []
    if variable_map:
        # Sort by key to ensure consistent ordering for hashing
        for k in sorted(variable_map.keys()):
            res.append(interface_pb2.VariableEntry(key=k, value=transform_type(variable_map[k])))
    return res


def transform_native_to_typed_interface(
    interface: Optional[NativeInterface],
) -> Optional[interface_pb2.TypedInterface]:
    """
    Transform the given simple python native interface to FlyteIDL's interface
    """
    if interface is None:
        return None

    inputs_list = transform_variable_map(interface.get_input_types())
    outputs_list = transform_variable_map(interface.outputs)
    return interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(variables=inputs_list),
        outputs=interface_pb2.VariableMap(variables=outputs_list),
    )


def transform_type(x: type) -> interface_pb2.Variable:
    # add artifact handling eventually
    return interface_pb2.Variable(
        type=TypeEngine.to_literal_type(x),
    )
