from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
from dataclasses import dataclass
from types import NoneType
from typing import Any, Dict, List, Tuple, Union, get_args

from flyteidl2.core import execution_pb2, interface_pb2, literals_pb2
from flyteidl2.task import common_pb2, task_definition_pb2

import flyte.errors
import flyte.storage as storage
from flyte._context import ctx
from flyte.models import ActionID, NativeInterface, TaskContext
from flyte.types import TypeEngine, TypeTransformerFailedError


@dataclass(frozen=True)
class Inputs:
    proto_inputs: common_pb2.Inputs

    @classmethod
    def empty(cls) -> "Inputs":
        return cls(proto_inputs=common_pb2.Inputs())

    @property
    def context(self) -> Dict[str, str]:
        """Get the context as a dictionary."""
        return {kv.key: kv.value for kv in self.proto_inputs.context}


@dataclass(frozen=True)
class Outputs:
    proto_outputs: common_pb2.Outputs


@dataclass
class Error:
    err: execution_pb2.ExecutionError


# ------------------------------- CONVERT Methods ------------------------------- #


def _clean_error_code(code: str) -> Tuple[str, str | None]:
    """
    The error code may have a server injected code and is of the form `RetriesExhausedError|<code>` or `<code>`.

    :param code:
    :return: "user code", optional server code
    """
    if "|" in code:
        server_code, user_code = code.split("|", 1)
        return user_code.strip(), server_code.strip()
    return code.strip(), None


async def convert_inputs_to_native(inputs: Inputs, python_interface: NativeInterface) -> Dict[str, Any]:
    literals = {named_literal.name: named_literal.value for named_literal in inputs.proto_inputs.literals}
    native_vals = await TypeEngine.literal_map_to_kwargs(
        literals_pb2.LiteralMap(literals=literals), python_interface.get_input_types()
    )
    return native_vals


async def convert_upload_default_inputs(interface: NativeInterface) -> List[common_pb2.NamedParameter]:
    """
    Converts the default inputs of a NativeInterface to a list of NamedParameters for upload.
    This is used to upload default inputs to the Flyte backend.
    """
    if not interface.inputs:
        return []

    vars = []
    literal_coros = []
    for input_name, (input_type, default_value) in interface.inputs.items():
        if default_value and default_value is not inspect.Parameter.empty:
            lt = TypeEngine.to_literal_type(input_type)
            literal_coros.append(TypeEngine.to_literal(default_value, input_type, lt))
            vars.append((input_name, lt))

    literals: List[literals_pb2.Literal] = await asyncio.gather(*literal_coros)
    named_params = []
    for (name, lt), literal in zip(vars, literals):
        param = interface_pb2.Parameter(
            var=interface_pb2.Variable(
                type=lt,
            ),
            default=literal,
        )
        named_params.append(
            common_pb2.NamedParameter(
                name=name,
                parameter=param,
            ),
        )
    return named_params


def is_optional_type(tp) -> bool:
    """
    True if the *annotation* `tp` is equivalent to Optional[…].
    Works for Optional[T], Union[T, None], and T | None.
    """
    return NoneType in get_args(tp)  # fastest check


async def convert_from_native_to_inputs(
    interface: NativeInterface, *args, custom_context: Dict[str, str] | None = None, **kwargs
) -> Inputs:
    kwargs = interface.convert_to_kwargs(*args, **kwargs)

    missing = [key for key in interface.required_inputs() if key not in kwargs]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    # Read custom_context from TaskContext if available (inside task execution)
    # Otherwise use the passed parameter (for remote run initiation)
    context_kvs = None
    tctx = ctx()
    if tctx and tctx.custom_context:
        # Inside a task - read from TaskContext
        context_to_use = tctx.custom_context
        context_kvs = [literals_pb2.KeyValuePair(key=k, value=v) for k, v in context_to_use.items()]
    elif custom_context:
        # Remote run initiation
        context_kvs = [literals_pb2.KeyValuePair(key=k, value=v) for k, v in custom_context.items()]

    if len(interface.inputs) == 0:
        # Handle context even for empty inputs
        return Inputs(proto_inputs=common_pb2.Inputs(context=context_kvs))

    # fill in defaults if missing
    type_hints: Dict[str, type] = {}
    already_converted_kwargs: Dict[str, literals_pb2.Literal] = {}
    for input_name, (input_type, default_value) in interface.inputs.items():
        if input_name in kwargs:
            type_hints[input_name] = input_type
        elif (
            (default_value is not None and default_value is not inspect.Signature.empty)
            or (default_value is None and is_optional_type(input_type))
            or input_type is None
            or input_type is type(None)
        ):
            if default_value == NativeInterface.has_default:
                if interface._remote_defaults is None or input_name not in interface._remote_defaults:
                    raise ValueError(f"Input '{input_name}' has a default value but it is not set in the interface.")
                already_converted_kwargs[input_name] = interface._remote_defaults[input_name]
            elif input_type is None or input_type is type(None):
                # If the type is 'None' or 'class<None>', we assume it's a placeholder for no type
                kwargs[input_name] = None
                type_hints[input_name] = NoneType
            else:
                kwargs[input_name] = default_value
                type_hints[input_name] = input_type

    literal_map = await TypeEngine.dict_to_literal_map(kwargs, type_hints)
    if len(already_converted_kwargs) > 0:
        copied_literals: Dict[str, literals_pb2.Literal] = {}
        for k, v in literal_map.literals.items():
            copied_literals[k] = v
        # Add the already converted kwargs to the literal map
        for k, v in already_converted_kwargs.items():
            copied_literals[k] = v
        literal_map = literals_pb2.LiteralMap(literals=copied_literals)

    # Make sure we the interface, not literal_map or kwargs, because those may have a different order
    return Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name=k, value=literal_map.literals[k]) for k in interface.inputs.keys()],
            context=context_kvs,
        )
    )


async def convert_from_inputs_to_native(native_interface: NativeInterface, inputs: Inputs) -> Dict[str, Any]:
    """
    Converts the inputs from a run definition proto to a native Python dictionary.
    :param native_interface: The native interface of the task.
    :param inputs: The run definition inputs proto.
    :return: A dictionary of input names to their native Python values.
    """
    if not inputs or not inputs.proto_inputs or not inputs.proto_inputs.literals:
        return {}

    literals = {named_literal.name: named_literal.value for named_literal in inputs.proto_inputs.literals}
    return await TypeEngine.literal_map_to_kwargs(
        literals_pb2.LiteralMap(literals=literals), native_interface.get_input_types()
    )


async def convert_from_native_to_outputs(o: Any, interface: NativeInterface, task_name: str = "") -> Outputs:
    # Always make it a tuple even if it's just one item to simplify logic below
    if not isinstance(o, tuple):
        o = (o,)

    if len(interface.outputs) == 0:
        if len(o) != 0:
            if len(o) == 1 and o[0] is not None:
                raise flyte.errors.RuntimeDataValidationError(
                    "o0",
                    f"Expected no outputs but got {o},did you miss a return type annotation?",
                    task_name,
                )
    else:
        assert len(o) == len(interface.outputs), (
            f"Received {len(o)} outputs but return annotation has {len(interface.outputs)} outputs specified. "
        )
    named = []
    for (output_name, python_type), v in zip(interface.outputs.items(), o):
        try:
            lit = await TypeEngine.to_literal(v, python_type, TypeEngine.to_literal_type(python_type))
            named.append(common_pb2.NamedLiteral(name=output_name, value=lit))
        except TypeTransformerFailedError as e:
            raise flyte.errors.RuntimeDataValidationError(output_name, e, task_name)

    return Outputs(proto_outputs=common_pb2.Outputs(literals=named))


async def convert_outputs_to_native(interface: NativeInterface, outputs: Outputs) -> Union[Any, Tuple[Any, ...]]:
    lm = literals_pb2.LiteralMap(
        literals={named_literal.name: named_literal.value for named_literal in outputs.proto_outputs.literals}
    )
    kwargs = await TypeEngine.literal_map_to_kwargs(lm, interface.outputs)
    if len(kwargs) == 0:
        return None
    elif len(kwargs) == 1:
        return next(iter(kwargs.values()))
    else:
        # Return as tuple if multiple outputs, make sure to order correctly as it seems proto maps can change ordering
        return tuple(kwargs[k] for k in interface.outputs.keys())


def convert_error_to_native(err: execution_pb2.ExecutionError | Exception | Error) -> Exception | None:
    if not err:
        return None

    if isinstance(err, Exception):
        return err

    if isinstance(err, Error):
        err = err.err

    user_code, _server_code = _clean_error_code(err.code)
    match err.kind:
        case execution_pb2.ExecutionError.UNKNOWN:
            return flyte.errors.RuntimeUnknownError(code=user_code, message=err.message, worker=err.worker)
        case execution_pb2.ExecutionError.USER:
            if "OOM" in err.code.upper():
                return flyte.errors.OOMError(code=user_code, message=err.message, worker=err.worker)
            elif "Interrupted" in err.code:
                return flyte.errors.TaskInterruptedError(code=user_code, message=err.message, worker=err.worker)
            elif "PrimaryContainerNotFound" in err.code:
                return flyte.errors.PrimaryContainerNotFoundError(
                    code=user_code, message=err.message, worker=err.worker
                )
            elif "RetriesExhausted" in err.code:
                return flyte.errors.RetriesExhaustedError(code=user_code, message=err.message, worker=err.worker)
            elif "Unknown" in err.code:
                return flyte.errors.RuntimeUnknownError(code=user_code, message=err.message, worker=err.worker)
            elif "InvalidImageName" in err.code:
                return flyte.errors.InvalidImageNameError(code=user_code, message=err.message, worker=err.worker)
            elif "ImagePullBackOff" in err.code:
                return flyte.errors.ImagePullBackOffError(code=user_code, message=err.message, worker=err.worker)
            return flyte.errors.RuntimeUserError(code=user_code, message=err.message, worker=err.worker)
        case execution_pb2.ExecutionError.SYSTEM:
            return flyte.errors.RuntimeSystemError(code=user_code, message=err.message, worker=err.worker)
    return None


def convert_from_native_to_error(err: BaseException) -> Error:
    if isinstance(err, flyte.errors.RuntimeUnknownError):
        return Error(
            err=execution_pb2.ExecutionError(
                kind=execution_pb2.ExecutionError.UNKNOWN,
                code=err.code,
                message=str(err),
                worker=err.worker,
            )
        )
    elif isinstance(err, flyte.errors.RuntimeUserError):
        return Error(
            err=execution_pb2.ExecutionError(
                kind=execution_pb2.ExecutionError.USER,
                code=err.code,
                message=str(err),
                worker=err.worker,
            )
        )
    elif isinstance(err, flyte.errors.RuntimeSystemError):
        return Error(
            err=execution_pb2.ExecutionError(
                kind=execution_pb2.ExecutionError.SYSTEM,
                code=err.code,
                message=str(err),
                worker=err.worker,
            )
        )
    else:
        return Error(
            err=execution_pb2.ExecutionError(
                kind=execution_pb2.ExecutionError.UNKNOWN,
                code=type(err).__name__,
                message=str(err),
                worker="UNKNOWN",
            )
        )


def hash_data(data: Union[str, bytes]) -> str:
    """
    Generate a hash for the given data. If the data is a string, it will be encoded to bytes before hashing.
    :param data: The data to hash, can be a string or bytes.
    :return: A hexadecimal string representation of the hash.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return base64.b64encode(digest).decode("utf-8")


def generate_inputs_hash(serialized_inputs: str | bytes) -> str:
    """
    Generate a hash for the inputs. This is used to uniquely identify the inputs for a task.
    :return: A hexadecimal string representation of the hash.
    """
    return hash_data(serialized_inputs)


def generate_inputs_repr_for_literal(literal: literals_pb2.Literal) -> bytes:
    """
    Generate a byte representation for a single literal that is meant to be hashed as part of the cache key
    computation for an Action. This function should just serialize the literal deterministically, but will
    use an existing hash value if present in the Literal.  This is trivial, except we need to handle nested literals
    (inside collections and maps), that may have the hash property set.

    :param literal: The literal to get a hashable representation for.
    :return: byte representation of the literal that can be fed into a hash function.
    """
    # If the literal has a hash value, use that instead of serializing the full literal
    if literal.hash:
        return literal.hash.encode("utf-8")

    if literal.HasField("collection"):
        buf = bytearray()
        for nested_literal in literal.collection.literals:
            if nested_literal.hash:
                buf += nested_literal.hash.encode("utf-8")
            else:
                buf += generate_inputs_repr_for_literal(nested_literal)

        b = bytes(buf)
        return b

    elif literal.HasField("map"):
        buf = bytearray()
        # Sort keys to ensure deterministic ordering
        for key in sorted(literal.map.literals.keys()):
            nested_literal = literal.map.literals[key]
            buf += key.encode("utf-8")
            if nested_literal.hash:
                buf += nested_literal.hash.encode("utf-8")
            else:
                buf += generate_inputs_repr_for_literal(nested_literal)

        b = bytes(buf)
        return b

    # For all other cases (scalars, etc.), just serialize the literal normally
    return literal.SerializeToString(deterministic=True)


def generate_inputs_hash_for_named_literals(inputs: list[common_pb2.NamedLiteral]) -> str:
    """
    Generate a hash for the inputs using the new literal representation approach that respects
    hash values already present in literals. This is used to uniquely identify the inputs for a task
    when some literals may have precomputed hash values.

    :param inputs: List of NamedLiteral inputs to hash.
    :return: A base64-encoded string representation of the hash.
    """
    if not inputs:
        return ""

    # Build the byte representation by concatenating each literal's representation
    combined_bytes = b""
    for named_literal in inputs:
        # Add the name to ensure order matters
        name_bytes = named_literal.name.encode("utf-8")
        literal_bytes = generate_inputs_repr_for_literal(named_literal.value)
        # Combine name and literal bytes with a separator to avoid collisions
        combined_bytes += name_bytes + b":" + literal_bytes + b";"

    return hash_data(combined_bytes)


def generate_inputs_hash_from_proto(inputs: common_pb2.Inputs) -> str:
    """
    Generate a hash for the inputs. This is used to uniquely identify the inputs for a task.
    :param inputs: The inputs to hash.
    :return: A hexadecimal string representation of the hash.
    """
    if not inputs or not inputs.literals:
        return ""
    return generate_inputs_hash_for_named_literals(list(inputs.literals))


def generate_interface_hash(task_interface: interface_pb2.TypedInterface) -> str:
    """
    Generate a hash for the task interface. This is used to uniquely identify the task interface.
    :param task_interface: The interface of the task.
    :return: A hexadecimal string representation of the hash.
    """
    if not task_interface:
        return ""
    serialized_interface = task_interface.SerializeToString(deterministic=True)
    return hash_data(serialized_interface)


def generate_cache_key_hash(
    task_name: str,
    inputs_hash: str,
    task_interface: interface_pb2.TypedInterface,
    cache_version: str,
    ignored_input_vars: List[str],
    proto_inputs: common_pb2.Inputs,
) -> str:
    """
    Generate a cache key hash based on the inputs hash, task name, task interface, and cache version.
    This is used to uniquely identify the cache key for a task.

    :param task_name: The name of the task.
    :param inputs_hash: The hash of the inputs.
    :param task_interface: The interface of the task.
    :param cache_version: The version of the cache.
    :param ignored_input_vars: A list of input variable names to ignore when generating the cache key.
    :param proto_inputs: The proto inputs for the task, only used if there are ignored inputs.
    :return: A hexadecimal string representation of the cache key hash.
    """
    if ignored_input_vars:
        filtered = [named_lit for named_lit in proto_inputs.literals if named_lit.name not in ignored_input_vars]
        final = common_pb2.Inputs(literals=filtered)
        final_inputs = generate_inputs_hash_from_proto(final)
    else:
        final_inputs = inputs_hash

    interface_hash = generate_interface_hash(task_interface)

    data = f"{final_inputs}{task_name}{interface_hash}{cache_version}"
    return hash_data(data)


def generate_sub_action_id_and_output_path(
    tctx: TaskContext,
    task_spec_or_name: task_definition_pb2.TaskSpec | str,
    inputs_hash: str,
    invoke_seq: int,
) -> Tuple[ActionID, str]:
    """
    Generate a sub-action ID and output path based on the current task context, task name, and inputs.

    action name = current action name + task name + input hash + group name (if available)
    :param tctx:
    :param task_spec_or_name: task specification or task name. Task name is only used in case of trace actions.
    :param inputs_hash: Consistent hash string of the inputs
    :param invoke_seq: The sequence number of the invocation, used to differentiate between multiple invocations.
    :return:
    """
    current_action_id = tctx.action
    current_output_path = tctx.run_base_dir
    if isinstance(task_spec_or_name, task_definition_pb2.TaskSpec):
        task_spec_or_name.task_template.interface
        task_hash = hash_data(task_spec_or_name.SerializeToString(deterministic=True))
    else:
        task_hash = task_spec_or_name
    sub_action_id = current_action_id.new_sub_action_from(
        task_hash=task_hash,
        input_hash=inputs_hash,
        group=tctx.group_data.name if tctx.group_data else None,
        task_call_seq=invoke_seq,
    )
    sub_run_output_path = storage.join(current_output_path, sub_action_id.name)
    return sub_action_id, sub_run_output_path
