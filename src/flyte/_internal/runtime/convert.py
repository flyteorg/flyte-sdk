from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import inspect
from dataclasses import dataclass
from types import NoneType
from typing import Any, Dict, List, Optional, Tuple, Union, cast, get_args

from flyteidl2.core import execution_pb2, interface_pb2, literals_pb2, tasks_pb2
from flyteidl2.task import common_pb2

import flyte.errors
import flyte.storage as storage
from flyte._context import ctx
from flyte.models import ActionID, NativeInterface, TaskContext
from flyte.types import TypeEngine, TypeTransformerFailedError

# Reserved key under which a scheduled trigger stashes the name of its kickoff-time-bound input arg
# in Inputs.context (set by trigger_serde at registration). The per-fire value is never in the
# (offloaded) inputs blob; at execution we fill that input from the run start time on the task
# context. The key is internal plumbing, so it is excluded from the user-facing Inputs.context.
KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY = "_u_kickoff_time_input_arg"

# Name of the output slot currently being serialized (e.g. "o0"), scoped to
# a single ``TypeEngine.to_literal`` call by ``convert_from_native_to_outputs``.
# Lets a TypeTransformer attribute a value to the output it's being returned
# as — without threading the name through ``to_literal``'s signature (which
# every transformer would have to adopt). ``None`` outside output conversion
# (e.g. on the input path), so readers must treat absence as "unknown".
_output_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "flyte_current_output_name", default=None
)


def current_output_name() -> Optional[str]:
    """The output slot name being converted right now (e.g. ``"o0"``), or
    ``None`` when not inside output conversion. See `_output_name_var`.
    """
    return _output_name_var.get()


@dataclass(frozen=True)
class Inputs:
    proto_inputs: common_pb2.Inputs

    @classmethod
    def empty(cls) -> "Inputs":
        return cls(proto_inputs=common_pb2.Inputs())

    @property
    def context(self) -> Dict[str, str]:
        """Get the context as a dictionary (excluding internal reserved keys)."""
        return {kv.key: kv.value for kv in self.proto_inputs.context if kv.key != KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY}


@dataclass(frozen=True)
class Outputs:
    proto_outputs: common_pb2.Outputs


@dataclass
class Error:
    err: execution_pb2.ExecutionError
    recoverable: bool = True


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
    # A scheduled trigger conveys the name of its kickoff-time-bound input via inputs.context (the
    # per-fire value is never carried in the inputs blob). Fill it from the run start time already on
    # the task context rather than reopening/mutating the proto inputs.
    kickoff_arg = next(
        (kv.value for kv in inputs.proto_inputs.context if kv.key == KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY),
        None,
    )
    if kickoff_arg:
        tctx = ctx()
        if tctx and tctx.run_start_time is not None:
            native_vals[kickoff_arg] = tctx.run_start_time
    return native_vals


async def convert_upload_default_inputs(
    interface: NativeInterface,
) -> List[common_pb2.NamedParameter]:
    """
    Converts the default inputs of a NativeInterface to a list of NamedParameters for upload.
    This is used to upload default inputs to the Flyte backend.
    """
    if not interface.inputs:
        return []

    # flyte.TriggerTime is a sentinel that gets bound at trigger fire time, not a
    # serializable default. Importing lazily avoids a circular import at module load.
    from flyte._trigger import _trigger_time

    vars = []
    literal_coros = []
    for input_name, (input_type, default_value) in interface.inputs.items():
        if default_value is not inspect.Parameter.empty:
            if isinstance(default_value, _trigger_time):
                raise ValueError(
                    f"Input '{input_name}' uses flyte.TriggerTime as its default value. "
                    "flyte.TriggerTime is only valid as a value in `flyte.Trigger(inputs=...)` — "
                    "it cannot be used as a regular task default. Remove the default from the "
                    "task signature and pass it through the Trigger inputs instead."
                )
            lt = TypeEngine.to_literal_type(input_type)
            literal_coros.append(TypeEngine.to_literal(default_value, input_type, lt))
            vars.append((input_name, lt))

    literals: List[literals_pb2.Literal] = cast(
        "List[literals_pb2.Literal]", await asyncio.gather(*literal_coros, return_exceptions=True)
    )
    named_params = []
    for (name, lt), literal in zip(vars, literals):
        if isinstance(literal, Exception):
            raise RuntimeError(f"Failed to convert default value for parameter '{name}'") from literal
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
    interface: NativeInterface,
    *args,
    custom_context: Dict[str, str] | None = None,
    **kwargs,
) -> Inputs:
    return await _convert_from_native_to_inputs_impl(interface, args, custom_context, kwargs)


# Depth bound for the nested-artifact walk, mirroring raise_if_nested_wrapper.
_ARTIFACT_SCAN_MAX_DEPTH = 10
_ARTIFACT_SCAN_PRIMITIVES = (str, int, float, bool, bytes, complex)


def _raise_if_nested_artifact(value: Any, arg_name: str, _depth: int = 0) -> None:
    """
    Reject an Artifact buried inside a dict, dataclass, or model.

    An artifact binds either as a whole input or as an element of a list input, because those
    are the two shapes whose literal we can assemble from the artifact's stored literal. Anywhere
    else the Artifact object reaches the type engine and dies with an unreadable message about
    the wrong python type, quoting the entire artifact protobuf. Fail here with something a
    caller can act on. Mirrors ``raise_if_nested_wrapper`` in flyte.artifacts._wrapper.
    """
    from flyte.remote import Artifact

    if _depth > _ARTIFACT_SCAN_MAX_DEPTH or value is None or isinstance(value, _ARTIFACT_SCAN_PRIMITIVES):
        return
    if isinstance(value, Artifact):
        raise ValueError(
            f"argument '{arg_name}' has an Artifact nested inside a container. Artifacts bind as a "
            f"whole input, or as elements of a list input -- not nested inside dicts, dataclasses, "
            f"or models. Pass the artifact directly, or materialize it first with `await artifact.to_python()`."
        )
    if isinstance(value, dict):
        for v in value.values():
            _raise_if_nested_artifact(v, arg_name, _depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _raise_if_nested_artifact(v, arg_name, _depth + 1)
    else:
        import dataclasses

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                _raise_if_nested_artifact(getattr(value, f.name), arg_name, _depth + 1)
        else:
            model_fields = getattr(type(value), "model_fields", None)
            if model_fields:  # pydantic BaseModel
                for field_name in model_fields:
                    _raise_if_nested_artifact(getattr(value, field_name, None), arg_name, _depth + 1)


async def _coerce_artifact(artifact: Any, arg_name: str, declared_type: type | None) -> literals_pb2.Literal:
    """Coerce one artifact to its input's declared type, turning a transformer failure into an
    error that names both sides."""
    try:
        return await artifact.coerce_to_literal(declared_type)
    except TypeTransformerFailedError as e:
        raise ValueError(
            f"artifact '{artifact.name}@{artifact.version}' cannot bind to input '{arg_name}' "
            f"declared as {declared_type}: {e}"
        ) from e


async def bind_artifact_literals(
    interface: NativeInterface, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Dict[str, literals_pb2.Literal], Dict[str, Any]]:
    """
    Split artifact-valued arguments out of ``kwargs`` into ready-made literals.

    Each artifact is coerced to its input's declared type via ``Artifact.coerce_to_literal``:
    the stored literal round-trips through the type engine, so the engine owns every
    compatibility rule (Optional/union wrapping, coercions) and a mismatch fails at submit
    time rather than inside the task. The coerced literal carries the artifact's identity,
    copied from the service's stamp -- nothing here computes provenance.

    :return: (literals keyed by input name, the remaining kwargs to convert normally)
    """
    from flyte.remote import Artifact

    named = interface.convert_to_kwargs(*args, **kwargs)
    bound: Dict[str, literals_pb2.Literal] = {}
    remaining: Dict[str, Any] = {}

    for name, value in named.items():
        declared_type = interface.inputs[name][0] if name in interface.inputs else None

        if isinstance(value, Artifact):
            bound[name] = await _coerce_artifact(value, name, declared_type)
            continue

        if isinstance(value, list) and any(isinstance(item, Artifact) for item in value):
            element_type = next(iter(get_args(declared_type)), None) if declared_type is not None else None
            element_literals = []
            for index, item in enumerate(value):
                if isinstance(item, Artifact):
                    element_literals.append(await _coerce_artifact(item, f"{name}[{index}]", element_type))
                else:
                    # Plain elements still convert normally, so mixed lists keep working.
                    if element_type is None:
                        raise ValueError(
                            f"argument '{name}' mixes artifacts with plain values but its element type "
                            f"could not be determined from the task interface."
                        )
                    lt = TypeEngine.to_literal_type(element_type)
                    element_literals.append(await TypeEngine.to_literal(item, element_type, lt))
            bound[name] = literals_pb2.Literal(collection=literals_pb2.LiteralCollection(literals=element_literals))
            continue

        _raise_if_nested_artifact(value, name)
        remaining[name] = value

    return bound, remaining


async def convert_from_native_to_inputs_binding_artifacts(
    interface: NativeInterface,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    custom_context: Dict[str, str] | None = None,
) -> Inputs:
    """
    Convert run arguments to Inputs, binding any ``flyte.remote.Artifact`` argument to its stored
    literal coerced to the input's declared type (see ``bind_artifact_literals``).

    Takes args/kwargs as explicit containers rather than ``*args, **kwargs`` so an input actually
    named ``custom_context`` cannot be swallowed by the signature.
    """
    bound, remaining = await bind_artifact_literals(interface, args, kwargs)
    return await _convert_from_native_to_inputs_impl(interface, (), custom_context, remaining, preconverted=bound)


def _is_has_default_sentinel(value: Any) -> bool:
    """
    Return True if `value` is the `_has_default` sentinel (the class itself or an instance).

    The class is intended purely as a *marker* on `NativeInterface.inputs[name][1]` to indicate
    "this remote task input has a default value stored on the spec". It must never be treated as a
    real input value. We check both forms because the CLI's click integration used to coerce the
    class into an instance (click instantiates callable defaults), so either shape can leak through
    into kwargs.
    """
    return value is NativeInterface.has_default or isinstance(value, NativeInterface.has_default)


async def _convert_from_native_to_inputs_impl(
    interface: NativeInterface,
    args: Tuple[Any, ...],
    custom_context: Dict[str, str] | None,
    kwargs: Dict[str, Any],
    preconverted: Dict[str, literals_pb2.Literal] | None = None,
) -> Inputs:
    kwargs = interface.convert_to_kwargs(*args, **kwargs)

    # Drop any sentinel values from kwargs so the loop below falls through to the `_remote_defaults`
    # branch and substitutes the literal default instead of attempting to serialize the sentinel.
    kwargs = {k: v for k, v in kwargs.items() if not _is_has_default_sentinel(v)}

    # Inputs whose literal is already built (artifact-bound values) are satisfied even though they
    # never appear in kwargs.
    preconverted = preconverted or {}
    missing = [key for key in interface.required_inputs() if key not in kwargs and key not in preconverted]
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
    already_converted_kwargs: Dict[str, literals_pb2.Literal] = dict(preconverted)
    for input_name, (input_type, default_value) in interface.inputs.items():
        if input_name in preconverted:
            continue
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
    from flyte.artifacts._metadata import to_produced_artifact
    from flyte.artifacts._wrapper import ArtifactWrapper, raise_if_nested_wrapper

    named = []
    produced: list[common_pb2.ProducedArtifact] = []
    for (output_name, python_type), v in zip(interface.outputs.items(), o):
        # Capture the metadata attached by flyte.artifacts.new(...) before to_literal unwraps
        # the wrapper and discards it, then emit a ProducedArtifact declaration on the Outputs
        # envelope so the backend can register the artifact. The declaration carries the
        # declared output type (this SDK is authoritative for it).
        raise_if_nested_wrapper(v)
        produced_md = v.get_flyte_metadata() if isinstance(v, ArtifactWrapper) else None

        # Expose the output slot name to transformers for the duration of this
        # single conversion (see ``current_output_name``), then always clear it.
        tok = _output_name_var.set(output_name)
        try:
            literal_type = TypeEngine.to_literal_type(python_type)
            lit = await TypeEngine.to_literal(v, python_type, literal_type)
            if produced_md is not None:
                produced.append(to_produced_artifact(produced_md, output=output_name, literal_type=literal_type))
            named.append(common_pb2.NamedLiteral(name=output_name, value=lit))
        except TypeTransformerFailedError as e:
            raise flyte.errors.RuntimeDataValidationError(output_name, e, task_name)
        finally:
            _output_name_var.reset(tok)

    return Outputs(proto_outputs=common_pb2.Outputs(literals=named, produced_artifacts=produced))


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
        # Return as tuple if multiple outputs are defined in the interface,
        # to match the order of outputs in the interface
        return tuple(kwargs[k] for k in interface.outputs.keys())


def convert_error_to_native(
    err: execution_pb2.ExecutionError | Exception | Error,
) -> Exception | None:
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
    if isinstance(err, flyte.errors.NonRecoverableError):
        return Error(
            err=execution_pb2.ExecutionError(
                kind=execution_pb2.ExecutionError.USER,
                code=err.code,
                message=str(err),
                worker=err.worker,
            ),
            recoverable=False,
        )
    elif isinstance(err, flyte.errors.RuntimeUnknownError):
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


def generate_inputs_hash_for_named_literals(
    inputs: list[common_pb2.NamedLiteral],
) -> str:
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

    # Create a copy and sort variables by key to ensure order-independent hashing
    sorted_interface = interface_pb2.TypedInterface()
    sorted_interface.CopyFrom(task_interface)

    if sorted_interface.inputs and sorted_interface.inputs.variables:
        sorted_inputs = sorted(sorted_interface.inputs.variables, key=lambda entry: entry.key)
        del sorted_interface.inputs.variables[:]
        sorted_interface.inputs.variables.extend(sorted_inputs)

    if sorted_interface.outputs and sorted_interface.outputs.variables:
        sorted_outputs = sorted(sorted_interface.outputs.variables, key=lambda entry: entry.key)
        del sorted_interface.outputs.variables[:]
        sorted_interface.outputs.variables.extend(sorted_outputs)

    serialized_interface = sorted_interface.SerializeToString(deterministic=True)
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
        final_inputs = generate_filtered_inputs_hash(proto_inputs, ignored_input_vars)
    else:
        final_inputs = inputs_hash

    interface_hash = generate_interface_hash(task_interface)

    data = f"{final_inputs}{task_name}{interface_hash}{cache_version}"
    return hash_data(data)


def generate_filtered_inputs_hash(proto_inputs: common_pb2.Inputs, ignored_input_vars: List[str]) -> str:
    """
    Generate an inputs hash excluding the given input variable names.

    :param proto_inputs: The proto inputs for the task.
    :param ignored_input_vars: Input variable names to exclude from the hash.
    :return: A hexadecimal string representation of the hash.
    """
    filtered = [named_lit for named_lit in proto_inputs.literals if named_lit.name not in ignored_input_vars]
    return generate_inputs_hash_from_proto(common_pb2.Inputs(literals=filtered))


def generate_task_identity_hash(task_template: tasks_pb2.TaskTemplate) -> str:
    """
    Hash of the task's run-independent identity: fully-qualified name, interface, and per-task
    code version (``metadata.discovery_version``, always populated by task_serde — the function
    body AST hash unless overridden).

    Deliberately excludes the container image, code-bundle version, resources, env vars, and
    plugin config, so that action names stay stable across code-only changes and recovery can
    match completed actions from a previous run. Editing a task's own function body changes its
    discovery_version and therefore its action name, so that task re-runs.

    :param task_template: The serialized task template.
    :return: A hexadecimal string representation of the hash.
    """
    interface_hash = generate_interface_hash(task_template.interface)
    version = task_template.metadata.discovery_version if task_template.HasField("metadata") else ""
    return hash_data(f"{task_template.id.name}-{interface_hash}-{version}")


def generate_trace_action_identity(func: Any) -> str:
    """
    Identity for a trace action: the function name plus a hash of the function body (AST), so an
    edited trace function re-executes on recovery instead of replaying a stale recorded result.
    Falls back to the bare name when the source is unavailable (e.g. REPL-defined functions).

    :param func: The traced function.
    :return: A stable identity string for the trace action.
    """
    name = getattr(func, "__name__", str(func))
    try:
        from flyte._cache.cache import VersionParameters
        from flyte._cache.policy_function_body import FunctionBodyPolicy

        version = FunctionBodyPolicy().get_version(salt="", params=VersionParameters(func=func))
    except Exception:
        return name
    return f"{name}-{version}"


def generate_sub_action_id_and_output_path(
    tctx: TaskContext,
    task_identity: str,
    inputs_hash: str,
    invoke_seq: int,
) -> Tuple[ActionID, str]:
    """
    Generate a sub-action ID and output path based on the current task context, task identity, and inputs.

    action name = hash(parent action name + inputs hash + task identity + invocation sequence [+ group])

    ``task_identity`` must be stable across runs for recovery to match completed actions: use
    `generate_task_identity_hash` for tasks and `generate_trace_action_identity` for
    trace actions. In particular it must not depend on the code-bundle version or container image.

    :param tctx:
    :param task_identity: Stable identity string for the task being invoked.
    :param inputs_hash: Consistent hash string of the inputs (filtered of cache-ignored vars if any).
    :param invoke_seq: The sequence number of the invocation, used to differentiate between multiple invocations.
    :return:
    """
    current_action_id = tctx.action
    current_output_path = tctx.run_base_dir
    sub_action_id = current_action_id.new_sub_action_from(
        task_hash=task_identity,
        input_hash=inputs_hash,
        group=tctx.group_data.name if tctx.group_data else None,
        task_call_seq=invoke_seq,
    )
    sub_run_output_path = storage.join(current_output_path, sub_action_id.name)
    return sub_action_id, sub_run_output_path
