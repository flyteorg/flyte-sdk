from flyteidl.core import interface_pb2

from flyte._internal.runtime import convert
from flyte._protos.workflow import run_definition_pb2


class LocalTaskCache(object):
    """
    This class implements a persistent store able to cache the result of local task executions.
    """

    _cache: None  # Cache  # Should use the sqlite
    _initialized: bool = False

    @staticmethod
    async def initialize():
        """Initialize the cache."""
        LocalTaskCache._cache = None  # Use sqlalchemy here # Cache(CACHE_LOCATION)
        LocalTaskCache._initialized = True

    @staticmethod
    async def clear():
        if not LocalTaskCache._initialized:
            LocalTaskCache.initialize()
        # See how we can clear the sqlite through sqlalchemy
        LocalTaskCache._cache.clear()

    @staticmethod
    async def get(
        task_name: str,
        input_hash: str,
        proto_inputs: run_definition_pb2.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
    ) -> run_definition_pb2.Outputs | None:
        if not LocalTaskCache._initialized:
            LocalTaskCache.initialize()

        cached_output = LocalTaskCache._cache.get(
            convert.generate_cache_key_hash(
                task_name, input_hash, task_interface, cache_version, ignore_input_vars, proto_inputs
            )
        )

        if cached_output is None:
            return None

        return cached_output

        """
        # If the serialized object is a model file, first convert it back to a proto object (which will force it to
        # use the installed flyteidl proto messages) and then convert it to a model object. This will guarantee
        # that the object is in the correct format.
        # if isinstance(serialized_obj, ModelLiteralMap):
        #     return ModelLiteralMap.from_flyte_idl(ModelLiteralMap.to_flyte_idl(serialized_obj))
        elif isinstance(serialized_obj, bytes):
            # If it is a bytes object, then it is a serialized proto object.
            # We need to convert it to a model object first.o
            pb_literal_map = LiteralMap()
            pb_literal_map.ParseFromString(serialized_obj)
            return ModelLiteralMap.from_flyte_idl(pb_literal_map)
        else:
            raise ValueError(f"Unexpected object type {type(serialized_obj)}")
        """

    @staticmethod
    async def set(
        task_name: str,
        inputs_hash: str,
        proto_inputs: run_definition_pb2.Inputs,
        task_interface: interface_pb2.TypedInterface,
        cache_version: str,
        ignore_input_vars: list[str],
        value: run_definition_pb2.Outputs,
    ) -> None:
        if not LocalTaskCache._initialized:
            LocalTaskCache.initialize()
        LocalTaskCache._cache.set(
            convert.generate_cache_key_hash(
                task_name,
                inputs_hash,
                task_interface,
                cache_version,
                ignore_input_vars,
                proto_inputs,
            ),
            value.SerializeToString(),
        )
