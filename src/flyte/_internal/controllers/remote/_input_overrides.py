"""Action input overrides for a fork (a recovery run with changed inputs on some task actions).

The server points every task pod of such a run at a file of overrides through the reserved
`_U_ACTION_INPUT_OVERRIDES` env var. When this pod's controller is about to enqueue a child whose
name is a key in that file, it overlays the override values, by input name, on the inputs it
computed for the child. The child keeps the name computed from the original inputs, so the key
matches; the server re-runs it rather than recovering it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, cast

from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

import flyte.errors
import flyte.storage as storage
from flyte._constants import ACTION_INPUT_OVERRIDES_ENV_VAR as INPUT_OVERRIDES_ENV_VAR
from flyte._logging import logger

#: Attempts to read the overrides file before failing the parent. A failed read fails every
#: parent pod of the run that enqueues a child, so ride out transient storage errors.
_LOAD_ATTEMPTS = 3
_LOAD_BACKOFF_S = 0.5


class ActionInputOverrides:
    """Lazily loaded action input overrides for the current run.

    The file is read at most once per pod, on the first child this pod enqueues, and only when
    the env var is set (a fork with overrides). Whether a child's name is a key is only known
    after the read.
    """

    def __init__(self, uri: Optional[str] = None):
        self._uri = uri if uri is not None else os.environ.get(INPUT_OVERRIDES_ENV_VAR) or None
        self._overrides: Optional[dict[str, common_pb2.Inputs]] = None

    @property
    def enabled(self) -> bool:
        return self._uri is not None

    async def _load(self) -> dict[str, common_pb2.Inputs]:
        # No lock: the controller submits from more than one event loop, and two concurrent
        # first reads of the same small file are harmless.
        if self._overrides is not None:
            return self._overrides
        if self._uri is None:  # not enabled: apply() returns before loading
            return {}
        uri = self._uri
        for attempt in range(1, _LOAD_ATTEMPTS + 1):
            try:
                data = b"".join([chunk async for chunk in storage.get_stream(uri)])
                # ActionInputOverridesData is newer than the pinned flyteidl2 stubs.
                parsed = cast(Any, common_pb2).ActionInputOverridesData()
                parsed.ParseFromString(data)
                break
            except Exception as e:
                if attempt < _LOAD_ATTEMPTS:
                    logger.warning(f"Reading action input overrides from {self._uri} failed (attempt {attempt}): {e}")
                    await asyncio.sleep(_LOAD_BACKOFF_S * attempt)
                    continue
                raise flyte.errors.RuntimeSystemError(
                    "ActionInputOverridesUnavailableError",
                    f"Could not read the action input overrides for this run from {self._uri}: {e}. "
                    f"The run will not fall back to the original inputs.",
                ) from e
        self._overrides = dict(parsed.overrides)
        logger.info(f"Loaded action input overrides for {sorted(self._overrides)}")
        return self._overrides

    async def apply(
        self,
        action_name: str,
        task_name: str,
        proto_inputs: common_pb2.Inputs,
        interface: interface_pb2.TypedInterface,
    ) -> Optional[common_pb2.Inputs]:
        """Return proto_inputs with action_name's overrides applied, or None if it has none.

        Raises a user error if an override names an input the task no longer has, or carries a
        value of a different kind than the input's type (e.g. a code change retyped it).
        """
        if not self.enabled:
            return None
        overrides = await self._load()
        override = overrides.get(action_name)
        if override is None:
            return None

        types = {v.key: v.value.type for v in interface.inputs.variables}
        for lit in override.literals:
            if lit.name not in types:
                raise flyte.errors.RuntimeUserError(
                    "ActionInputOverrideMismatchError",
                    f"Override for action {action_name} sets input '{lit.name}', which task {task_name} "
                    f"no longer has. Inputs: {', '.join(types) or '<none>'}.",
                )
            if not literal_matches_type(lit.value, types[lit.name]):
                raise flyte.errors.RuntimeUserError(
                    "ActionInputOverrideMismatchError",
                    f"Override for action {action_name} sets input '{lit.name}' to a value that does not "
                    f"match its type in task {task_name}.",
                )

        values = {lit.name: lit.value for lit in override.literals}
        merged = common_pb2.Inputs(context=proto_inputs.context)
        for lit in proto_inputs.literals:
            merged.literals.append(common_pb2.NamedLiteral(name=lit.name, value=values.pop(lit.name, lit.value)))
        # An input the computed inputs omitted (e.g. left to its default) still lands, in interface order.
        for name in types:
            if name in values:
                merged.literals.append(common_pb2.NamedLiteral(name=name, value=values.pop(name)))
        overridden = sorted(lit.name for lit in override.literals)
        logger.info(f"Applied input override to action {action_name} (inputs: {overridden})")
        return merged


_SIMPLE_SCALAR_FIELD = {
    types_pb2.INTEGER: "integer",
    types_pb2.FLOAT: "float_value",
    types_pb2.STRING: "string_value",
    types_pb2.BOOLEAN: "boolean",
    types_pb2.DATETIME: "datetime",
    types_pb2.DURATION: "duration",
}


def literal_matches_type(literal: literals_pb2.Literal, lt: types_pb2.LiteralType) -> bool:
    """A structural check that a literal is of the same kind as a literal type.

    It catches a retyped input (an int where a string is now expected, a scalar where a list is),
    not every incompatibility; anything it cannot judge is accepted and left to the task.
    """
    kind = lt.WhichOneof("type")
    if kind is None:
        return True
    if kind == "union_type":
        if literal.HasField("scalar") and literal.scalar.HasField("union"):
            literal = literal.scalar.union.value
        return any(literal_matches_type(literal, v) for v in lt.union_type.variants)
    if kind == "collection_type":
        if not literal.HasField("collection"):
            return False
        return all(literal_matches_type(item, lt.collection_type) for item in literal.collection.literals)
    if kind == "map_value_type":
        if not literal.HasField("map"):
            return False
        return all(literal_matches_type(v, lt.map_value_type) for v in literal.map.literals.values())
    if not literal.HasField("scalar"):
        # Offloaded literals and promises carry no inline scalar to compare.
        return literal.HasField("offloaded_metadata")
    scalar = literal.scalar
    if kind == "blob":
        return scalar.HasField("blob")
    if kind == "structured_dataset_type":
        return scalar.HasField("structured_dataset")
    if kind == "schema":
        return scalar.HasField("schema")
    if kind == "enum_type":
        return scalar.HasField("primitive") and scalar.primitive.WhichOneof("value") == "string_value"
    if kind == "simple":
        if lt.simple == types_pb2.NONE:
            return scalar.HasField("none_type")
        if lt.simple == types_pb2.STRUCT:
            return scalar.HasField("generic") or scalar.HasField("binary")
        if lt.simple == types_pb2.BINARY:
            return scalar.HasField("binary")
        field = _SIMPLE_SCALAR_FIELD.get(lt.simple)
        if field is None:
            return True
        return scalar.HasField("primitive") and scalar.primitive.WhichOneof("value") == field
    return True
