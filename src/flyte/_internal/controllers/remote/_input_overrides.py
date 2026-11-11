"""Action overrides for a fork (a recovery run with changed inputs or code on some task actions).

The fork's client points every action of such a run at a file of overrides through the reserved
`_u_action_overrides` key of the root action's Inputs.context, which every descendant inherits
(reusable containers included, since it travels with each action's inputs rather than the pod).
When a controller is about to enqueue a child whose name is a key in that file, it reads the
child's override inputs from their path in the data plane, overlays them, by input name, on the
inputs it computed for the child, and runs the child from its replacement code bundle, if it has
one. The child keeps the name computed from the original inputs and code, so the key matches; the
client lists it in Recover.force_rerun_actions, so it re-runs rather than being recovered.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Callable, Optional, cast

from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

import flyte.errors
import flyte.storage as storage
from flyte._logging import logger
from flyte.models import CodeBundle

#: Attempts to read the overrides file, or an action's override inputs, before failing the
#: parent. A failed read fails the parent pod, so ride out transient storage errors.
_LOAD_ATTEMPTS = 3
_LOAD_BACKOFF_S = 0.5

#: Overrides files kept loaded per process. A reusable container serves actions of many runs, so
#: the files are keyed by URI; each is small (it only names paths), so a few dozen is plenty.
_MAX_CACHED_FILES = 32
_by_uri: "OrderedDict[str, ActionOverrides]" = OrderedDict()


def overrides_for(uri: Optional[str]) -> Optional["ActionOverrides"]:
    """The overrides of the run whose actions carry uri in their context, or None without one."""
    if not uri:
        return None
    overrides = _by_uri.get(uri)
    if overrides is None:
        overrides = _by_uri[uri] = ActionOverrides(uri)
        while len(_by_uri) > _MAX_CACHED_FILES:
            _by_uri.popitem(last=False)
    else:
        _by_uri.move_to_end(uri)
    return overrides


class ActionOverrides:
    """Lazily loaded action overrides of one fork, from the overrides file at uri.

    The file, which only names each overridden action's inputs path and code bundle and the
    overridden actions' ancestors, is read on the first child enqueued under it. An action's
    override inputs are read from their path when that action is enqueued, so a pod only reads the
    inputs of the actions it launches.
    """

    def __init__(self, uri: Optional[str] = None):
        self._uri = uri or None
        # Action name -> URI of its serialized override Inputs; None until the file is read.
        self._input_uris: Optional[dict[str, str]] = None
        # Action name -> ActionCodeBundle; loaded with _input_uris.
        self._code_bundles: dict[str, Any] = {}
        # Ancestors of the overridden actions (excluding a0); loaded with _input_uris.
        self._ancestors: frozenset[str] = frozenset()

    @property
    def enabled(self) -> bool:
        return self._uri is not None

    async def _read(self, uri: str, what: str, new_message: Callable[[], Any]) -> Any:
        """Parse the protobuf at uri into a new_message(), retrying transient storage errors."""
        for attempt in range(1, _LOAD_ATTEMPTS + 1):
            try:
                data = b"".join([chunk async for chunk in storage.get_stream(uri)])
                message = new_message()
                message.ParseFromString(data)
                return message
            except Exception as e:
                if attempt < _LOAD_ATTEMPTS:
                    logger.warning(f"Reading {what} from {uri} failed (attempt {attempt}): {e}")
                    await asyncio.sleep(_LOAD_BACKOFF_S * attempt)
                    continue
                raise flyte.errors.RuntimeSystemError(
                    "ActionOverridesUnavailableError",
                    f"Could not read {what} from {uri}: {e}. The run will not fall back to the original inputs.",
                ) from e
        raise AssertionError("unreachable")

    async def _load(self) -> dict[str, str]:
        # No lock: the controller submits from more than one event loop, and two concurrent
        # first reads of the same small file are harmless.
        if self._input_uris is not None:
            return self._input_uris
        if self._uri is None:  # not enabled: apply() returns before loading
            return {}
        if not hasattr(common_pb2, "ActionOverridesData"):
            # The image is fixed for every attempt, so fail without _read's retries or the
            # action's own.
            raise flyte.errors.NonRecoverableError(
                "This run has action overrides, but the flyteidl2 installed in this task's image predates "
                "them. Upgrade flyte (and flyteidl2) in the image to fork this run with overrides.",
                code="ActionOverridesUnsupportedError",
            )
        # ActionOverridesData is newer than the pinned flyteidl2 stubs.
        parsed = await self._read(
            self._uri,
            "the action input overrides for this run",
            lambda: cast(Any, common_pb2).ActionOverridesData(),
        )
        self._code_bundles = dict(getattr(parsed, "code_bundles", {}))
        self._ancestors = frozenset(getattr(parsed, "ancestor_actions", ()))
        self._input_uris = dict(parsed.input_uris)
        logger.info(
            f"Loaded action overrides from {self._uri}: inputs for {sorted(self._input_uris)}, "
            f"code for {sorted(self._code_bundles)}, ancestors {sorted(self._ancestors)}"
        )
        return self._input_uris

    async def is_ancestor(self, action_name: str) -> bool:
        """Whether action_name is an ancestor of an overridden action.

        Its inputs are unchanged, so it must run without a cache key: a cache hit would skip the
        subtree that holds the override, and a write would store the new result under the old key.
        """
        if not self.enabled:
            return False
        await self._load()
        return action_name in self._ancestors

    async def has_code_bundle(self, action_name: str) -> bool:
        """Whether action_name runs from a replacement code bundle."""
        if not self.enabled:
            return False
        await self._load()
        return action_name in self._code_bundles

    async def code_bundle(
        self, action_name: str, task_name: str, parent_bundle: Optional[CodeBundle]
    ) -> Optional[CodeBundle]:
        """The code bundle action_name runs instead of parent_bundle (this pod's), or None if it has none.

        The replacement keeps the parent's inflation destination, so the child's module paths
        resolve as they would from the parent's bundle. Raises a user error when this pod has no
        source bundle to replace: an image-only or pickled (interactive) run.
        """
        if not await self.has_code_bundle(action_name):
            return None
        override = self._code_bundles[action_name]
        if parent_bundle is None or parent_bundle.pkl or not parent_bundle.tgz:
            raise flyte.errors.RuntimeUserError(
                "ActionCodeOverrideUnsupportedError",
                f"Action {action_name} (task {task_name}) has a code override, but its parent runs without a "
                f"source code bundle (its code is in the image or pickled), so there is no bundle to replace.",
            )
        logger.info(f"Applied code override to action {action_name}: code bundle {override.tgz}")
        return CodeBundle(
            computed_version=override.version or parent_bundle.computed_version,
            destination=parent_bundle.destination,
            tgz=override.tgz,
        )

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
        input_uri = (await self._load()).get(action_name)
        if input_uri is None:
            return None
        override = await self._read(input_uri, f"the override inputs of action {action_name}", common_pb2.Inputs)

        types = {v.key: v.value.type for v in interface.inputs.variables}
        for lit in override.literals:
            if lit.name not in types:
                raise flyte.errors.RuntimeUserError(
                    "ActionOverrideMismatchError",
                    f"Override for action {action_name} sets input '{lit.name}', which task {task_name} "
                    f"no longer has. Inputs: {', '.join(types) or '<none>'}.",
                )
            if not literal_matches_type(lit.value, types[lit.name]):
                raise flyte.errors.RuntimeUserError(
                    "ActionOverrideMismatchError",
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
