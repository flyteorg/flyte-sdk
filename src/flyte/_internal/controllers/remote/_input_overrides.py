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

import functools
from dataclasses import dataclass
from typing import Iterable, Optional

from flyteidl2.task import common_pb2

import flyte.errors
import flyte.storage as storage
from flyte._internal.runtime import convert, io
from flyte._logging import logger
from flyte.models import CodeBundle, TaskContext


@dataclass(frozen=True)
class ChildOverrides:
    """What a fork's overrides change for one child action."""

    #: The inputs to store and hash for the child, or None to keep the ones computed for it.
    inputs: Optional[common_pb2.Inputs]
    #: The code bundle the child runs instead of its parent's, or None to keep the parent's.
    code_bundle: Optional[CodeBundle]
    #: Enqueue the child without a cache key. An ancestor of an overridden action keeps its inputs,
    #: so a hit would skip the subtree that holds the override, and a write would store the new
    #: result under the old key. A code override keeps the cache version of this pod's copy of the
    #: function, so a hit would return the original code's result.
    skip_cache: bool

    @property
    def applied(self) -> bool:
        """Whether the child's inputs or code changed, which the action records as override_applied."""
        return self.inputs is not None or self.code_bundle is not None


class _OverridesFile:
    """The overrides file at uri, parsed on first use."""

    def __init__(self, uri: str):
        self.uri = uri
        self._data: Optional[common_pb2.ActionOverridesData] = None

    async def load(self) -> common_pb2.ActionOverridesData:
        # No lock: the controller submits from more than one event loop, and two concurrent first
        # reads of the same small file are harmless.
        if self._data is None:
            try:
                data = common_pb2.ActionOverridesData()
                data.ParseFromString(b"".join([chunk async for chunk in storage.get_stream(self.uri)]))
            except Exception as e:
                raise _unavailable(f"the action overrides of this run from {self.uri}", e) from e
            logger.info(
                f"Loaded action overrides from {self.uri}: inputs for {sorted(data.input_uris)}, "
                f"code for {sorted(data.code_bundles)}, ancestors {sorted(data.ancestor_actions)}"
            )
            self._data = data
        return self._data


@functools.lru_cache(maxsize=32)
def _overrides_file(uri: str) -> _OverridesFile:
    """The overrides file at uri. Keyed by URI, since a reusable container serves many runs."""
    return _OverridesFile(uri)


def _unavailable(what: str, e: Exception) -> flyte.errors.RuntimeSystemError:
    return flyte.errors.RuntimeSystemError(
        "ActionOverridesUnavailableError",
        f"Could not read {what}: {e}. The run will not fall back to the original inputs.",
    )


async def child_overrides(
    tctx: TaskContext,
    action_name: str,
    task_name: str,
    proto_inputs: common_pb2.Inputs,
    input_names: Iterable[str],
    parent_bundle: Optional[CodeBundle] = None,
    *,
    task_ref: bool = False,
) -> Optional[ChildOverrides]:
    """What the run's fork overrides change for the child action_name, or None if nothing.

    None also outside a fork with overrides, which every other run is, at the cost of one lookup.
    proto_inputs are the inputs computed for the child and input_names those of its task; a code
    override replaces parent_bundle (this pod's). task_ref marks a child that runs a deployed task,
    whose code is fixed by its registration.
    """
    uri = (tctx.custom_context or {}).get(convert.ACTION_OVERRIDES_CONTEXT_KEY)
    if not uri:
        return None
    data = await _overrides_file(uri).load()

    inputs: Optional[common_pb2.Inputs] = None
    input_uri = data.input_uris.get(action_name)
    if input_uri:
        try:
            override = (await io.load_inputs(input_uri)).proto_inputs
        except Exception as e:
            raise _unavailable(f"the override inputs of action {action_name} from {input_uri}", e) from e
        # The fork client converts override values against the action's interface; a name the task
        # no longer has means its code changed since.
        names = set(input_names)
        unknown = sorted(lit.name for lit in override.literals if lit.name not in names)
        if unknown:
            raise flyte.errors.RuntimeUserError(
                "ActionOverrideMismatchError",
                f"Override for action {action_name} sets inputs {unknown}, which task {task_name} no longer "
                f"has. Inputs: {', '.join(sorted(names)) or '<none>'}.",
            )
        inputs = common_pb2.Inputs(
            literals=convert.overlay_named_literals(proto_inputs.literals, override.literals),
            context=proto_inputs.context,
        )
        logger.info(
            f"Applied input override to action {action_name} (inputs: {sorted(lit.name for lit in override.literals)})"
        )

    code_bundle: Optional[CodeBundle] = None
    code = data.code_bundles.get(action_name)
    if code is not None:
        if task_ref:
            raise flyte.errors.RuntimeUserError(
                "ActionCodeOverrideUnsupportedError",
                f"Action {action_name} runs the deployed task {task_name}, whose code is fixed by its "
                f"registration, so its code cannot be overridden in a fork.",
            )
        if parent_bundle is None or parent_bundle.pkl or not parent_bundle.tgz:
            raise flyte.errors.RuntimeUserError(
                "ActionCodeOverrideUnsupportedError",
                f"Action {action_name} (task {task_name}) has a code override, but its parent runs without a "
                f"source code bundle (its code is in the image or pickled), so there is no bundle to replace.",
            )
        # Keep the parent's inflation destination, so the child's module paths resolve as they would
        # from the parent's bundle.
        code_bundle = CodeBundle(
            computed_version=code.version or parent_bundle.computed_version,
            destination=parent_bundle.destination,
            tgz=code.tgz,
        )
        logger.info(f"Applied code override to action {action_name}: code bundle {code.tgz}")

    skip_cache = code_bundle is not None or action_name in data.ancestor_actions
    if inputs is None and not skip_cache:
        return None
    return ChildOverrides(inputs=inputs, code_bundle=code_bundle, skip_cache=skip_cache)
