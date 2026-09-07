"""
Action-name stability — remaining validation scope (ENG26-831).

Every test here asserts the behavior recovery REQUIRES: running the same workflow twice
must produce the identical set of action names. The three instabilities this file
originally confirmed (as strict xfails) are now fixed and their tests assert the fix:

* ``TestGroupSequencerInteraction`` — the group is folded into the sequencer call key
  (as it is into the action name), so identical calls made from different groups never
  share a counter and sequence assignment is independent of scheduling order.
* ``TestUnorderedInputs::test_untyped_dict_*`` — msgpack binary literals are
  re-encoded with recursively sorted map keys at hash time, so semantically-equal
  untyped dicts hash identically regardless of insertion order.
* ``TestUnorderedInputs::test_set_inputs_stable_across_processes`` — pickled
  set/frozenset literals carry a canonical content hash (pickle of the sorted
  elements), so the PYTHONHASHSEED-dependent pickle bytes no longer feed input hashing.
  Sets of unsortable elements remain unstable (documented recovery limitation).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import Dict, List

import pytest
from flyteidl2.core import literals_pb2, types_pb2

from flyte._internal.controllers import TaskCallSequencer
from flyte._internal.runtime import convert
from flyte.models import ActionID, GroupData, RawDataPath, TaskContext
from flyte.report import Report
from flyte.types import TypeEngine

TASK_IDENTITY = "task-identity-hash"
INPUTS_HASH = "inputs-hash"


def _make_tctx() -> TaskContext:
    return TaskContext(
        action=ActionID(name="parent", run_name="run1", project="p", domain="d"),
        run_base_dir="s3://bucket/metadata/p/d/run1",
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw/p/d/run1"),
        output_path="s3://bucket/output/p/d/run1",
        report=Report(name="test"),
    )


def _submit_name(
    sequencer: TaskCallSequencer,
    tctx: TaskContext,
    task_identity: str,
    inputs_hash: str,
    group: str | None = None,
) -> str:
    """Replica of the remote controller's naming path (_controller.py::_submit):
    call_key folds in every name component — task identity, inputs hash, and group."""
    call_key = f"{task_identity}:{inputs_hash}"
    if group:
        call_key = f"{call_key}:{group}"
    seq = sequencer.next_seq(call_key, tctx.action.name)
    call_tctx = tctx.replace(group_data=GroupData(group)) if group else tctx
    sub_id, _ = convert.generate_sub_action_id_and_output_path(call_tctx, task_identity, inputs_hash, seq)
    return sub_id.name


def _simulate_run(calls: list[tuple[str, str, str | None]]) -> set[str]:
    """Simulate one run: submit *calls* (task_identity, inputs_hash, group) in arrival
    order and return the resulting action-name set."""
    sequencer = TaskCallSequencer()
    tctx = _make_tctx()
    return {_submit_name(sequencer, tctx, ti, ih, g) for ti, ih, g in calls}


class TestGroupSequencerInteraction:
    """The group is folded into both the name and the sequencer call key, so identical
    calls made from different groups never race for sequence numbers."""

    def test_same_call_in_two_groups_scheduling_order_insensitive(self):
        """Same task + same inputs invoked from two different groups (independent async
        branches): the name set must not depend on which branch reaches the controller
        first."""
        run1 = _simulate_run([(TASK_IDENTITY, INPUTS_HASH, "group_a"), (TASK_IDENTITY, INPUTS_HASH, "group_b")])
        run2 = _simulate_run([(TASK_IDENTITY, INPUTS_HASH, "group_b"), (TASK_IDENTITY, INPUTS_HASH, "group_a")])
        assert run1 == run2

    def test_grouped_and_ungrouped_call_scheduling_order_insensitive(self):
        """Same task + same inputs invoked once inside a group and once outside."""
        run1 = _simulate_run([(TASK_IDENTITY, INPUTS_HASH, "group_a"), (TASK_IDENTITY, INPUTS_HASH, None)])
        run2 = _simulate_run([(TASK_IDENTITY, INPUTS_HASH, None), (TASK_IDENTITY, INPUTS_HASH, "group_a")])
        assert run1 == run2

    def test_two_named_maps_over_same_items_scheduling_order_insensitive(self):
        """Two concurrent flyte.map calls (distinct group_name) mapping the same task
        over the same items: per-shard names must not depend on how the two maps
        interleave."""
        items = ["item_hash_1", "item_hash_1"]  # duplicate items across the two maps
        map_a = [(TASK_IDENTITY, ih, "t_map_a") for ih in items]
        map_b = [(TASK_IDENTITY, ih, "t_map_b") for ih in items]
        # run 1: map_a fully first; run 2: interleaved starting with map_b
        run1 = _simulate_run(map_a + map_b)
        run2 = _simulate_run([map_b[0], map_a[0], map_b[1], map_a[1]])
        assert run1 == run2


class TestMapFanOut:
    """Fan-out within a single map/group (single counter per (identity, inputs))."""

    def test_distinct_items_stable_regardless_of_completion_order(self):
        """Distinct items never share a counter → order-insensitive."""
        items = ["item_hash_1", "item_hash_2", "item_hash_3"]
        run1 = _simulate_run([(TASK_IDENTITY, ih, "t_map") for ih in items])
        run2 = _simulate_run([(TASK_IDENTITY, ih, "t_map") for ih in reversed(items)])
        assert run1 == run2

    def test_duplicate_items_within_one_map_interchangeable(self):
        """Duplicate items in one map share a counter, but within a single group the
        resulting names are byte-identical calls — the name SET is stable."""
        calls = [(TASK_IDENTITY, "item_hash_1", "t_map")] * 3 + [(TASK_IDENTITY, "item_hash_2", "t_map")]
        run1 = _simulate_run(calls)
        run2 = _simulate_run(list(reversed(calls)))
        assert run1 == run2


class TestTaskCallSequencer:
    """Direct unit tests for the sequencer (previously untested)."""

    def test_different_call_keys_do_not_share_counters(self):
        s = TaskCallSequencer()
        assert s.next_seq("id1:inputs_a", "parent") == 1
        assert s.next_seq("id1:inputs_b", "parent") == 1
        assert s.next_seq("id2:inputs_a", "parent") == 1

    def test_same_call_key_increments(self):
        s = TaskCallSequencer()
        assert s.next_seq("id1:inputs_a", "parent") == 1
        assert s.next_seq("id1:inputs_a", "parent") == 2

    def test_counters_scoped_per_parent_action(self):
        s = TaskCallSequencer()
        assert s.next_seq("id1:inputs_a", "parent1") == 1
        assert s.next_seq("id1:inputs_a", "parent2") == 1

    def test_clear_resets_parent(self):
        s = TaskCallSequencer()
        s.next_seq("id1:inputs_a", "parent1")
        s.clear("parent1")
        assert s.next_seq("id1:inputs_a", "parent1") == 1


class TestStructuralChanges:
    """Loops / conditionals: since counters are keyed by (identity + inputs), only
    byte-identical repeated calls share a counter — inserting or removing a call with
    different inputs must not shift anyone else's name."""

    def test_appending_an_identical_call_preserves_existing_names(self):
        """Loop grows from 2 to 3 iterations over the same inputs: the first two names
        must be unchanged (recovery reuses them; only the new call runs)."""
        seq_run1 = TaskCallSequencer()
        seq_run2 = TaskCallSequencer()
        tctx = _make_tctx()
        run1 = [_submit_name(seq_run1, tctx, TASK_IDENTITY, INPUTS_HASH) for _ in range(2)]
        run2 = [_submit_name(seq_run2, tctx, TASK_IDENTITY, INPUTS_HASH) for _ in range(3)]
        assert run2[:2] == run1

    def test_inserting_a_different_call_does_not_shift_others(self):
        """A new conditional branch adds a call with different inputs between two
        existing identical calls — the existing calls' names must not change."""
        run1 = _simulate_run([(TASK_IDENTITY, INPUTS_HASH, None), (TASK_IDENTITY, INPUTS_HASH, None)])
        run2_calls = [
            (TASK_IDENTITY, INPUTS_HASH, None),
            (TASK_IDENTITY, "new_branch_inputs", None),
            (TASK_IDENTITY, INPUTS_HASH, None),
        ]
        run2 = _simulate_run(run2_calls)
        assert run1 <= run2


class TestUnorderedInputs:
    """Input hashing must be insensitive to semantically-irrelevant ordering."""

    @pytest.mark.asyncio
    async def test_untyped_dict_insertion_order_stable(self):
        """Untyped dicts serialize via msgpack (insertion-ordered): two equal dicts
        built in different key orders must still hash identically."""
        lt = TypeEngine.to_literal_type(dict)
        lit_ab = await TypeEngine.to_literal({"a": 1, "b": 2}, dict, lt)
        lit_ba = await TypeEngine.to_literal({"b": 2, "a": 1}, dict, lt)
        assert convert.generate_inputs_repr_for_literal(lit_ab) == convert.generate_inputs_repr_for_literal(lit_ba)

    @pytest.mark.asyncio
    async def test_typed_dict_insertion_order_stable(self):
        """Dict[str, T] becomes a map literal whose keys are sorted at hash time."""
        t = Dict[str, int]
        lt = TypeEngine.to_literal_type(t)
        lit_ab = await TypeEngine.to_literal({"a": 1, "b": 2}, t, lt)
        lit_ba = await TypeEngine.to_literal({"b": 2, "a": 1}, t, lt)
        assert convert.generate_inputs_repr_for_literal(lit_ab) == convert.generate_inputs_repr_for_literal(lit_ba)

    def test_set_inputs_stable_across_processes(self):
        """Set[str] inputs fall back to pickle in set iteration order, which depends on
        PYTHONHASHSEED — two runs (two interpreter processes) of the same workflow must
        still produce the same inputs hash."""
        script = textwrap.dedent(
            """
            import asyncio, hashlib
            from typing import Set
            from flyte._internal.runtime import convert
            from flyte.types import TypeEngine

            async def main():
                st = Set[str]
                lt = TypeEngine.to_literal_type(st)
                lit = await TypeEngine.to_literal(
                    {"alpha", "bravo", "charlie", "delta", "echo"}, st, lt
                )
                print(hashlib.md5(convert.generate_inputs_repr_for_literal(lit)).hexdigest())

            asyncio.run(main())
            """
        )

        def run_with_seed(seed: str) -> str:
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
            return out.stdout.strip().splitlines()[-1]

        assert run_with_seed("1") == run_with_seed("42")

    @pytest.mark.asyncio
    async def test_list_order_is_significant(self):
        """Sanity check (not an instability): list order is data — different orders are
        different inputs and must hash differently."""
        t = List[int]
        lt = TypeEngine.to_literal_type(t)
        lit_12 = await TypeEngine.to_literal([1, 2], t, lt)
        lit_21 = await TypeEngine.to_literal([2, 1], t, lt)
        assert convert.generate_inputs_repr_for_literal(lit_12) != convert.generate_inputs_repr_for_literal(lit_21)


def _blob_literal(uri: str, hash_value: str = "") -> literals_pb2.Literal:
    lit = literals_pb2.Literal(
        scalar=literals_pb2.Scalar(
            blob=literals_pb2.Blob(
                metadata=literals_pb2.BlobMetadata(
                    type=types_pb2.BlobType(format="", dimensionality=types_pb2.BlobType.SINGLE)
                ),
                uri=uri,
            )
        )
    )
    if hash_value:
        lit.hash = hash_value
    return lit


class TestOffloadedLiteralUris:
    """File/Dir/DataFrame literals embed URIs containing the source run name; recovery
    correctness depends on how those URIs feed the input hash."""

    def test_precomputed_hash_wins_over_uri(self):
        """A literal carrying a content hash must hash the same regardless of which
        run's URI it points at — recovered upstream output (old URI) then matches."""
        lit_run1 = _blob_literal("s3://bucket/outputs/run1/file", hash_value="content-hash")
        lit_run2 = _blob_literal("s3://bucket/outputs/run2/file", hash_value="content-hash")
        assert convert.generate_inputs_repr_for_literal(lit_run1) == convert.generate_inputs_repr_for_literal(lit_run2)

    def test_uri_change_without_hash_changes_input_hash(self):
        """Without a content hash the URI is the identity: a re-run upstream (new URI)
        must invalidate downstream (consistent rerun cascade, documented in ENG26-1042)."""
        lit_run1 = _blob_literal("s3://bucket/outputs/run1/file")
        lit_run2 = _blob_literal("s3://bucket/outputs/run2/file")
        assert convert.generate_inputs_repr_for_literal(lit_run1) != convert.generate_inputs_repr_for_literal(lit_run2)
