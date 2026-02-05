"""
Tests for generate_task_call_sequence with .override() calls.

Demonstrates that using id(task_obj) to key the sequence counter is unreliable
when .override() creates a new object each time via dataclasses.replace().
This causes duplicate action IDs when the same task is called in a loop with
.override(), because each override gets task_call_seq=1 if its id() differs
from the previous override's id().

Combined with a stale completion event in ActionCache (remove() does not clean
up _completion_events), duplicate action names cause
"Task X did not return an output path, but the task has outputs defined."
"""

import hashlib
from collections import defaultdict

import flyte
from flyte._utils.helpers import base36_encode
from flyte.models import ActionID

env = flyte.TaskEnvironment("test")


@env.task
async def parse_entity_extraction(
    short_name: str,
    value: int,
) -> str:
    return f"{short_name}-{value}"


# ---------------------------------------------------------------------------
# Reproduce the two SDK functions under test without triggering the
# flyteidl2.auth import chain that is broken in this environment.
# ---------------------------------------------------------------------------


def generate_task_call_sequence(
    task_obj: object,
    sequencer: dict[int, int],
) -> int:
    """
    Exact copy of RemoteController.generate_task_call_sequence logic.
    See: src/flyte/_internal/controllers/remote/_controller.py:141
    """
    current_task_id = id(task_obj)
    v = sequencer[current_task_id]
    new_seq = v + 1
    sequencer[current_task_id] = new_seq
    return new_seq


def make_action_name(parent_name: str, input_hash: str, task_hash: str, seq: int) -> str:
    """
    Exact copy of ActionID.new_sub_action_from logic.
    See: src/flyte/models.py:67
    """
    components = f"{parent_name}-{input_hash}-{task_hash}-{seq}"
    digest = hashlib.md5(components.encode()).digest()
    return base36_encode(digest)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateTaskCallSequenceWithOverride:
    """
    Stress tests for generate_task_call_sequence when used with .override().

    override() uses dataclasses.replace() which creates a new Python object each
    time. generate_task_call_sequence keys its counter on id(task_obj), which is
    the CPython memory address. When id() differs across override calls (common
    with interleaved allocations in async code), the counter resets to 1 on every
    call, producing duplicate action names.
    """

    def test_override_creates_new_object_each_time(self):
        """Verify that .override() returns a distinct Python object each call."""
        overrides = [parse_entity_extraction.override(short_name="same_name") for _ in range(10)]
        obj_ids = [id(o) for o in overrides]
        # All objects alive simultaneously -> all have distinct ids
        assert len(set(obj_ids)) == 10

    def test_sequence_resets_when_override_objects_alive_simultaneously(self):
        """
        When multiple override objects are alive at the same time (distinct id()s),
        generate_task_call_sequence returns 1 for each -- producing duplicate
        action names.
        """
        sequencer: dict[int, int] = defaultdict(int)

        overrides = [parse_entity_extraction.override(short_name="same_name") for _ in range(10)]
        seqs = [generate_task_call_sequence(o, sequencer) for o in overrides]

        # BUG: every override gets seq=1 because each has a unique id()
        assert all(s == 1 for s in seqs), f"Expected all sequences to be 1 (demonstrating the bug), got {seqs}"

    def test_duplicate_action_names_from_override_in_loop(self):
        """
        End-to-end: override in a loop with same inputs produces duplicate
        action IDs. This is the scenario that causes
        'did not return an output path' errors at runtime.
        """
        sequencer: dict[int, int] = defaultdict(int)
        parent_name = "a0"
        input_hash = "abc123"
        task_hash = "def456"

        # Keep all overrides alive to guarantee distinct id() values,
        # simulating what happens in async code where references are held.
        overrides = [parse_entity_extraction.override(short_name="same_name") for _ in range(20)]

        action_names = []
        for override in overrides:
            seq = generate_task_call_sequence(override, sequencer)
            name = make_action_name(parent_name, input_hash, task_hash, seq)
            action_names.append(name)

        unique_names = set(action_names)

        # BUG: all 20 iterations produce the same action name
        assert len(unique_names) == 1, f"Expected 1 unique name (all duplicates), got {len(unique_names)}"

    def test_sequence_increments_for_same_object_identity(self):
        """
        When the same task object (not an override) is used repeatedly, the
        sequence increments correctly because id() is stable.
        """
        sequencer: dict[int, int] = defaultdict(int)

        seqs = [generate_task_call_sequence(parse_entity_extraction, sequencer) for _ in range(10)]

        assert seqs == list(range(1, 11))

    def test_interleaved_allocations_prevent_id_reuse(self):
        """
        Simulate what happens in real async code: allocations between loop
        iterations (logging, protobuf serialization, other coroutines) prevent
        CPython from reusing the same memory address for the next override
        object.
        """
        sequencer: dict[int, int] = defaultdict(int)
        noise = []  # hold references to prevent address reuse

        seqs = []
        for i in range(10):
            override = parse_entity_extraction.override(short_name="same_name")
            seq = generate_task_call_sequence(override, sequencer)
            seqs.append(seq)
            del override
            # Allocate an object of the same type/size to grab the freed address
            noise.append(parse_entity_extraction.override(short_name=f"noise-{i}"))

        # With interleaved allocations, id() is not reused -> seq stays at 1
        assert all(s == 1 for s in seqs), f"Expected all sequences to be 1 (id not reused), got {seqs}"

    def test_action_name_determinism(self):
        """
        Verify that ActionID.new_sub_action_from produces identical names when
        given identical components -- confirming that duplicate seq values lead
        to duplicate action names.
        """
        parent = ActionID(name="a0")
        input_hash = "abc123"
        task_hash = "def456"

        name1 = parent.new_sub_action_from(task_call_seq=1, task_hash=task_hash, input_hash=input_hash, group=None)
        name2 = parent.new_sub_action_from(task_call_seq=1, task_hash=task_hash, input_hash=input_hash, group=None)
        name3 = parent.new_sub_action_from(task_call_seq=2, task_hash=task_hash, input_hash=input_hash, group=None)

        assert name1.name == name2.name, "Same seq should produce same name"
        assert name1.name != name3.name, "Different seq should produce different name"
