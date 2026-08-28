"""Tests for durable harness sessions — checkpoint-backed session resume.

Offline: a ``FakeCheckpoint`` stands in for ``flyte.Checkpoint`` (``save`` writes
this attempt's blob, ``load`` reads the previous attempt's), so the
first-attempt-starts / retry-resumes behavior is exercised without a backend.
"""

import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flyteplugins.agents.deepseek._durable import (
    CheckpointSession,
    deterministic_session_id,
    restore,
    snapshot,
    wire_durable_session,
)


def _ctx(run_name="run-1", name="act-1", checkpoint=None):
    action = SimpleNamespace(run_name=run_name, name=name)
    return SimpleNamespace(action=action, task_action=None, checkpoint=checkpoint)


class FakeCheckpoint:
    """Mimics flyte.Checkpoint: ``save`` -> this attempt's dir, ``load`` -> previous."""

    def __init__(self, save_dir, prev_dir=None):
        self.save_dir = pathlib.Path(save_dir)
        self.prev_dir = pathlib.Path(prev_dir) if prev_dir else None

    async def save(self, data):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "payload").write_bytes(data if isinstance(data, bytes) else str(data).encode())

    async def load(self):
        if self.prev_dir is not None and (self.prev_dir / "payload").is_file():
            return self.prev_dir
        return None


def test_deterministic_session_id_is_stable_and_per_action():
    tc = _ctx()
    sid = deterministic_session_id(tc)
    assert sid == deterministic_session_id(tc)  # stable -> same across retries
    assert deterministic_session_id(_ctx(name="act-2")) != sid  # per-action


def test_deterministic_session_id_prefers_task_action():
    """Inside a @trace pseudo-action the id must still track the real task."""
    pinned = SimpleNamespace(run_name="r", name="real-task")
    via_task_action = SimpleNamespace(action=SimpleNamespace(run_name="r", name="trace-pseudo"), task_action=pinned)
    via_action = SimpleNamespace(action=pinned, task_action=None)
    assert deterministic_session_id(via_task_action) == deterministic_session_id(via_action)


def test_snapshot_and_restore_round_trip_a_session_tree(tmp_path):
    source = tmp_path / "sessions"
    (source / "nested").mkdir(parents=True)
    (source / "s1.jsonl").write_text('{"type":"assistant/message"}\n')
    (source / "nested" / "s2.jsonl").write_text('{"type":"turn/end"}\n')

    archive = snapshot(source)
    assert set(archive) == {"s1.jsonl", str(pathlib.Path("nested") / "s2.jsonl")}

    target = tmp_path / "restored"
    target.mkdir()
    assert restore(target, archive) is True
    assert (target / "s1.jsonl").read_text() == '{"type":"assistant/message"}\n'
    assert (target / "nested" / "s2.jsonl").read_text() == '{"type":"turn/end"}\n'


def test_snapshot_of_a_missing_root_is_empty(tmp_path):
    assert snapshot(tmp_path / "nope") == {}
    assert restore(tmp_path, {}) is False


def test_snapshot_skips_unreadable_entries(tmp_path):
    """Best-effort: a binary transcript must not fail the run that produced it."""
    source = tmp_path / "sessions"
    source.mkdir()
    (source / "good.jsonl").write_text("{}\n")
    (source / "binary.bin").write_bytes(b"\xff\xfe\x00")
    assert set(snapshot(source)) == {"good.jsonl"}


@pytest.mark.asyncio
async def test_session_round_trips_across_attempts(tmp_path):
    """Attempt 1 persists its transcript; attempt 2 seeds from it and reports resume."""
    first_root = tmp_path / "run1"
    first_root.mkdir()
    (first_root / "s.jsonl").write_text('{"turn": 1}\n')

    blob = tmp_path / "ckpt"
    await CheckpointSession(FakeCheckpoint(blob), "sid").persist(first_root)

    second_root = tmp_path / "run2"
    second_root.mkdir()
    resumed = CheckpointSession(FakeCheckpoint(tmp_path / "ckpt2", prev_dir=blob), "sid")
    assert await resumed.seed(second_root) is True
    assert resumed.resumed is True
    assert (second_root / "s.jsonl").read_text() == '{"turn": 1}\n'


@pytest.mark.asyncio
async def test_first_attempt_has_nothing_to_seed(tmp_path):
    session = CheckpointSession(FakeCheckpoint(tmp_path / "ckpt"), "sid")
    assert await session.seed(tmp_path) is False
    assert session.resumed is False


@pytest.mark.asyncio
async def test_persist_never_raises(tmp_path):
    """Durability is best-effort: a failing checkpoint must not break the run."""

    class BrokenCheckpoint(FakeCheckpoint):
        async def save(self, data):
            raise OSError("blob store unavailable")

    (tmp_path / "s.jsonl").write_text("{}\n")
    await CheckpointSession(BrokenCheckpoint(tmp_path / "ckpt"), "sid").persist(tmp_path)


@pytest.mark.asyncio
async def test_wire_pins_the_action_session_id(tmp_path):
    tc = _ctx(checkpoint=FakeCheckpoint(tmp_path / "ckpt"))
    with patch("flyte.ctx", return_value=tc):
        session = await wire_durable_session(tmp_path, durable=True)
    assert session is not None
    assert session.session_id == deterministic_session_id(tc)
    assert session.resumed is False


@pytest.mark.asyncio
async def test_wire_on_retry_seeds_the_prior_transcript(tmp_path):
    blob = tmp_path / "ckpt"
    first_root = tmp_path / "run1"
    first_root.mkdir()
    (first_root / "s.jsonl").write_text('{"turn": 1}\n')
    await CheckpointSession(FakeCheckpoint(blob), "sid").persist(first_root)

    second_root = tmp_path / "run2"
    second_root.mkdir()
    tc = _ctx(checkpoint=FakeCheckpoint(tmp_path / "ckpt2", prev_dir=blob))
    with patch("flyte.ctx", return_value=tc):
        session = await wire_durable_session(second_root, durable=True)

    assert session is not None
    assert session.resumed is True
    # The retry targets the same session the crashed attempt was building.
    assert session.session_id == deterministic_session_id(tc)
    assert (second_root / "s.jsonl").read_text() == '{"turn": 1}\n'


@pytest.mark.asyncio
async def test_wire_is_noop_when_durable_false(tmp_path):
    assert await wire_durable_session(tmp_path, durable=False) is None


@pytest.mark.asyncio
async def test_wire_is_noop_without_task_context(tmp_path):
    with patch("flyte.ctx", return_value=None):
        assert await wire_durable_session(tmp_path, durable=True) is None


@pytest.mark.asyncio
async def test_wire_is_noop_without_a_checkpoint(tmp_path):
    """Running locally there is no checkpoint; the run proceeds without resume."""
    with patch("flyte.ctx", return_value=_ctx(checkpoint=None)):
        assert await wire_durable_session(tmp_path, durable=True) is None
