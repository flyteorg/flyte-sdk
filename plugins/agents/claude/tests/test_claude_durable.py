"""Tests for Claude durable sessions — checkpoint-backed SessionStore + resume wiring.

Offline: a ``FakeCheckpoint`` stands in for ``flyte.Checkpoint`` (``save`` writes this
attempt's blob, ``load`` reads the previous attempt's), so we can exercise the
first-attempt-pins / retry-resumes behavior without a backend.
"""

import pathlib
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flyteplugins.agents.claude._durable import (
    CheckpointSessionStore,
    deterministic_session_id,
    wire_durable_session,
)

KEY = {"project_key": "p", "session_id": "s-123"}


def _ctx(run_name="run-1", name="act-1"):
    action = SimpleNamespace(run_name=run_name, name=name)
    return SimpleNamespace(action=action, task_action=None)


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


def test_deterministic_session_id_is_stable_and_valid_uuid():
    tc = _ctx()
    sid = deterministic_session_id(tc)
    assert sid == deterministic_session_id(tc)  # stable -> same across retries of the action
    assert str(uuid.UUID(sid)) == sid  # valid uuid the CLI will accept
    assert deterministic_session_id(_ctx(name="act-2")) != sid  # per-action


def test_deterministic_session_id_prefers_task_action():
    pinned = SimpleNamespace(run_name="r", name="real-task")
    via_task_action = SimpleNamespace(action=SimpleNamespace(run_name="r", name="trace-pseudo"), task_action=pinned)
    via_action = SimpleNamespace(action=pinned, task_action=None)
    assert deterministic_session_id(via_task_action) == deterministic_session_id(via_action)


@pytest.mark.asyncio
async def test_store_round_trips_across_attempts(tmp_path):
    entries = [{"type": "user", "uuid": "u1"}, {"type": "assistant", "uuid": "u2"}]
    first = CheckpointSessionStore(FakeCheckpoint(tmp_path / "att1"))
    assert await first.seed_from_prev() is False  # first attempt: nothing prior
    await first.append(KEY, entries)

    # retry: this attempt's checkpoint resumes from att1
    retry = CheckpointSessionStore(FakeCheckpoint(tmp_path / "att2", prev_dir=tmp_path / "att1"))
    assert await retry.seed_from_prev() is True
    assert await retry.load(KEY) == entries


@pytest.mark.asyncio
async def test_append_is_idempotent_by_uuid(tmp_path):
    store = CheckpointSessionStore(FakeCheckpoint(tmp_path / "a"))
    await store.append(KEY, [{"type": "user", "uuid": "u1"}])
    await store.append(KEY, [{"type": "user", "uuid": "u1"}])  # duplicate -> ignored
    await store.append(KEY, [{"type": "user", "uuid": "u2"}])
    assert await store.load(KEY) == [{"type": "user", "uuid": "u1"}, {"type": "user", "uuid": "u2"}]


@pytest.mark.asyncio
async def test_load_returns_none_for_unseen_key(tmp_path):
    store = CheckpointSessionStore(FakeCheckpoint(tmp_path / "a"))
    assert await store.load(KEY) is None


@pytest.mark.asyncio
async def test_wire_first_attempt_pins_session_id(tmp_path):
    from claude_agent_sdk import ClaudeAgentOptions

    tc = _ctx()
    tc.checkpoint = FakeCheckpoint(tmp_path / "att1")  # no prev -> fresh
    opts = ClaudeAgentOptions()
    with patch("flyte.ctx", return_value=tc):
        store = await wire_durable_session(opts, durable=True)

    assert store is not None
    assert opts.session_id == deterministic_session_id(tc)
    assert opts.resume is None
    assert opts.session_store is store


@pytest.mark.asyncio
async def test_wire_retry_sets_resume_and_seeds(tmp_path):
    from claude_agent_sdk import ClaudeAgentOptions

    sid = deterministic_session_id(_ctx())
    # a prior attempt persisted a session under that id
    prior = CheckpointSessionStore(FakeCheckpoint(tmp_path / "att1"))
    await prior.append({"session_id": sid}, [{"type": "user", "uuid": "u1"}])

    tc = _ctx()
    tc.checkpoint = FakeCheckpoint(tmp_path / "att2", prev_dir=tmp_path / "att1")
    opts = ClaudeAgentOptions()
    with patch("flyte.ctx", return_value=tc):
        store = await wire_durable_session(opts, durable=True)

    assert opts.resume == sid
    assert opts.session_id is None
    assert await store.load({"session_id": sid}) == [{"type": "user", "uuid": "u1"}]


@pytest.mark.asyncio
async def test_wire_is_noop_when_durable_false():
    from claude_agent_sdk import ClaudeAgentOptions

    opts = ClaudeAgentOptions()
    assert await wire_durable_session(opts, durable=False) is None
    assert opts.session_store is None


@pytest.mark.asyncio
async def test_wire_is_noop_without_task_context():
    from claude_agent_sdk import ClaudeAgentOptions

    opts = ClaudeAgentOptions()
    with patch("flyte.ctx", return_value=None):
        assert await wire_durable_session(opts, durable=True) is None
    assert opts.session_store is None
