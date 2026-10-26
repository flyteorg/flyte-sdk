"""Unit tests for the DeepInfer launch helpers.

The multi-process (`dp_size > 1`) paths require `spawn`, which is exercised only
when a child actually runs; these tests cover the in-process paths plus argument
construction without spawning GPU work.
"""

import os

from flyteplugins.deepinfer import _constants as const
from flyteplugins.deepinfer import launch_dp, run_dp, run_engines


def test_run_dp_single_in_process():
    seen = []
    run_dp(lambda size, rank: seen.append((size, rank)), dp_size=1)
    assert seen == [(1, 0)]


def test_run_engines_single_in_process():
    def worker(size, rank, out):
        out.append(f"rank-{rank}")

    assert run_engines(worker, dp_size=1) == ["rank-0"]


def test_run_engines_single_with_initializer():
    ran = []

    def init():
        ran.append(True)

    def worker(size, rank, out):
        out.append(str(rank))

    assert run_engines(worker, dp_size=1, initializer=init) == ["0"]
    assert ran == [True]


def test_launch_dp_single_defaults(monkeypatch):
    """With one DP engine, entry runs in-process and RANK/WORLD_SIZE defaults are set."""
    monkeypatch.delenv(const.RANK, raising=False)
    monkeypatch.delenv(const.WORLD_SIZE, raising=False)
    seen = []
    launch_dp(lambda size, rank: seen.append((size, rank)), dp_size=1)
    assert seen == [(1, 0)]
    assert os.environ[const.RANK] == "0"
    assert os.environ[const.WORLD_SIZE] == "1"


def test_run_dp_two_ranks_constraint(monkeypatch):
    """dp_size must split evenly across nodes."""
    monkeypatch.setenv(const.WORLD_SIZE, "2")
    try:
        run_dp(lambda s, r: None, dp_size=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for dp_size not divisible by WORLD_SIZE")


# Module-level so `spawn`ed children can unpickle them.
def _emit_rank(tag, dp_size, dp_rank, out):
    out.append(f"{tag}-{dp_rank}")
    out.extend([dp_rank * 10, dp_rank * 10 + 1])


def _set_marker():
    os.environ["DEEPINFER_TEST_MARKER"] = "set"


def _emit_marker(dp_size, dp_rank, out):
    out.append(os.environ.get("DEEPINFER_TEST_MARKER"))


def test_run_engines_spawn_collects_in_rank_order(monkeypatch):
    import functools

    monkeypatch.delenv(const.RANK, raising=False)
    monkeypatch.delenv(const.WORLD_SIZE, raising=False)
    got = run_engines(functools.partial(_emit_rank, "r"), dp_size=3)
    assert got == ["r-0", 0, 1, "r-1", 10, 11, "r-2", 20, 21]


def test_run_engines_spawn_with_initializer(monkeypatch):
    monkeypatch.delenv(const.RANK, raising=False)
    monkeypatch.delenv(const.WORLD_SIZE, raising=False)
    monkeypatch.delenv("DEEPINFER_TEST_MARKER", raising=False)
    assert run_engines(_emit_marker, dp_size=2, initializer=_set_marker) == ["set", "set"]


def test_run_engines_only_runs_local_ranks(monkeypatch):
    import functools

    monkeypatch.setenv(const.RANK, "1")
    monkeypatch.setenv(const.WORLD_SIZE, "2")
    got = run_engines(functools.partial(_emit_rank, "n"), dp_size=4)
    assert got == ["n-2", 20, 21, "n-3", 30, 31]
