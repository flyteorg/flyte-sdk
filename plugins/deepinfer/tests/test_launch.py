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
