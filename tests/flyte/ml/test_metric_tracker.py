"""Tests flyte.ml._metric_tracker"""

from __future__ import annotations

import pytest

from flyte.ml import MetricTracker


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        MetricTracker(mode="worse")  # type: ignore[arg-type]


def test_empty_tracker_returns_none() -> None:
    tracker = MetricTracker()
    assert tracker.count == 0
    assert tracker.last is None
    assert tracker.mean is None
    assert tracker.min is None
    assert tracker.max is None
    assert tracker.best is None
    assert tracker.best_step == -1


def test_single_update() -> None:
    tracker = MetricTracker(mode="min")
    tracker.update(1.0)
    assert tracker.count == 1
    assert tracker.last == 1.0
    assert tracker.mean == 1.0
    assert tracker.min == 1.0
    assert tracker.max == 1.0
    assert tracker.best == 1.0
    assert tracker.best_step == 0


def test_running_stats_across_updates() -> None:
    tracker = MetricTracker(mode="min")
    for value in [3.0, 1.0, 2.0]:
        tracker.update(value)

    assert tracker.count == 3
    assert tracker.last == 2.0
    assert tracker.mean == pytest.approx(2.0)
    assert tracker.min == 1.0
    assert tracker.max == 3.0
    assert tracker.history == [3.0, 1.0, 2.0]


def test_min_mode_tracks_lowest_as_best() -> None:
    tracker = MetricTracker(mode="min")
    tracker.update(1.0)
    tracker.update(0.5)
    tracker.update(0.8)
    assert tracker.best == 0.5
    assert tracker.best_step == 1


def test_max_mode_tracks_highest_as_best() -> None:
    tracker = MetricTracker(mode="max")
    tracker.update(0.5)
    tracker.update(0.9)
    tracker.update(0.7)
    assert tracker.best == 0.9
    assert tracker.best_step == 1


def test_best_unchanged_when_no_improvement() -> None:
    tracker = MetricTracker(mode="min")
    tracker.update(1.0)
    tracker.update(2.0)
    tracker.update(3.0)
    assert tracker.best == 1.0
    assert tracker.best_step == 0


def test_reset_clears_state() -> None:
    tracker = MetricTracker(mode="min")
    tracker.update(1.0)
    tracker.update(0.5)
    tracker.reset()
    assert tracker.count == 0
    assert tracker.history == []
    assert tracker.best is None
    assert tracker.best_step == -1
