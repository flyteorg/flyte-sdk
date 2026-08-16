"""Tests flyte.ml._early_stopping"""

from __future__ import annotations

import pytest

from flyte.ml import EarlyStopping


def test_invalid_patience_raises() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=1, mode="worse")  # type: ignore[arg-type]


def test_invalid_min_delta_raises() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=1, min_delta=-0.1)


def test_first_step_is_always_improvement() -> None:
    es = EarlyStopping(patience=2)
    stop = es.step(1.0)
    assert not stop
    assert es.best == 1.0
    assert es.best_step == 0
    assert es.num_bad_steps == 0


def test_min_mode_stops_after_patience_exhausted() -> None:
    es = EarlyStopping(patience=2, mode="min")
    assert not es.step(1.0)  # improvement (first value)
    assert not es.step(0.9)  # improvement
    assert not es.step(0.95)  # bad step 1 -- still < patience
    assert es.step(0.95)  # bad step 2 -- reaches patience
    assert es.should_stop
    assert es.best == 0.9
    assert es.best_step == 1


def test_max_mode_treats_higher_as_better() -> None:
    es = EarlyStopping(patience=1, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)  # improvement, resets bad steps
    assert es.step(0.6)  # not an improvement (needs to exceed best)
    assert es.should_stop
    assert es.best == 0.6


def test_min_delta_requires_meaningful_improvement() -> None:
    es = EarlyStopping(patience=1, mode="min", min_delta=0.1)
    assert not es.step(1.0)
    # 0.95 is lower but within min_delta, so it doesn't count as improvement.
    assert es.step(0.95)
    assert es.should_stop
    assert es.best == 1.0


def test_improvement_resets_bad_step_counter() -> None:
    es = EarlyStopping(patience=2, mode="min")
    es.step(1.0)
    es.step(1.1)  # bad step 1
    es.step(0.5)  # improvement, resets counter
    assert es.num_bad_steps == 0
    assert not es.should_stop
    es.step(0.6)  # bad step 1 again
    assert es.num_bad_steps == 1
    assert not es.should_stop


def test_reset_clears_state() -> None:
    es = EarlyStopping(patience=1, mode="min")
    es.step(1.0)
    es.step(2.0)
    assert es.should_stop
    es.reset()
    assert es.best is None
    assert es.best_step == -1
    assert es.num_bad_steps == 0
    assert not es.should_stop
