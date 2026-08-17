"""Tests flyte.ml._gradient_accumulator"""

from __future__ import annotations

import pytest

from flyte.ml import GradientAccumulator


def test_invalid_accumulation_steps_raises() -> None:
    with pytest.raises(ValueError):
        GradientAccumulator(accumulation_steps=0)


def test_accumulation_steps_of_one_fires_every_step() -> None:
    acc = GradientAccumulator(accumulation_steps=1)
    assert acc.step()
    assert acc.step()
    assert acc.step()


def test_fires_only_on_the_nth_step() -> None:
    acc = GradientAccumulator(accumulation_steps=4)
    assert not acc.step()  # 1
    assert not acc.step()  # 2
    assert not acc.step()  # 3
    assert acc.step()  # 4 -- fires


def test_counter_resets_after_firing() -> None:
    acc = GradientAccumulator(accumulation_steps=2)
    assert not acc.step()  # 1
    assert acc.step()  # 2 -- fires, resets
    assert not acc.step()  # 1 again
    assert acc.step()  # 2 -- fires again


def test_is_accumulating_reflects_pending_microbatches() -> None:
    acc = GradientAccumulator(accumulation_steps=3)
    assert not acc.is_accumulating
    acc.step()
    assert acc.is_accumulating
    acc.step()
    assert acc.is_accumulating
    acc.step()  # fires, resets
    assert not acc.is_accumulating


def test_reset_clears_counter() -> None:
    acc = GradientAccumulator(accumulation_steps=4)
    acc.step()
    acc.step()
    assert acc.micro_step == 2
    acc.reset()
    assert acc.micro_step == 0
    assert not acc.is_accumulating


def test_many_cycles_fire_at_correct_multiples() -> None:
    acc = GradientAccumulator(accumulation_steps=3)
    fired_at = [i for i in range(1, 13) if acc.step()]
    assert fired_at == [3, 6, 9, 12]
