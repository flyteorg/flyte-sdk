import math
from datetime import timedelta

import pytest

import flyte
from flyte._retry import Backoff, RetryStrategy


def test_retry_strategy_basic():
    r = RetryStrategy(count=3)
    assert r.count == 3
    assert r.backoff is None


def test_retry_strategy_zero():
    r = RetryStrategy(count=0)
    assert r.count == 0


def test_retry_strategy_high_count():
    r = RetryStrategy(count=100)
    assert r.count == 100


def test_retry_strategy_with_backoff():
    b = Backoff(base=timedelta(seconds=10), factor=2.0, cap=timedelta(minutes=5))
    r = RetryStrategy(count=5, backoff=b)
    assert r.count == 5
    assert r.backoff is b


def test_retry_strategy_importable():
    assert flyte.RetryStrategy is RetryStrategy


def test_backoff_importable():
    assert flyte.Backoff is Backoff


def test_backoff_constant_factor_one():
    b = Backoff(base=timedelta(seconds=2))
    assert b.factor == 1.0
    assert b.cap is None
    assert b.compute_delay(0) == timedelta(seconds=2)
    assert b.compute_delay(5) == timedelta(seconds=2)


def test_backoff_with_factor_and_cap():
    b = Backoff(base=timedelta(seconds=10), factor=2.0, cap=timedelta(minutes=5))
    assert b.compute_delay(0) == timedelta(seconds=10)
    assert b.compute_delay(1) == timedelta(seconds=20)
    assert b.compute_delay(2) == timedelta(seconds=40)
    assert b.compute_delay(3) == timedelta(seconds=80)
    assert b.compute_delay(4) == timedelta(seconds=160)
    # 10 * 2**5 = 320s, but cap is 300s.
    assert b.compute_delay(5) == timedelta(minutes=5)
    assert b.compute_delay(20) == timedelta(minutes=5)


def test_backoff_zero_base():
    b = Backoff(base=timedelta(0))
    assert b.compute_delay(0) == timedelta(0)
    assert b.compute_delay(5) == timedelta(0)


def test_backoff_negative_base_rejected():
    with pytest.raises(ValueError, match="base must be >= 0"):
        Backoff(base=timedelta(seconds=-1))


def test_backoff_factor_below_one_rejected():
    with pytest.raises(ValueError, match="factor must be >= 1.0"):
        Backoff(base=timedelta(seconds=1), factor=0.5)


def test_backoff_negative_cap_rejected():
    with pytest.raises(ValueError, match="cap must be >= 0"):
        Backoff(base=timedelta(seconds=1), cap=timedelta(seconds=-1))


def test_backoff_factor_gt_one_requires_cap():
    with pytest.raises(ValueError, match="cap is required when factor > 1.0"):
        Backoff(base=timedelta(seconds=1), factor=2.0)


def test_backoff_nan_factor_rejected():
    with pytest.raises(ValueError, match="finite number"):
        Backoff(base=timedelta(seconds=1), factor=math.nan, cap=timedelta(seconds=1))


def test_backoff_inf_factor_rejected():
    with pytest.raises(ValueError, match="finite number"):
        Backoff(base=timedelta(seconds=1), factor=math.inf, cap=timedelta(seconds=1))


def test_backoff_compute_delay_negative_n_rejected():
    b = Backoff(base=timedelta(seconds=1))
    with pytest.raises(ValueError, match="n must be >= 0"):
        b.compute_delay(-1)


def test_backoff_is_hashable():
    # Frozen dataclass: usable in sets / as dict keys.
    b1 = Backoff(base=timedelta(seconds=1))
    b2 = Backoff(base=timedelta(seconds=1))
    assert hash(b1) == hash(b2)
    assert {b1, b2} == {b1}
