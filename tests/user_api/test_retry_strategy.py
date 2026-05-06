import flyte
from flyte._retry import RetryStrategy


def test_retry_strategy_basic():
    r = RetryStrategy(count=3)
    assert r.count == 3


def test_retry_strategy_zero():
    r = RetryStrategy(count=0)
    assert r.count == 0


def test_retry_strategy_high_count():
    r = RetryStrategy(count=100)
    assert r.count == 100


def test_retry_strategy_importable():
    assert flyte.RetryStrategy is RetryStrategy
