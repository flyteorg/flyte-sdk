from datetime import timedelta

import pytest

import flyte


def test_reuse_policy_happy():
    r = flyte.ReusePolicy(
        replicas=(1, 10),
        idle_ttl=300,
        scaledown_ttl=600,
        concurrency=5,
    )
    assert r.replicas == (1, 10)
    assert r.idle_ttl == timedelta(seconds=300)
    assert r.scaledown_ttl == timedelta(seconds=600)
    assert r.concurrency == 5


def test_reuse_policy_minimum_replicas():
    r = flyte.ReusePolicy(
        replicas=2,
        idle_ttl=300,
        concurrency=5,
    )
    assert r.replicas == (2, 2)
    assert r.idle_ttl == timedelta(seconds=300)
    assert r.concurrency == 5


def test_reuse_policy_scaledown_with_no_min_max():
    with pytest.raises(ValueError):
        flyte.ReusePolicy(
            replicas=2,
            idle_ttl=300,
            scaledown_ttl=600,
            concurrency=5,
        )


def test_reuse_policy_low_replicas_and_concurrency():
    r = flyte.ReusePolicy(
        replicas=(1, 1),
        idle_ttl=300,
        concurrency=1,
    )
    assert r.replicas == (1, 1)
    assert r.idle_ttl == timedelta(seconds=300)
    assert r.concurrency == 1
    # This should not raise a warning in the test environment, but it would in production.
