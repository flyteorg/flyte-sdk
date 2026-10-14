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
    r = flyte.ReusePolicy(
        replicas=2,
        idle_ttl=300,
        scaledown_ttl=600,
        concurrency=5,
    )
    assert r.replicas == (2, 2)
    assert r.idle_ttl == timedelta(seconds=300)
    assert r.scaledown_ttl == timedelta(seconds=600)
    assert r.concurrency == 5


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


# `replicas` was only shape-checked (an int, or a tuple of length two): the tuple's *elements* were
# never validated, and no spelling was range-checked. `concurrency` was not checked at all. Both feed
# protobuf uint32 fields in `reuse_policy_to_pb`, where a non-int died as
# `TypeError: 'str' object cannot be interpreted as an integer` -- SDK frames naming neither the field
# nor the task -- while a negative or zero value serialized *silently* and deployed a pool that could
# never run a task. `reuse_policy_to_pb`'s docstring already claims these accessors are "always
# well-defined" because of `__post_init__`; these pin that claim.


@pytest.mark.parametrize("bad", [(1, "3"), ("1", 3), (1.5, 3), (1, 3.0), (None, 3)])
def test_reuse_policy_rejects_non_int_replica_elements(bad):
    with pytest.raises(ValueError, match=r"replicas (min|max) must be an int"):
        flyte.ReusePolicy(replicas=bad)


@pytest.mark.parametrize("bad", [-1, -5])
def test_reuse_policy_rejects_negative_replicas(bad):
    with pytest.raises(ValueError, match=r"replicas (min|max) must be greater than or equal to 0"):
        flyte.ReusePolicy(replicas=bad)


def test_reuse_policy_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match=r"replicas min \(3\) must be less than or equal to max \(1\)"):
        flyte.ReusePolicy(replicas=(3, 1))


@pytest.mark.parametrize("bad", [1.5, "2", [2], None])
def test_reuse_policy_rejects_non_int_concurrency(bad):
    with pytest.raises(ValueError, match=r"concurrency must be an int"):
        flyte.ReusePolicy(concurrency=bad)


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_reuse_policy_rejects_concurrency_below_one(bad):
    with pytest.raises(ValueError, match=r"concurrency must be at least 1"):
        flyte.ReusePolicy(concurrency=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"replicas": 1},
        {"replicas": (0, 3)},  # scale-to-zero min stays legal
        {"replicas": (2, 2)},
        {"replicas": (1, 10), "concurrency": 5},
        {"concurrency": 1},
    ],
)
def test_reuse_policy_accepts_valid_combinations(kwargs):
    r = flyte.ReusePolicy(**kwargs)
    assert isinstance(r.replicas, tuple) and len(r.replicas) == 2
    assert r.replicas[0] <= r.replicas[1]
    assert r.concurrency >= 1


def test_reuse_policy_valid_values_still_serialize():
    """The guards must not disturb a policy that was always well-formed."""
    from flyte._internal.runtime.reuse import reuse_policy_to_pb

    pb = reuse_policy_to_pb(flyte.ReusePolicy(replicas=(1, 10), concurrency=5))
    assert (pb.min_replicas, pb.max_replicas, pb.concurrency) == (1, 10, 5)
