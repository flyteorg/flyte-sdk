"""The bare `cache=` spelling on a task must behave like the `Cache(behavior=...)` field spelling.

`Cache.__post_init__` validates `behavior` against `CacheBehavior` and normalizes the alias
table (`"enable"` -> `"auto"`, `"off"` -> `"disable"`). `TaskTemplate.__post_init__` used to
re-implement a three-case subset of that with a `match` statement carrying no `case _`, so any
other value was silently left on the task as the raw string the user typed.
"""

import pytest

import flyte
from flyte._cache import Cache


@pytest.fixture
def env():
    return flyte.TaskEnvironment(name="cache_request_env")


def _task_with_cache(env, cache):
    @env.task(cache=cache)
    async def t() -> int:
        return 1

    return t


@pytest.mark.parametrize("behavior", ["auto", "override", "disable"])
def test_canonical_behavior_is_coerced_to_a_cache(env, behavior):
    cache = _task_with_cache(env, behavior).cache
    assert isinstance(cache, Cache)
    assert cache.behavior == behavior


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("enable", "auto"),
        ("enabled", "auto"),
        ("on", "auto"),
        ("true", "auto"),
        ("yes", "auto"),
        ("off", "disable"),
        ("false", "disable"),
        ("no", "disable"),
        ("none", "disable"),
    ],
)
def test_alias_is_applied_at_the_decorator(env, spelling, expected):
    """The alias table exists for exactly this spelling, and the decorator is where it is typed.

    Its own comment says it keeps `cache="enable"` working "instead of crashing with a
    ValueError that ends up in Sentry as an SDK error" -- but the task decorator never
    reached it, so `task.cache` stayed the string `'enable'`.
    """
    assert _task_with_cache(env, spelling).cache.behavior == expected


@pytest.mark.parametrize("bad", ["always", "enabl", "AUTO-ish", 5, True, 1.5, ["auto"]])
def test_invalid_cache_request_is_rejected_where_it_is_written(env, bad):
    """Same value, same message, whichever spelling the user reached for."""
    with pytest.raises(ValueError, match="Invalid cache behavior"):
        _task_with_cache(env, bad)


def test_a_cache_instance_is_passed_through_unchanged(env):
    cache = Cache(behavior="override", version_override="v1")
    assert _task_with_cache(env, cache).cache is cache


def test_the_default_is_still_disable(env):
    @env.task
    async def t() -> int:
        return 1

    assert t.cache.behavior == "disable"


def test_task_cache_is_always_a_cache_object(env):
    """What the `.cache.behavior` readers depend on.

    `app/extras/_webhook_app.py` and `ai/mcp/_tools.py` both do `<task>.cache.behavior`;
    against a raw string that is `AttributeError: 'str' object has no attribute 'behavior'`.
    """
    for spelling in ("auto", "enable", "off", Cache(behavior="disable")):
        assert isinstance(_task_with_cache(env, spelling).cache, Cache)


def test_override_keeps_the_coercion(env):
    """`override()` builds a new TaskTemplate, so it runs __post_init__ again."""
    t = _task_with_cache(env, "disable")

    assert t.override(cache="enable").cache.behavior == "auto"

    with pytest.raises(ValueError, match="Invalid cache behavior"):
        t.override(cache="always")
