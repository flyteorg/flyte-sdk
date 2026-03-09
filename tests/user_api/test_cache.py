import pytest

from flyte._cache import Cache
from flyte._cache.cache import cache_from_request


def test_cache_auto_behavior():
    cache = Cache(behavior="auto")
    assert cache.is_enabled()
    assert cache.behavior == "auto"


def test_cache_override_behavior():
    cache = Cache(behavior="override")
    assert cache.is_enabled()
    assert cache.behavior == "override"


def test_cache_disable_behavior():
    cache = Cache(behavior="disable")
    assert not cache.is_enabled()
    assert cache.behavior == "disable"


def test_cache_invalid_behavior():
    with pytest.raises(ValueError, match="Invalid cache behavior"):
        Cache(behavior="invalid")


def test_cache_version_override():
    cache = Cache(behavior="auto", version_override="v1.0")
    assert cache.get_version() == "v1.0"


def test_cache_disabled_returns_empty_version():
    cache = Cache(behavior="disable")
    assert cache.get_version() == ""


def test_cache_serialize_flag():
    cache = Cache(behavior="auto", serialize=True)
    assert cache.serialize is True


def test_cache_ignored_inputs_string():
    cache = Cache(behavior="auto", ignored_inputs="x")
    assert cache.get_ignored_inputs() == ("x",)


def test_cache_ignored_inputs_tuple():
    cache = Cache(behavior="auto", ignored_inputs=("x", "y"))
    assert cache.get_ignored_inputs() == ("x", "y")


def test_cache_ignored_inputs_default():
    cache = Cache(behavior="auto")
    assert cache.get_ignored_inputs() == ()


def test_cache_salt():
    cache = Cache(behavior="auto", salt="my_salt")
    assert cache.salt == "my_salt"


def test_cache_from_request_string():
    cache = cache_from_request("auto")
    assert isinstance(cache, Cache)
    assert cache.behavior == "auto"


def test_cache_from_request_cache_object():
    original = Cache(behavior="override", version_override="v2")
    result = cache_from_request(original)
    assert result is original


def test_cache_from_request_disable():
    cache = cache_from_request("disable")
    assert isinstance(cache, Cache)
    assert not cache.is_enabled()


def test_cache_version_requires_params_when_no_override():
    cache = Cache(behavior="auto")
    with pytest.raises(ValueError, match="Version parameters must be provided"):
        cache.get_version(None)


def test_cache_default_policies_set():
    cache = Cache(behavior="auto")
    assert cache.policies is not None
    assert len(cache.policies) > 0
