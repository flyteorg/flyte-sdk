from flyte._cache import Cache, FunctionBodyPolicy
from flyte._cache.cache import VersionParameters, cache_from_request


def test_function_body():
    def test_func(a: str) -> str:
        return "Hello World"

    # Function to get the second version of the exact same code
    def get_fn2():
        def test_func(a: str) -> str:
            return "Hello World"

        return test_func

    vp1 = VersionParameters(
        func=test_func,
    )
    policy = FunctionBodyPolicy()
    v1 = policy.get_version("", vp1)
    vp2 = VersionParameters(
        func=get_fn2(),
    )
    v2 = policy.get_version("", vp2)

    assert v1 == v2


def test_coercion():
    c = cache_from_request("auto")
    assert isinstance(c, Cache)


def test_defaults():
    c = Cache(behavior="auto")
    assert len(c.policies) >= 1
