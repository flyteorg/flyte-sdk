import pytest

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="retry_tests")

_flaky_attempts = {"count": 0}
_always_fail_attempts = {"count": 0}


@env.task(retries=2)
def flaky_lookup(user_id: int) -> str:
    _flaky_attempts["count"] += 1
    if _flaky_attempts["count"] < 3:
        raise RuntimeError("Transient upstream error")
    return f"user-{user_id}"


@env.task(retries=1)
def always_fail() -> str:
    _always_fail_attempts["count"] += 1
    raise RuntimeError("permanent failure")


def test_local_run_retries_until_success():
    flyte.init_from_config(None)
    _flaky_attempts["count"] = 0

    run = flyte.with_runcontext(mode="local").run(flaky_lookup, user_id=7)

    assert run.outputs()[0] == "user-7"
    assert _flaky_attempts["count"] == 3


def test_local_run_retries_exhausted():
    flyte.init_from_config(None)
    _always_fail_attempts["count"] = 0

    with pytest.raises(flyte.errors.RuntimeUserError):
        flyte.with_runcontext(mode="local").run(always_fail)

    assert _always_fail_attempts["count"] == 2
