from typing import List

import pytest

import flyte
import flyte.errors


def test_map_partials_happy():
    from functools import partial

    # Define an environment with specific resources
    env = flyte.TaskEnvironment(name="map-test", resources=flyte.Resources(cpu="1"))

    @env.task
    def my_task(batch_id: int, name: str, constant_param: str) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def main() -> List[str]:
        compounds = list(range(100))
        constant_param = "shared_config"

        curry_consts = partial(my_task, constant_param=constant_param, name="daniel")

        return list(flyte.map(curry_consts, compounds))

    flyte.init()
    run = flyte.with_runcontext(mode="local").run(main)
    print(run.outputs())


def test_partial_validation():
    from functools import partial

    from flyte._map import _Mapper

    env = flyte.TaskEnvironment(name="map-test")

    @env.task
    def my_task(name: str, constant_param: str, batch_id: int) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def three_param_task(a: int, b: str, c: float) -> str:
        return f"{a}-{b}-{c}"

    mapper = _Mapper()

    # Case 1: No parameters left for mapping (all parameters provided)
    full_partial = partial(my_task, "alice", "config", 42)
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(full_partial)

    # Case 2: Too many parameters left for mapping (no parameters provided)
    empty_partial = partial(my_task)
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(empty_partial)

    # Case 3: Two parameters left for mapping
    two_left_partial = partial(my_task, "alice")
    with pytest.raises(TypeError, match="must leave exactly one parameter unspecified"):
        mapper.validate_partial(two_left_partial)

    # Case 4: Parameter provided both as positional and keyword argument
    duplicate_param_partial = partial(three_param_task, 1, a=2)
    with pytest.raises(TypeError, match="provided both as positional argument and keyword argument"):
        mapper.validate_partial(duplicate_param_partial)


def test_map_partials_unhappy():
    from functools import partial

    # Define an environment with specific resources
    env = flyte.TaskEnvironment(name="map-test", resources=flyte.Resources(cpu="1"))

    @env.task
    def my_task(name: str, constant_param: str, batch_id: int) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def main() -> List[str]:
        compounds = list(range(100))
        constant_param = "shared_config"

        curry_consts = partial(my_task, constant_param=constant_param, name="daniel")

        return list(flyte.map(curry_consts, compounds))

    flyte.init()
    with pytest.raises(flyte.errors.RuntimeUserError) as excinfo:
        flyte.with_runcontext(mode="local").run(main)
    assert excinfo.value.code == "TypeError", excinfo.value
