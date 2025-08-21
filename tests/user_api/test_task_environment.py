import pytest

import flyte

env_with_tasks = flyte.TaskEnvironment(
    name="env_with_tasks",
)


@env_with_tasks.task
async def sample_task(x: int, y: int) -> int:
    """
    A sample task that adds two numbers.
    """
    return x + y


@env_with_tasks.task(short_name="test")
async def sample_task2(x: int, y: int) -> int:
    return x + y


@pytest.fixture
def base_env():
    return flyte.TaskEnvironment(name="env", image="img", resources=flyte.Resources(cpu="1", memory="1Gi"))


def test_clone_with_defaults(base_env):
    clone = base_env.clone_with(name="env2")
    assert clone.name == "env2"
    assert clone.image == base_env.image
    assert clone.resources == base_env.resources
    assert clone.cache == base_env.cache
    assert clone.reusable is None
    assert clone.depends_on == []


def test_clone_with_overrides(base_env):
    other = flyte.TaskEnvironment(name="other", image="x", resources=base_env.resources)
    clone = base_env.clone_with(
        name="new",
        image="new_img",
        resources=flyte.Resources(cpu="2", memory="2Gi"),
        cache="custom",
        reusable="yes",
        env_vars={"A": "B"},
        secrets="sec",
        depends_on=[other],
    )
    assert clone.image == "new_img"
    assert clone.cache == "custom"
    assert clone.reusable == "yes"
    assert clone.env_vars == {"A": "B"}
    assert clone.secrets == "sec"
    assert clone.depends_on == [other]


@pytest.mark.asyncio
async def test_async_task_decorator_and_wrapper(base_env):
    @base_env.task
    async def foo(x, y):
        return x + y

    # template created and stored
    assert foo.name == "env.foo"
    assert foo in base_env.tasks.values()
    # wrapper calls original
    result = await foo.func(2, 3)
    assert result == 5


def test_reusable_conflict_pod_template(base_env):
    env = base_env.clone_with(name="r", reusable=flyte.ReusePolicy(replicas=(1, 2)))

    async def z():
        return None

    with pytest.raises(ValueError):
        env.task(z, pod_template="tmpl")


def test_add_task_and_duplicates(base_env):
    class Dummy:
        def __init__(self, name):
            self.name = name

    t1 = Dummy("t1")
    base_env.add_task(t1)
    assert "t1" in base_env.tasks
    with pytest.raises(ValueError):
        base_env.add_task(t1)


def test_clone_no_tasks(base_env):
    # Ensure cloning does not carry over tasks
    clone = base_env.clone_with(name="clone_no_tasks")
    assert clone.tasks == {}
    assert clone.name == "clone_no_tasks"

    @clone.task
    async def new_task(x: int) -> int:
        return x * 2

    assert new_task.name == "clone_no_tasks.new_task"
    assert new_task in clone.tasks.values()


def test_task_environment_name_validation():
    with pytest.raises(
        ValueError, match="Environment name 'invalid-name!' must be in snake_case or kebab-case format."
    ):
        flyte.TaskEnvironment(name="invalid-name!")

    # Valid names should not raise
    flyte.TaskEnvironment(name="valid_name")
    flyte.TaskEnvironment(name="valid-name")
    flyte.TaskEnvironment(name="valid123_name")  # numbers allowed


def test_env_with_tasks():
    assert len(env_with_tasks.tasks) == 2
    assert list(env_with_tasks.tasks.keys()) == ["env_with_tasks.sample_task", "env_with_tasks.sample_task2"]
    assert sample_task.short_name == "sample_task"
    assert sample_task.name == "env_with_tasks.sample_task"
    assert sample_task2.short_name == "test"
    assert sample_task2.name == "env_with_tasks.sample_task2"
