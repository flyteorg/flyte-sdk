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
        reusable=flyte.ReusePolicy(replicas=1),
        env_vars={"A": "B"},
        secrets="sec",
        depends_on=[other],
    )
    assert clone.image == "new_img"
    assert clone.cache == "custom"
    assert clone.reusable == flyte.ReusePolicy(replicas=1)
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
        ValueError, match=r"Environment name 'invalid-name!' must be in snake_case or kebab-case format\."
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


def test_task_evironment_typechecks():
    with pytest.raises(TypeError, match="Expected image to be of type str or Image, got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_image", image=123)

    with pytest.raises(TypeError, match="Expected secrets to be of type SecretRequest, got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_secrets", image="img", secrets=123)

    with pytest.raises(TypeError, match="Expected depends_on to be of type List\\[Environment\\], got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_depends", image="img", depends_on=[123])

    with pytest.raises(TypeError, match="Expected resources to be of type Resources, got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_resources", image="img", resources=123)

    with pytest.raises(TypeError, match="Expected env_vars to be of type Dict\\[str, str\\], got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_env_vars", image="img", env_vars=123)

    with pytest.raises(TypeError, match="Expected Environment, got <class 'int'>"):
        env = flyte.TaskEnvironment(name="test_env", image="img")
        env.add_dependency(123)

    # check cache request and reusable request type checks
    with pytest.raises(TypeError, match="Expected cache to be of type str or Cache, got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_cache", image="img", cache=123)

    with pytest.raises(TypeError, match="Expected reusable to be of type ReusePolicy, got <class 'int'>"):
        flyte.TaskEnvironment(name="bad_reusable", image="img", reusable=123)
