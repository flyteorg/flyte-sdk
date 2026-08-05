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


def test_clone_with_overrides(base_env: flyte.TaskEnvironment):
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


# --- include files -----------------------------------------------------------


def test_task_environment_include_default_is_empty_tuple():
    env = flyte.TaskEnvironment(name="te_inc_default")
    assert env.include == ()
    assert isinstance(env.include, tuple)


def test_task_environment_include_accepts_tuple():
    inc = ("template.html", "assets/")
    env = flyte.TaskEnvironment(name="te_inc_tuple", include=inc)
    assert env.include == inc
    assert isinstance(env.include, tuple)


def test_task_environment_include_normalizes_list_to_tuple():
    env = flyte.TaskEnvironment(name="te_inc_list", include=["a.html", "b.html"])
    assert env.include == ("a.html", "b.html")
    assert isinstance(env.include, tuple)


def test_task_environment_include_rejects_bare_string():
    with pytest.raises(TypeError, match="sequence of str paths"):
        flyte.TaskEnvironment(name="te_inc_bare", include="solo.html")


def test_task_environment_include_rejects_non_str_entries():
    with pytest.raises(TypeError, match="include entries must be str"):
        flyte.TaskEnvironment(name="te_inc_bad", include=["ok.html", 1])


def test_task_environment_include_populates_declaring_file():
    env = flyte.TaskEnvironment(name="te_inc_declaring", include=("a.html",))
    assert env._declaring_file is not None
    assert env._declaring_file.endswith("test_task_environment.py")


def test_task_environment_works_with_include_and_tasks():
    env = flyte.TaskEnvironment(name="te_inc_with_tasks", include=["assets/config.yaml"])

    @env.task
    async def my_task(x: int) -> int:
        return x + 1

    assert env.include == ("assets/config.yaml",)
    assert "te_inc_with_tasks.my_task" in env.tasks


# --- clone_with + include ----------------------------------------------------


def test_task_environment_clone_with_inherits_empty_include():
    base = flyte.TaskEnvironment(name="te_clone_base_empty")
    clone = base.clone_with(name="te_clone_empty_child")
    assert clone.include == ()


def test_task_environment_clone_with_inherits_nonempty_include():
    base = flyte.TaskEnvironment(name="te_clone_base_inc", include=("a.html", "b.html"))
    clone = base.clone_with(name="te_clone_child_inc")
    # clone_with() without an explicit include must carry forward the original.
    assert clone.include == ("a.html", "b.html")


def test_task_environment_clone_with_overrides_include_tuple():
    base = flyte.TaskEnvironment(name="te_clone_override_base", include=("a.html",))
    clone = base.clone_with(name="te_clone_override_child", include=("c.html", "d.html"))
    assert clone.include == ("c.html", "d.html")
    # Base must not be mutated.
    assert base.include == ("a.html",)


def test_task_environment_clone_with_coerces_list_override_to_tuple():
    base = flyte.TaskEnvironment(name="te_clone_coerce_base", include=("a.html",))
    clone = base.clone_with(name="te_clone_coerce_child", include=["z.html"])
    assert clone.include == ("z.html",)
    assert isinstance(clone.include, tuple)


# --- hashability regression --------------------------------------------------


def test_task_environment_include_attr_is_hashable():
    env = flyte.TaskEnvironment(name="te_inc_hash", include=("a.html", "b.html"))
    assert hash(env.include) == hash(("a.html", "b.html"))


def test_task_environment_include_equal_values_hash_equal():
    env_a = flyte.TaskEnvironment(name="te_hash_a", include=["a.html", "b.html"])
    env_b = flyte.TaskEnvironment(name="te_hash_b", include=("a.html", "b.html"))
    assert env_a.include == env_b.include
    assert hash(env_a.include) == hash(env_b.include)


def test_task_environment_include_usable_in_set():
    env_a = flyte.TaskEnvironment(name="te_set_a", include=("x.html",))
    env_b = flyte.TaskEnvironment(name="te_set_b", include=("x.html",))
    env_c = flyte.TaskEnvironment(name="te_set_c", include=("y.html",))
    assert len({env_a.include, env_b.include, env_c.include}) == 2


def test_task_environment_clone_preserves_include_hashability():
    base = flyte.TaskEnvironment(name="te_clone_hash_base", include=("a.html",))
    clone = base.clone_with(name="te_clone_hash_child")
    # Still a tuple after replace() → still hashable.
    assert isinstance(clone.include, tuple)
    assert hash(clone.include) == hash(base.include)
