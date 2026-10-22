import importlib
import pathlib
import sys
import textwrap
from unittest import mock

import pytest

import flyte
from flyte._internal.resolvers.default import DefaultTaskResolver
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import SerializationContext

_TASK_MODULE = textwrap.dedent(
    """
    from ns_shared_env import env

    @env.task(namespace="{tag}")
    async def validate_inputs(x: int) -> str:
        return "{tag}"
    """
)


@pytest.fixture
def namespaced_modules(tmp_path, monkeypatch):
    (tmp_path / "ns_shared_env.py").write_text(
        "import flyte\n"
        "env = flyte.TaskEnvironment(name='shared', reusable=flyte.ReusePolicy(replicas=1, concurrency=2))\n"
    )
    for tag in ("train", "predict"):
        (tmp_path / f"ns_{tag}_mod.py").write_text(_TASK_MODULE.format(tag=tag))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for mod in ("ns_shared_env", "ns_train_mod", "ns_predict_mod"):
        monkeypatch.delitem(sys.modules, mod, raising=False)


def test_namespace_is_inserted_between_env_and_function_name():
    env = flyte.TaskEnvironment(name="ns_env")

    @env.task(namespace="train")
    async def validate_inputs(x: int) -> int:
        return x

    assert validate_inputs.name == "ns_env.train.validate_inputs"
    assert validate_inputs.short_name == "validate_inputs"
    assert list(env.tasks) == ["ns_env.train.validate_inputs"]


def test_namespace_can_be_dotted():
    env = flyte.TaskEnvironment(name="dotted")

    @env.task(namespace="myproject.train")
    async def fn() -> None: ...

    assert fn.name == "dotted.myproject.train.fn"


def test_no_namespace_keeps_default_name():
    env = flyte.TaskEnvironment(name="plain")

    @env.task
    async def fn() -> None: ...

    assert fn.name == "plain.fn"


def test_same_function_name_in_two_modules_with_namespaces_does_not_collide(namespaced_modules):
    with mock.patch("flyte._task_environment.logger") as logger:
        train = importlib.import_module("ns_train_mod")
        predict = importlib.import_module("ns_predict_mod")

    env = importlib.import_module("ns_shared_env").env
    assert sorted(env.tasks) == ["shared.predict.validate_inputs", "shared.train.validate_inputs"]
    assert not logger.warning.called

    # Both tasks stay in the same (reusable) environment, and each still resolves to its own function.
    resolver = DefaultTaskResolver()
    for mod in (train, predict):
        task = mod.validate_inputs
        assert task.parent_env_name == "shared"
        assert task.reusable == env.reusable
        assert resolver.load_task(resolver.loader_args(task, namespaced_modules)) is task


def test_serialized_task_id_uses_namespaced_name():
    env = flyte.TaskEnvironment(name="serde_env", image="python:3.12")

    @env.task(namespace="train")
    async def fn(x: int) -> int:
        return x

    ctx = SerializationContext(version="v1", project="p", domain="d", org="o", root_dir=pathlib.Path(__file__).parent)
    spec = translate_task_to_wire(fn, ctx)
    assert spec.task_template.id.name == "serde_env.train.fn"
    assert spec.short_name == "fn"
    assert spec.environment.name == "serde_env"


@pytest.mark.parametrize("namespace", ["__main__", "pkg.__main__", "__main__-x"])
def test_main_module_namespace_is_rejected(namespace):
    env = flyte.TaskEnvironment(name="main_env")
    with pytest.raises(ValueError, match="__main__"):
        env.task(namespace=namespace)


@pytest.mark.parametrize("namespace", ["", "has space", "trailing.", ".leading", "a..b", "a/b"])
def test_invalid_namespace_is_rejected(namespace):
    env = flyte.TaskEnvironment(name="bad_env")
    with pytest.raises(ValueError, match="namespace"):
        env.task(namespace=namespace)
