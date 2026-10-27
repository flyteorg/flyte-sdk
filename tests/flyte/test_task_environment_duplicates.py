import importlib
import sys
import textwrap
from unittest import mock

import pytest

import flyte

_TASK_MODULE = textwrap.dedent(
    """
    from shared_env import env

    @env.task
    def validate_inputs(x: int) -> str:
        return "{tag}"
    """
)


@pytest.fixture
def shared_env_modules(tmp_path, monkeypatch):
    (tmp_path / "shared_env.py").write_text("import flyte\nenv = flyte.TaskEnvironment(name='shared')\n")
    for tag in ("train_mod", "predict_mod"):
        (tmp_path / f"{tag}.py").write_text(_TASK_MODULE.format(tag=tag))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    for mod in ("shared_env", "train_mod", "predict_mod"):
        monkeypatch.delitem(sys.modules, mod, raising=False)


@pytest.fixture
def warned():
    with mock.patch("flyte._task_environment.logger") as logger:
        yield lambda: [c.args[0] for c in logger.warning.call_args_list]


def test_same_function_name_in_two_modules_warns(shared_env_modules, warned):
    importlib.import_module("train_mod")
    importlib.import_module("predict_mod")

    env = importlib.import_module("shared_env").env
    assert list(env.tasks) == ["shared.validate_inputs"]
    assert env.tasks["shared.validate_inputs"].func.__module__ == "predict_mod"

    msgs = warned()
    assert len(msgs) == 1
    assert "'shared.validate_inputs'" in msgs[0]
    assert "predict_mod.validate_inputs replaces train_mod.validate_inputs" in msgs[0]


def test_reregistering_same_function_does_not_warn(warned):
    env = flyte.TaskEnvironment(name="rereg")

    def fn(x: int) -> int:
        return x

    env.task(fn)
    env.task(fn)

    assert warned() == []


def test_distinct_function_names_do_not_warn(warned):
    env = flyte.TaskEnvironment(name="distinct")

    @env.task
    def a(x: int) -> int:
        return x

    @env.task
    def b(x: int) -> int:
        return x

    assert set(env.tasks) == {"distinct.a", "distinct.b"}
    assert warned() == []


def test_cloned_environments_keep_same_named_tasks_apart(warned):
    base = flyte.TaskEnvironment(name="base")
    train_env = base.clone_with(name="base-train")
    predict_env = base.clone_with(name="base-predict")

    @train_env.task
    def validate_inputs(x: int) -> int:
        return x

    @predict_env.task
    def validate_inputs(x: int) -> int:  # noqa: F811
        return x

    assert list(train_env.tasks) == ["base-train.validate_inputs"]
    assert list(predict_env.tasks) == ["base-predict.validate_inputs"]
    assert warned() == []
