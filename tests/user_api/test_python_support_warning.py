"""Tests for the warning shown when the running Python version is about to lose support."""

import sys
import warnings
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

import flyte
from flyte import _initialize as init_module
from flyte._initialize import _warn_python_support_ending, init_in_cluster
from flyte.cli.main import main

IN_POD_ENV_VAR = "FLYTE_INTERNAL_EXECUTION_PROJECT"


@pytest.fixture
def running_on(monkeypatch):
    """Pretend to run on the client side with a given Python version, the warning not emitted yet."""
    monkeypatch.setattr(init_module, "_python_support_warned", False)
    monkeypatch.delenv(IN_POD_ENV_VAR, raising=False)

    def _running_on(*version: int) -> None:
        monkeypatch.setattr(init_module, "_python_version", lambda: version)

    return _running_on


def _support_warnings(func) -> list[str]:
    """Call `func` and return the end-of-support warnings it emitted, whatever the active filters."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        func()
    return [str(w.message) for w in record if w.category is FutureWarning and "Support for Python" in str(w.message)]


def test_warns_when_python_is_about_to_lose_support(running_on):
    running_on(3, 11)

    assert _support_warnings(_warn_python_support_ending) == [
        "Support for Python 3.11 will be removed in an upcoming Flyte release. Upgrade to Python 3.12 or newer."
    ]


@pytest.mark.parametrize("version", [(3, 12), (3, 13), (3, 14)])
def test_silent_on_python_that_stays_supported(running_on, version):
    running_on(*version)

    assert _support_warnings(_warn_python_support_ending) == []


def test_warns_once_per_process(running_on):
    running_on(3, 11)

    def _call_repeatedly():
        for _ in range(3):
            _warn_python_support_ending()

    assert len(_support_warnings(_call_repeatedly)) == 1


def test_silent_in_task_or_app_container(running_on, monkeypatch):
    running_on(3, 11)
    monkeypatch.setenv(IN_POD_ENV_VAR, "my-project")

    assert _support_warnings(_warn_python_support_ending) == []


def test_init_warns_once(running_on, monkeypatch):
    running_on(3, 11)
    # init replaces the global config and extends sys.path: put both back afterwards
    monkeypatch.setattr(init_module, "_init_config", init_module._init_config)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def _init_twice():
        flyte.init()
        flyte.init()

    assert len(_support_warnings(_init_twice)) == 1


def test_init_is_silent_on_python_that_stays_supported(running_on, monkeypatch):
    running_on(3, 12)
    monkeypatch.setattr(init_module, "_init_config", init_module._init_config)
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert _support_warnings(flyte.init) == []


def test_silent_after_init_in_cluster(running_on):
    """The in-cluster entrypoints pass the project as an argument, so the environment may not give it away."""
    running_on(3, 11)

    def _init_in_cluster_then_check():
        with patch("flyte._initialize.init", new_callable=AsyncMock):
            init_in_cluster(project="my-project", domain="development", api_key="key")
        # What `flyte.init` runs, also when the task's own module calls it inside the container
        _warn_python_support_ending()

    assert _support_warnings(_init_in_cluster_then_check) == []


def test_cli_warns(running_on):
    running_on(3, 11)

    result = None

    def _invoke():
        nonlocal result
        result = CliRunner().invoke(main, ["get", "--help"])

    assert len(_support_warnings(_invoke)) == 1
    assert result is not None and result.exit_code == 0, result.output


def test_cli_is_silent_on_python_that_stays_supported(running_on):
    running_on(3, 12)

    assert _support_warnings(lambda: CliRunner().invoke(main, ["get", "--help"])) == []
