from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import DictConfig
from rich_click import UsageError

from flyteplugins.hydra._cli import _completed_result_value, _extract_config_overrides


def _task_with_config_param(name: str):
    return SimpleNamespace(native_interface=SimpleNamespace(inputs={name: (DictConfig, None)}))


def test_extract_config_overrides_uses_dictconfig_param_name() -> None:
    task = _task_with_config_param("config")

    overrides, remaining = _extract_config_overrides(
        task,
        [
            "--dataset",
            "s3://bucket/data",
            "--config",
            "optimizer.lr=0.01",
            "--config=training.epochs=20",
        ],
    )

    assert overrides == ["optimizer.lr=0.01", "training.epochs=20"]
    assert remaining == ["--dataset", "s3://bucket/data"]


def test_extract_config_overrides_rejects_missing_value() -> None:
    task = _task_with_config_param("cfg")

    with pytest.raises(UsageError, match="requires an override value"):
        _extract_config_overrides(task, ["--cfg"])


def test_completed_result_value_suppresses_unresolved_remote_run_url() -> None:
    run = SimpleNamespace(url="https://flyte.example/runs/abc")

    assert _completed_result_value(run) is None


def test_completed_result_value_returns_waited_result_value() -> None:
    run = SimpleNamespace(url="https://flyte.example/runs/abc", value=0.25)

    assert _completed_result_value(run) == 0.25
