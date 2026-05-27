from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import DictConfig
from rich_click import UsageError

import flyteplugins.hydra._cli as cli
from flyteplugins.hydra._cli import (
    _complete_hydra_override_values,
    _completed_result_value,
    _extract_config_overrides,
    _hydra_override_option_complete,
    _override_completion_context,
    hydra_run_cmd,
)


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


def test_override_completion_context_detects_dynamic_option_value() -> None:
    context = _override_completion_context(
        ["--dataset", "s3://bucket/data", "--cfg"],
        "optimizer.",
        {"--cfg", "--hydra-override"},
    )

    assert context == ([], "optimizer.", "")


def test_override_completion_context_collects_previous_overrides() -> None:
    context = _override_completion_context(
        ["--cfg", "training.epochs=5", "--hydra-override=hydra/launcher=flyte", "--cfg"],
        "+task_env=",
        {"--cfg", "--hydra-override"},
    )

    assert context == (["training.epochs=5", "hydra/launcher=flyte"], "+task_env=", "")


def test_complete_hydra_override_values_uses_hydra_config(hydra_conf) -> None:
    suggestions = _complete_hydra_override_values(
        config_path=str(hydra_conf),
        config_name="config",
        multirun=False,
        previous_overrides=[],
        incomplete="training.",
    )

    assert "training.epochs=" in suggestions


def test_hydra_run_shell_complete_suggests_dynamic_dictconfig_option(tmp_path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text("")
    monkeypatch.setattr(cli, "_load_script_task", lambda *_: _task_with_config_param("config"))

    with hydra_run_cmd.make_context("run", [str(script), "pipeline"], resilient_parsing=True) as ctx:
        suggestions = [item.value for item in hydra_run_cmd.shell_complete(ctx, "--co")]

    assert "--config" in suggestions


def test_hydra_run_shell_complete_routes_dictconfig_values(tmp_path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text("")
    monkeypatch.setattr(cli, "_load_script_task", lambda *_: _task_with_config_param("config"))
    monkeypatch.setattr(
        cli,
        "_complete_hydra_override_values",
        lambda **kwargs: [f"{kwargs['incomplete']}epochs="],
    )

    with hydra_run_cmd.make_context("run", [str(script), "pipeline", "--config"], resilient_parsing=True) as ctx:
        suggestions = [item.value for item in hydra_run_cmd.shell_complete(ctx, "training.")]

    assert "training.epochs=" in suggestions


def test_hydra_override_option_complete_routes_known_option_values(tmp_path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text("")
    monkeypatch.setattr(cli, "_load_script_task", lambda *_: _task_with_config_param("cfg"))
    monkeypatch.setattr(
        cli,
        "_complete_hydra_override_values",
        lambda **kwargs: [f"{kwargs['incomplete']}launcher="],
    )

    with hydra_run_cmd.make_context(
        "run",
        [str(script), "pipeline", "--hydra-override"],
        resilient_parsing=True,
    ) as ctx:
        suggestions = [item.value for item in _hydra_override_option_complete(ctx, None, "hydra/")]

    assert suggestions == ["hydra/launcher="]


def test_hydra_override_option_complete_finds_script_task_in_extra_args(tmp_path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text("")
    monkeypatch.setattr(cli, "_load_script_task", lambda *_: _task_with_config_param("cfg"))

    captured = {}

    def _fake_complete(**kwargs):
        captured.update(kwargs)
        return ["hydra.launcher.wait="]

    monkeypatch.setattr(cli, "_complete_hydra_override_values", _fake_complete)

    with hydra_run_cmd.make_context(
        "run",
        [
            "--multirun",
            str(script),
            "pipeline",
            "--cfg",
            "training.epochs=5",
            "--hydra-override",
            "hydra/launcher=flyte",
            "--hydra-override",
        ],
        resilient_parsing=True,
    ) as ctx:
        suggestions = [item.value for item in _hydra_override_option_complete(ctx, None, "hydra.launcher.w")]

    assert suggestions == ["hydra.launcher.wait="]
    assert captured["multirun"] is True
    assert captured["previous_overrides"] == ["hydra/launcher=flyte", "training.epochs=5"]
    assert captured["incomplete"] == "hydra.launcher.w"
