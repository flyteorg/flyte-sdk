from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import flyte
from omegaconf import DictConfig, OmegaConf

from flyteplugins.hydra import _run
from flyteplugins.hydra._run import (
    _config_param_name,
    _expand_basic_sweep_overrides,
    _task_override_kwargs,
    hydra_run,
    hydra_sweep,
)


class FakeTask:
    def __init__(self, name: str = "pipeline", config_param: str = "cfg") -> None:
        self.__name__ = name
        self.short_name = name
        self.native_interface = SimpleNamespace(
            inputs={
                config_param: (DictConfig, inspect._empty),
                "dataset": (str, inspect._empty),
            }
        )
        self.overrides = None

    def override(self, **kwargs):
        clone = FakeTask(self.__name__)
        clone.overrides = kwargs
        return clone


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def run(self, task, **kwargs):
        self.calls.append((task, kwargs))
        return OmegaConf.select(next(iter(kwargs.values())), "training.epochs")


def test_config_param_name_uses_task_dictconfig_input_name() -> None:
    assert _config_param_name(FakeTask(config_param="config")) == "config"


def test_task_override_kwargs_reads_entry_task_by_name(hydra_conf: Path) -> None:
    with _run._hydra_init(hydra_conf) as loader:
        cfg = loader.load_configuration("config", [], run_mode="RUN", from_shell=False)

    overrides = _task_override_kwargs(cfg, "task_env", "pipeline")

    assert isinstance(overrides["resources"], flyte.Resources)


def test_expand_basic_sweep_overrides_parses_hydra_overrides(hydra_conf: Path, tmp_path: Path) -> None:
    with _run._hydra_init(hydra_conf) as loader:
        jobs = _expand_basic_sweep_overrides(
            loader,
            [
                "training.epochs=1,2",
                f"hydra.sweep.dir={tmp_path / 'sweep'}",
            ],
        )

    assert jobs == [
        ["training.epochs=1", f"hydra.sweep.dir={tmp_path / 'sweep'}"],
        ["training.epochs=2", f"hydra.sweep.dir={tmp_path / 'sweep'}"],
    ]


def test_hydra_run_composes_config_and_uses_named_config_param(
    hydra_conf: Path, tmp_path: Path, monkeypatch
) -> None:
    runner = RecordingRunner()
    task = FakeTask(config_param="config")

    monkeypatch.setattr(_run.flyte, "with_runcontext", lambda **_: runner)

    result = hydra_run(
        task,
        config_path=hydra_conf,
        config_name="config",
        overrides=[f"hydra.run.dir={tmp_path / 'run'}"],
        dataset="s3://bucket/runtime",
        mode="local",
    )

    assert result == 3
    submitted_task, kwargs = runner.calls[0]
    assert submitted_task is not task
    assert isinstance(submitted_task.overrides["resources"], flyte.Resources)
    assert kwargs["dataset"] == "s3://bucket/runtime"
    assert list(kwargs["config"].keys()) == ["data", "training", "model", "task_env"]
    assert kwargs["config"].data.path == "s3://bucket/train"


def test_hydra_sweep_expands_grid_and_preserves_task_kwargs(
    hydra_conf: Path, tmp_path: Path, monkeypatch
) -> None:
    runner = RecordingRunner()
    task = FakeTask()

    monkeypatch.setattr(_run.flyte, "with_runcontext", lambda **_: runner)

    results = hydra_sweep(
        task,
        config_path=hydra_conf,
        config_name="config",
        overrides=[
            "training.epochs=1,2",
            f"hydra.sweep.dir={tmp_path / 'sweep'}",
        ],
        dataset="s3://bucket/runtime",
        mode="local",
    )

    assert results == [1, 2]
    assert [call[1]["dataset"] for call in runner.calls] == [
        "s3://bucket/runtime",
        "s3://bucket/runtime",
    ]
    assert [call[1]["cfg"].training.epochs for call in runner.calls] == [1, 2]
