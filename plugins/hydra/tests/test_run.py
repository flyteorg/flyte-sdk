from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import flyte
from omegaconf import DictConfig, OmegaConf

from flyteplugins.hydra import _run
from flyteplugins.hydra._run import (
    _coerce_override_kwargs,
    _config_param_name,
    _expand_basic_sweep_overrides,
    _merge_task_env_image,
    _task_override_kwargs,
    apply_task_env,
    hydra_run,
    hydra_sweep,
)


class FakeTask:
    def __init__(
        self,
        name: str = "pipeline",
        config_param: str = "cfg",
        pod_template: flyte.PodTemplate | str | None = None,
    ) -> None:
        self.__name__ = name
        self.short_name = name
        self.pod_template = pod_template
        self.native_interface = SimpleNamespace(
            inputs={
                config_param: (DictConfig, inspect._empty),
                "dataset": (str, inspect._empty),
            }
        )
        self.overrides = None

    def override(self, **kwargs):
        clone = FakeTask(self.__name__, pod_template=kwargs.get("pod_template", self.pod_template))
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


def test_task_override_kwargs_preserves_image_until_task_is_available() -> None:
    overrides = _coerce_override_kwargs(
        OmegaConf.create(
            {
                "image": "ghcr.io/acme/train:latest",
                "primary_container_name": "main",
                "resources": {"cpu": "2", "memory": "8Gi"},
            }
        )
    )

    assert overrides["image"] == "ghcr.io/acme/train:latest"
    assert overrides["primary_container_name"] == "main"
    assert isinstance(overrides["resources"], flyte.Resources)


def test_merge_task_env_image_creates_pod_template_without_existing_template() -> None:
    overrides = _merge_task_env_image(
        FakeTask(),
        _coerce_override_kwargs(
            OmegaConf.create(
                {
                    "image": "ghcr.io/acme/train:latest",
                    "primary_container_name": "main",
                    "resources": {"cpu": "2", "memory": "8Gi"},
                }
            )
        ),
    )

    pod_template = overrides["pod_template"]

    assert "image" not in overrides
    assert "primary_container_name" not in overrides
    assert isinstance(pod_template, flyte.PodTemplate)
    assert pod_template.primary_container_name == "main"
    assert pod_template.pod_spec.containers[0].name == "main"
    assert pod_template.pod_spec.containers[0].image == "ghcr.io/acme/train:latest"
    assert isinstance(overrides["resources"], flyte.Resources)


def test_merge_task_env_image_patches_copy_of_existing_pod_template() -> None:
    from kubernetes.client import V1Container, V1PodSpec

    existing = flyte.PodTemplate(
        pod_spec=V1PodSpec(
            containers=[
                V1Container(name="worker"),
                V1Container(name="metrics", image="ghcr.io/acme/metrics:latest"),
            ]
        ),
        primary_container_name="worker",
        labels={"team": "ml"},
        annotations={"owner": "training"},
    )
    task = FakeTask(pod_template=existing)
    overrides = _merge_task_env_image(task, {"image": "ghcr.io/acme/train:latest"})

    pod_template = overrides["pod_template"]
    containers = {container.name: container for container in pod_template.pod_spec.containers}

    assert pod_template is not existing
    assert pod_template.primary_container_name == "worker"
    assert pod_template.labels == {"team": "ml"}
    assert pod_template.annotations == {"owner": "training"}
    assert containers["worker"].image == "ghcr.io/acme/train:latest"
    assert containers["metrics"].image == "ghcr.io/acme/metrics:latest"
    assert existing.pod_spec.containers[0].image is None


def test_merge_task_env_image_rejects_named_pod_template() -> None:
    task = FakeTask(pod_template="cluster-pod-template")

    try:
        _merge_task_env_image(task, {"image": "ghcr.io/acme/train:latest"})
    except ValueError as e:
        assert "named pod_template string" in str(e)
    else:
        raise AssertionError("Expected named pod_template strings to be rejected")


def test_apply_task_env_overrides_child_task_image_and_resources() -> None:
    cfg = OmegaConf.create(
        {
            "task_env": {
                "train_model": {
                    "image": "ghcr.io/acme/train:py3.13",
                    "resources": {"cpu": "4", "memory": "6Gi"},
                }
            }
        }
    )
    task = FakeTask(name="train_model")

    overridden = apply_task_env(task, cfg)

    pod_template = overridden.overrides["pod_template"]

    assert overridden is not task
    assert pod_template.pod_spec.containers[0].image == "ghcr.io/acme/train:py3.13"
    assert isinstance(overridden.overrides["resources"], flyte.Resources)


def test_example_prebuilt_image_config_covers_entry_and_child_tasks() -> None:
    examples_conf = Path(__file__).parents[1] / "examples" / "conf"
    with _run._hydra_init(examples_conf) as loader:
        cfg = loader.load_configuration("training", ["+task_env=prebuilt_image"], run_mode="RUN", from_shell=False)

    assert cfg.task_env.pipeline.image == "ghcr.io/flyteorg/flyte:py3.13-v2.0.12"
    assert cfg.task_env.pipeline_with_podtemplate.image == "ghcr.io/flyteorg/flyte:py3.13-v2.0.12"
    assert cfg.task_env.train_model.image == "ghcr.io/flyteorg/flyte:py3.13-v2.0.12"

    train_task = apply_task_env(FakeTask(name="train_model"), cfg)

    assert train_task.overrides["pod_template"].pod_spec.containers[0].image == (
        "ghcr.io/flyteorg/flyte:py3.13-v2.0.12"
    )


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


def test_hydra_run_composes_config_and_uses_named_config_param(hydra_conf: Path, tmp_path: Path, monkeypatch) -> None:
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


def test_hydra_run_merges_task_env_image_with_existing_pod_template(tmp_path: Path, monkeypatch) -> None:
    from kubernetes.client import V1Container, V1PodSpec

    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "config.yaml").write_text(
        """data:
  path: s3://bucket/train
training:
  epochs: 3
model:
  name: resnet50
task_env:
  pipeline_with_podtemplate:
    image: ghcr.io/acme/train:py3.13
    primary_container_name: primary
    resources:
      cpu: '4'
      memory: 6Gi
""",
    )
    existing = flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(containers=[V1Container(name="primary", image="python:3.9")]),
    )
    runner = RecordingRunner()
    task = FakeTask(name="pipeline_with_podtemplate", pod_template=existing)

    monkeypatch.setattr(_run.flyte, "with_runcontext", lambda **_: runner)

    hydra_run(
        task,
        config_path=conf,
        config_name="config",
        mode="local",
        dataset="s3://bucket/runtime",
    )

    submitted_task, _ = runner.calls[0]
    pod_template = submitted_task.overrides["pod_template"]

    assert pod_template.pod_spec.containers[0].image == "ghcr.io/acme/train:py3.13"
    assert existing.pod_spec.containers[0].image == "python:3.9"
    assert isinstance(submitted_task.overrides["resources"], flyte.Resources)


def test_hydra_sweep_expands_grid_and_preserves_task_kwargs(hydra_conf: Path, tmp_path: Path, monkeypatch) -> None:
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
