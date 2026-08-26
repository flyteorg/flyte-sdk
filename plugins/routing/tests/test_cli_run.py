"""Routing `flyte run` through the CLI hook, with no support from the SDK.

These drive the real `flyte` CLI end to end, standing in only for the network call, because the
whole point of the hook is that it reaches through Click's group chain to the leaf command -- and
that chain is exactly what a unit test would stub away.
"""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

_CONFIG = """
task:
  project: shared
  domain: development
profiles:
  us-east:
    task: {project: east-proj}
  eu-west:
    task: {project: west-proj}
  gpu-pool:
    task: {project: gpu-proj}
"""

_TASKS = """
import flyte

env = flyte.TaskEnvironment("routed", resources=flyte.Resources(cpu=2))


@env.task
async def train(dataset: str, epochs: int = 3) -> int:
    return epochs
"""


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch):
    (tmp_path / "config.yaml").write_text(_CONFIG)
    (tmp_path / "tasks.py").write_text(_TASKS)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def submissions(monkeypatch):
    """Record what each run would have submitted, and under which profile."""
    from flyte._initialize import _get_init_config
    from flyte._run import _Runner

    recorded = []

    async def fake(self, obj, *args, **kwargs):
        cfg = _get_init_config()
        recorded.append({"name": self._name, "project": cfg.project, "labels": dict(self._labels or {})})
        return "submitted"

    monkeypatch.setattr(_Runner, "_run_remote", fake)
    return recorded


def _run(project, *args, run_opts=()):
    """`--name` / `--label` are options on the `run` group, so they precede the file."""
    from flyte.cli.main import main

    cfg = str(project / "config.yaml")
    return CliRunner().invoke(main, ["--config", cfg, "run", *run_opts, "tasks.py", "train", *args])


def test_run_is_routed(project, submissions) -> None:
    res = _run(project, "--dataset", "s3://bucket/x")
    assert res.exit_code == 0, res.output
    (sub,) = submissions
    assert sub["project"] in {"east-proj", "west-proj", "gpu-proj"}
    assert sub["project"] != "shared", "the run should not have gone to the default"


def test_placement_follows_the_arguments(project, submissions) -> None:
    """Data-location routing: the hook sees parsed arguments, so placement can depend on them."""
    for i in range(30):
        _run(project, "--dataset", f"s3://bucket/{i}")
    assert len({s["project"] for s in submissions}) > 1


def test_the_same_arguments_always_route_the_same_way(project, submissions) -> None:
    for _ in range(3):
        _run(project, "--dataset", "s3://bucket/x")
    assert len({s["project"] for s in submissions}) == 1


def test_each_run_gets_its_own_name(project, submissions) -> None:
    """Placement is deterministic; names must not be, or the second run collides."""
    for _ in range(3):
        _run(project, "--dataset", "s3://bucket/x")
    assert len({s["name"] for s in submissions}) == 3


def test_the_run_name_decodes_back_to_its_profile(project, submissions) -> None:
    from flyteplugins.routing import resolve_run_profile

    _run(project, "--dataset", "s3://bucket/x")
    (sub,) = submissions
    profile = resolve_run_profile(sub["name"], project / "config.yaml")
    assert profile is not None
    expected = {"us-east": "east-proj", "eu-west": "west-proj", "gpu-pool": "gpu-proj"}
    assert expected[profile] == sub["project"]


def test_the_decision_is_labelled(project, submissions) -> None:
    _run(project, "--dataset", "s3://bucket/x")
    (sub,) = submissions
    assert sub["labels"]["routed-by"] == "consistent-hash"
    assert sub["labels"]["routed-to"] in {"us-east", "eu-west", "gpu-pool"}


def test_explicit_profile_wins(project, submissions) -> None:
    """`--profile` pins the run; the policy declines rather than overruling it."""
    from flyte.cli.main import main

    cfg = str(project / "config.yaml")
    res = CliRunner().invoke(
        main,
        ["--config", cfg, "--profile", "eu-west", "run", "tasks.py", "train", "--dataset", "s3://bucket/x"],
    )
    assert res.exit_code == 0, res.output
    (sub,) = submissions
    assert sub["project"] == "west-proj"
    assert sub["name"] is None, "a pinned run keeps the control plane's naming"


def test_explicit_name_is_kept(project, submissions) -> None:
    _run(project, "--dataset", "s3://bucket/x", run_opts=("--name", "mine"))
    (sub,) = submissions
    assert sub["name"] == "mine"


def test_caller_labels_win_on_conflict(project, submissions) -> None:
    _run(project, "--dataset", "s3://bucket/x", run_opts=("--label", "routed-by=me"))
    (sub,) = submissions
    assert sub["labels"]["routed-by"] == "me"


def test_a_second_run_does_not_inherit_the_first_name(project, submissions) -> None:
    """Click reuses command objects in-process, so a name minted for one run must not leak."""
    _run(project, "--dataset", "s3://bucket/x", run_opts=("--name", "mine"))
    _run(project, "--dataset", "s3://bucket/x")
    assert submissions[0]["name"] == "mine"
    assert submissions[1]["name"] != "mine"


def test_a_config_without_profiles_is_left_alone(tmp_path: pathlib.Path, submissions, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text("task:\n  project: shared\n  domain: development\n")
    (tmp_path / "tasks.py").write_text(_TASKS)
    monkeypatch.chdir(tmp_path)
    res = _run(tmp_path, "--dataset", "s3://bucket/x")
    assert res.exit_code == 0, res.output
    (sub,) = submissions
    assert sub["project"] == "shared"
    assert sub["name"] is None


def test_a_failing_policy_falls_back_to_the_default(project, submissions, monkeypatch) -> None:
    """A plugin fault must not take the run down -- it should submit as if uninstalled."""
    import flyteplugins.routing.cli_run as cli_run

    monkeypatch.setattr(cli_run, "_decide", lambda cmd, ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    res = _run(project, "--dataset", "s3://bucket/x")
    assert res.exit_code == 0, res.output
    (sub,) = submissions
    assert sub["project"] == "shared"
