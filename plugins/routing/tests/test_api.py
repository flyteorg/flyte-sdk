"""`flyteplugins.routing.run` — the Python-API path the CLI hook cannot reach."""

from __future__ import annotations

import pathlib

import flyte
import pytest
from flyte._initialize import _get_init_config

from flyteplugins.routing import decide, resolve_run_profile, run, with_runcontext

# An endpoint is present so runs resolve to remote mode -- routing only applies there. No
# connection is made: the submit call itself is replaced below.
_CONFIG = """
admin:
  endpoint: dns:///default.example.com
  insecure: true
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

env = flyte.TaskEnvironment("api_routed")


@env.task
async def train(dataset: str, epochs: int = 3) -> int:
    return epochs


@pytest.fixture
def cfg_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "config.yaml"
    p.write_text(_CONFIG)
    return p


@pytest.fixture
def initialized(cfg_file: pathlib.Path):
    import flyte._initialize as init_mod

    previous = init_mod._init_config
    flyte.init_from_config(cfg_file)
    try:
        yield
    finally:
        init_mod._init_config = previous


@pytest.fixture
def submissions(monkeypatch):
    """Record what each submission would have sent, in place of the network call."""
    from flyte._run import _Runner

    recorded = []

    async def fake(self, obj, *args, **kwargs):
        cfg = _get_init_config()
        recorded.append({"name": self._name, "project": cfg.project, "labels": dict(self._labels or {})})
        return "submitted"

    monkeypatch.setattr(_Runner, "_run_remote", fake)
    return recorded


class TestDecide:
    """`decide` applies the policy without submitting, so you can see where a run would go."""

    def test_returns_a_profile(self, cfg_file: pathlib.Path) -> None:
        d = decide(train, config_file=cfg_file, dataset="s3://bucket/x")
        assert d is not None and d.profile in {"us-east", "eu-west", "gpu-pool"}

    def test_is_deterministic(self, cfg_file: pathlib.Path) -> None:
        a = decide(train, config_file=cfg_file, dataset="s3://bucket/x")
        b = decide(train, config_file=cfg_file, dataset="s3://bucket/x")
        assert a.profile == b.profile

    def test_declines_when_a_profile_is_pinned(self, cfg_file: pathlib.Path) -> None:
        assert decide(train, config_file=cfg_file, profile="eu-west", dataset="s3://b/x") is None

    def test_matches_what_the_cli_would_choose(self, cfg_file: pathlib.Path) -> None:
        """The two paths must agree, or a run routes differently depending on how it was launched.

        Both go through the same `route()` on the same context fields; this pins that they stay
        that way.
        """
        from flyteplugins.routing import RoutingContext, route

        direct = route(
            RoutingContext(
                profiles=("us-east", "eu-west", "gpu-pool"),
                project="shared",
                domain="development",
                task_name=train.name,
                inputs={"dataset": "s3://bucket/x"},
            )
        )
        via_api = decide(train, config_file=cfg_file, dataset="s3://bucket/x")
        assert direct.profile == via_api.profile


class TestRun:
    def test_submits_under_the_routed_profile(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file).run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["project"] != "shared"

    def test_names_the_run_so_it_can_be_found(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file).run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert resolve_run_profile(sub["name"], cfg_file) is not None

    def test_labels_the_decision(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file).run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["labels"]["routed-by"] == "consistent-hash"

    def test_restores_the_ambient_profile(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file).run(train, dataset="s3://bucket/x")
        assert _get_init_config().project == "shared"

    def test_repeat_runs_share_a_profile_but_not_a_name(self, initialized, submissions, cfg_file) -> None:
        for _ in range(3):
            with_runcontext(config_file=cfg_file).run(train, dataset="s3://bucket/x")
        assert len({s["project"] for s in submissions}) == 1
        assert len({s["name"] for s in submissions}) == 3

    def test_plain_flyte_run_is_not_routed(self, initialized, submissions, cfg_file) -> None:
        """The documented gap. `flyte.run` submits to the default profile -- this plugin cannot
        see it, and the test exists so the limitation is asserted rather than assumed."""
        flyte.run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["project"] == "shared"
        assert sub["name"] is None


class TestDropIn:
    """The swap from `flyte.run` must be mechanical, or the script path is not really available."""

    def test_positional_arguments_work(self, initialized, submissions, cfg_file) -> None:
        run(train, "s3://bucket/x")
        (sub,) = submissions
        assert sub["project"] != "shared"

    def test_positional_and_keyword_describe_identically(self, cfg_file: pathlib.Path) -> None:
        """`run(t, "s3://x")` and `run(t, dataset="s3://x")` must place the same run in the same
        cluster -- otherwise placement depends on how the call was written."""
        a = decide(train, "s3://bucket/x", config_file=cfg_file)
        b = decide(train, config_file=cfg_file, dataset="s3://bucket/x")
        assert a.profile == b.profile

    def test_mixed_positional_and_keyword(self, initialized, submissions, cfg_file) -> None:
        run(train, "s3://bucket/x", epochs=7)
        (sub,) = submissions
        assert sub["project"] != "shared"

    def test_with_runcontext_options_pass_through(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file, name="mine").run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["name"] == "mine", "a caller-supplied name must survive routing"

    def test_with_runcontext_labels_merge(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file, labels={"team": "ml"}).run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["labels"]["team"] == "ml"
        assert sub["labels"]["routed-by"] == "consistent-hash"

    def test_caller_labels_win_on_conflict(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file, labels={"routed-by": "me"}).run(train, dataset="s3://b/x")
        (sub,) = submissions
        assert sub["labels"]["routed-by"] == "me"

    def test_pinned_profile_is_honoured(self, initialized, submissions, cfg_file) -> None:
        with_runcontext(config_file=cfg_file, profile="eu-west").run(train, dataset="s3://bucket/x")
        (sub,) = submissions
        assert sub["project"] == "west-proj"
        assert sub["name"] is None
