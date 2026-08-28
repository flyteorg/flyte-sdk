# test/cli/test_hello_world.py
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import flyte
from flyte.cli._deploy import HELLO_WORLD_APP_CMD, HELLO_WORLD_TASK_CMD, deploy
from flyte.cli._hello_world import (
    APP_EXAMPLE,
    TASK_EXAMPLE,
    endpoint_configured,
    get_hello_world_command,
    get_hello_world_task_deploy_command,
)
from flyte.cli._run import HELLO_WORLD_CMD, RunArguments, run


@pytest.fixture
def runner():
    return CliRunner()


def _patch_with_runcontext(monkeypatch, captured):
    """Replace flyte.with_runcontext with a stub that records kwargs and returns a no-op runner."""

    class _FakeResult:
        url = "local://fake"

        def outputs(self):
            return None

    class _FakeRun:
        async def aio(self, *args, **kwargs):
            return _FakeResult()

    class _FakeRunner:
        run = _FakeRun()

    def _fake_with_runcontext(*args, **kwargs):
        captured.update(kwargs)
        return _FakeRunner()

    monkeypatch.setattr(flyte, "with_runcontext", _fake_with_runcontext)


def _patch_deploy(monkeypatch, captured):
    """Replace flyte.deploy with a stub that records what it was handed."""

    class _FakeDeployment:
        def env_repr(self):
            return [[("name", "fake")]]

        def table_repr(self):
            return [[("name", "fake")]]

    def _fake_deploy(*envs, **kwargs):
        captured["envs"] = envs
        captured.update(kwargs)
        return [_FakeDeployment()]

    monkeypatch.setattr(flyte, "deploy", _fake_deploy)


# --- the examples themselves -------------------------------------------------


def test_materialize_writes_the_task_example(tmp_path):
    path = TASK_EXAMPLE.materialize(tmp_path)
    assert path == tmp_path / "hello_world.py"
    assert path.read_text() == TASK_EXAMPLE.source


def test_materialize_writes_the_app_example_with_its_index_html(tmp_path):
    path = APP_EXAMPLE.materialize(tmp_path)
    assert path == tmp_path / "hello_world_app.py"
    index = tmp_path / "index.html"
    assert index.exists()
    # The landing page's whole job is to point at the interactive docs.
    assert 'href="/docs"' in index.read_text()


def test_materialize_overwrites_a_stale_copy(tmp_path):
    stale = tmp_path / TASK_EXAMPLE.filename
    stale.write_text("# left behind by an older SDK\n")

    assert TASK_EXAMPLE.materialize(tmp_path).read_text() == TASK_EXAMPLE.source


def test_the_examples_get_separate_directories():
    # The code bundle is rooted at the example's directory, so anything sharing it ships too.
    assert TASK_EXAMPLE.directory != APP_EXAMPLE.directory


def test_load_loads_from_the_materialized_file(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, TASK_EXAMPLE.module, raising=False)
    path = TASK_EXAMPLE.materialize(tmp_path)

    task = TASK_EXAMPLE.load(path, "main")

    assert task.name == "hello_world.main"
    # The bundle resolves a task through its module, so the module has to be the file on
    # disk under the root dir -- not an equivalent copy inside the installed package.
    assert sys.modules[TASK_EXAMPLE.module].__file__ == str(path)


def test_load_reports_a_missing_dependency_by_name(tmp_path):
    import rich_click as click

    from flyte.cli._hello_world import Example

    broken = Example(slug="broken", filename="broken.py", module="broken", source="import not_a_real_package\n")
    path = broken.materialize(tmp_path)

    with pytest.raises(click.ClickException, match="not_a_real_package"):
        broken.load(path, "anything")


def test_the_app_example_declares_its_index_html_and_docs_link(tmp_path):
    path = APP_EXAMPLE.materialize(tmp_path)
    app_env = APP_EXAMPLE.load(path, "app_env")

    # index.html is not a Python module, so it only ships because the env names it.
    assert app_env.include == ("index.html",)
    assert "/docs" in [link.path for link in app_env.links]


def test_the_app_example_serves_its_endpoints(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    path = APP_EXAMPLE.materialize(tmp_path)
    app = APP_EXAMPLE.load(path, "app")

    with fastapi_testclient.TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["content-type"].startswith("text/html")
        assert "/docs" in index.text
        assert client.get("/docs").status_code == 200

        assert client.get("/hello", params={"name": "flyte"}).json() == {"message": "Hello, flyte!"}
        assert client.get("/line/4").json() == {"x": 4, "y": 13}
        assert client.post("/mean", json=[1, 2, 3]).json() == {"mean": 2.0}
        assert client.post("/mean", json=[]).status_code == 400
        assert client.get("/health").json() == {"status": "healthy"}


# --- endpoint detection ------------------------------------------------------


def test_endpoint_configured_reads_the_config(monkeypatch):
    monkeypatch.delenv("FLYTE_API_KEY", raising=False)

    no_endpoint = SimpleNamespace(endpoint=None, config=SimpleNamespace(platform=SimpleNamespace(endpoint=None)))
    assert not endpoint_configured(SimpleNamespace(obj=no_endpoint))

    from_config = SimpleNamespace(
        endpoint=None, config=SimpleNamespace(platform=SimpleNamespace(endpoint="dns:///flyte.example.com"))
    )
    assert endpoint_configured(SimpleNamespace(obj=from_config))

    from_flag = SimpleNamespace(endpoint="dns:///flyte.example.com", config=None)
    assert endpoint_configured(SimpleNamespace(obj=from_flag))


def test_endpoint_configured_honours_api_key(monkeypatch):
    monkeypatch.setenv("FLYTE_API_KEY", "some-key")
    assert endpoint_configured(SimpleNamespace(obj=None))


# --- flyte run hello-world ---------------------------------------------------


def test_hello_world_is_listed_on_run(runner):
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert HELLO_WORLD_CMD in result.output


def test_get_hello_world_command_roots_the_bundle_at_the_example(runner):
    run_args = RunArguments()
    with run.make_context("run", [HELLO_WORLD_CMD], obj=None) as ctx:
        cmd = get_hello_world_command(ctx, run_args)

    assert cmd.name == HELLO_WORLD_CMD
    assert run_args.root_dir == str(TASK_EXAMPLE.directory)
    assert (TASK_EXAMPLE.directory / TASK_EXAMPLE.filename).exists()


def test_hello_world_help_exposes_task_inputs(runner):
    result = runner.invoke(run, [HELLO_WORLD_CMD, "--help"])
    assert result.exit_code == 0, result.output
    assert "--x_list" in result.output


def test_hello_world_runs_locally(runner):
    try:
        result = runner.invoke(run, ["--local", HELLO_WORLD_CMD])
        assert result.exit_code == 0, result.output
        # mean of 2x + 5 over x in 0..9
        assert "14.0" in result.output
    except ValueError as ve:
        if "I/O operation on closed file" in str(ve):
            # https://github.com/pallets/click/issues/824
            return
        raise


def test_hello_world_takes_task_inputs(runner):
    try:
        result = runner.invoke(run, ["--local", HELLO_WORLD_CMD, "--x_list", "[1, 3]"])
        assert result.exit_code == 0, result.output
        # mean of 2x + 5 over [1, 3]
        assert "9.0" in result.output
    except ValueError as ve:
        if "I/O operation on closed file" in str(ve):
            return
        raise


def test_hello_world_falls_back_to_local_without_an_endpoint(runner, monkeypatch):
    captured = {}
    _patch_with_runcontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: False)

    result = runner.invoke(run, [HELLO_WORLD_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "local"
    assert "No Flyte endpoint is configured" in result.output


def test_hello_world_runs_remotely_when_an_endpoint_is_configured(runner, monkeypatch):
    captured = {}
    _patch_with_runcontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: True)

    result = runner.invoke(run, ["--project", "p", "--domain", "d", HELLO_WORLD_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "remote"
    assert "No Flyte endpoint is configured" not in result.output


# --- flyte deploy hello-world-task / hello-world-app -------------------------


def test_both_deploy_examples_are_listed(runner):
    result = runner.invoke(deploy, ["--help"])
    assert result.exit_code == 0, result.output
    assert HELLO_WORLD_TASK_CMD in result.output
    assert HELLO_WORLD_APP_CMD in result.output


def test_deploy_task_roots_the_bundle_at_the_example():
    from flyte.cli._deploy import DeployArguments

    deploy_args = DeployArguments()
    with deploy.make_context("deploy", [HELLO_WORLD_TASK_CMD], obj=None) as ctx:
        cmd = get_hello_world_task_deploy_command(ctx, deploy_args)

    assert cmd.name == HELLO_WORLD_TASK_CMD
    assert cmd.env_name == "hello_world"
    assert deploy_args.root_dir == str(TASK_EXAMPLE.directory)


def test_deploy_task_deploys_the_task_environment(runner, monkeypatch):
    captured = {}
    _patch_deploy(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: True)

    result = runner.invoke(deploy, ["--project", "p", "--domain", "d", HELLO_WORLD_TASK_CMD])
    assert result.exit_code == 0, result.output
    (env,) = captured["envs"]
    assert env.name == "hello_world"
    # The next step is the whole point of deploying it.
    assert "flyte run deployed-task hello_world.main" in result.output


def test_deploy_app_deploys_the_app_environment(runner, monkeypatch):
    pytest.importorskip("fastapi")
    captured = {}
    _patch_deploy(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: True)

    result = runner.invoke(deploy, ["--project", "p", "--domain", "d", HELLO_WORLD_APP_CMD])
    assert result.exit_code == 0, result.output
    (env,) = captured["envs"]
    assert env.name == "hello-world-app"
    assert env.include == ("index.html",)
    assert "/docs" in result.output


def test_deploy_without_an_endpoint_says_which_command_fixes_it(runner, monkeypatch):
    captured = {}
    _patch_deploy(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: False)

    result = runner.invoke(deploy, [HELLO_WORLD_TASK_CMD])
    assert result.exit_code != 0
    assert "flyte create config" in result.output
    # Nothing was deployed, rather than a connection error from deep in the client.
    assert "envs" not in captured


def test_deploy_dry_run_needs_no_endpoint(runner, monkeypatch):
    captured = {}
    _patch_deploy(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello_world.endpoint_configured", lambda ctx: False)

    result = runner.invoke(deploy, ["--dry-run", "--project", "p", "--domain", "d", HELLO_WORLD_TASK_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("dryrun") is True


# --- docs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["flyte run hello-world", "flyte deploy hello-world-task", "flyte deploy hello-world-app"],
)
def test_the_built_in_examples_are_documented(path):
    """The doc walker stops at `flyte run`/`flyte deploy` except for their static subcommands."""
    import click

    from flyte.cli._gen import walk_commands

    root = click.Group(name="root")
    root.add_command(run, name="run")
    root.add_command(deploy, name="deploy")

    paths = [p for p, _, _ in walk_commands(click.Context(root, info_name="flyte"))]
    assert path in paths
