# test/cli/test_hello.py
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import flyte
from flyte.cli._common import HELLO_CMD
from flyte.cli._hello import (
    APP_EXAMPLE,
    TASK_EXAMPLE,
    endpoint_configured,
    get_hello_run_command,
    get_hello_serve_command,
)
from flyte.cli._run import RunArguments, run
from flyte.cli._serve import ServeArguments, serve


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


def _patch_with_servecontext(monkeypatch, captured):
    """Replace flyte.with_servecontext with a stub that records what it was handed."""

    class _FakeApp:
        name = "hello-app"
        url = "https://fake.example.com/hello-app"
        endpoint = "http://localhost:8080"

        def activate(self, wait=False):
            captured["activated"] = True

        def deactivate(self):
            captured["deactivated"] = True

    class _FakeServe:
        async def aio(self, env, *args, **kwargs):
            captured["env"] = env
            return _FakeApp()

        def __call__(self, env, *args, **kwargs):
            captured["env"] = env
            return _FakeApp()

    class _FakeServer:
        serve = _FakeServe()

    def _fake_with_servecontext(*args, **kwargs):
        captured.update(kwargs)
        return _FakeServer()

    monkeypatch.setattr(flyte, "with_servecontext", _fake_with_servecontext)


# --- the examples themselves -------------------------------------------------


def test_materialize_writes_the_task_example(tmp_path):
    path = TASK_EXAMPLE.materialize(tmp_path)
    assert path == tmp_path / "hello.py"
    assert path.read_text() == TASK_EXAMPLE.source


def test_materialize_writes_the_app_example_with_its_index_html(tmp_path):
    path = APP_EXAMPLE.materialize(tmp_path)
    assert path == tmp_path / "hello_app.py"
    index = tmp_path / "index.html"
    assert index.exists()
    # The landing page's whole job is to point at the interactive docs.
    assert 'href="/docs"' in index.read_text()


def test_materialize_overwrites_a_stale_copy(tmp_path):
    stale = tmp_path / TASK_EXAMPLE.filename
    stale.write_text("# left behind by an older SDK\n")

    assert TASK_EXAMPLE.materialize(tmp_path).read_text() == TASK_EXAMPLE.source


def test_the_example_directories_are_already_resolved():
    """A symlinked temp dir would break the bundle.

    The code bundle lists a module by its `__file__` and takes that relative to the
    *resolved* root directory. On macOS `gettempdir()` sits under `/var`, a symlink to
    `/private/var`, so an unresolved directory makes every file look like it is outside
    the root and the run dies with "is not in the subpath of".
    """
    for example in (TASK_EXAMPLE, APP_EXAMPLE):
        assert example.directory == example.directory.resolve()


@pytest.mark.parametrize("example", [TASK_EXAMPLE, APP_EXAMPLE], ids=["task", "app"])
def test_the_example_bundles_from_its_scratch_directory(example, tmp_path):
    """Bundle the example for real -- the step that broke on macOS.

    `copy_code_bundle_to_context` resolves the root dir and then takes each loaded
    module's `__file__` relative to it. When the scratch path was unresolved the two
    disagreed and every remote `flyte run hello` died with "is not in the subpath of".
    """
    from flyte._code_bundle._utils import copy_code_bundle_to_context

    if Path(tempfile.gettempdir()).resolve() == Path(tempfile.gettempdir()):
        pytest.skip("temp dir is not a symlink on this platform")

    path = example.materialize()
    example.load(path, "main" if example is TASK_EXAMPLE else "app_env")

    dst = copy_code_bundle_to_context(root_dir=example.directory, copy_style="loaded_modules", context_path=tmp_path)

    assert example.filename in [f.name for f in Path(dst).rglob("*") if f.is_file()]


def test_the_examples_get_separate_directories():
    # The code bundle is rooted at the example's directory, so anything sharing it ships too.
    assert TASK_EXAMPLE.directory != APP_EXAMPLE.directory


def test_load_loads_from_the_materialized_file(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, TASK_EXAMPLE.module, raising=False)
    path = TASK_EXAMPLE.materialize(tmp_path)

    task = TASK_EXAMPLE.load(path, "main")

    assert task.name == "hello.main"
    # The task `main` fans out with `flyte.map`.
    assert TASK_EXAMPLE.load(path, "worker").name == "hello.worker"
    # The bundle resolves a task through its module, so the module has to be the file on
    # disk under the root dir -- not an equivalent copy inside the installed package.
    assert sys.modules[TASK_EXAMPLE.module].__file__ == str(path)


def test_load_reports_a_missing_dependency_by_name(tmp_path):
    import rich_click as click

    from flyte.cli._hello import Example

    broken = Example(slug="broken", filename="broken.py", module="broken", source="import not_a_real_package\n")
    path = broken.materialize(tmp_path)

    with pytest.raises(click.ClickException, match="not_a_real_package"):
        broken.load(path, "anything")


def test_the_app_example_declares_its_index_html_and_docs_link(tmp_path):
    path = APP_EXAMPLE.materialize(tmp_path)
    app_env = APP_EXAMPLE.load(path, "app_env")

    assert app_env.name == "hello-app"
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
        assert client.get("/worker/4").json() == {"x": 4, "y": 13}
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


# --- flyte run hello ---------------------------------------------------------


def test_hello_is_listed_on_run(runner):
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert HELLO_CMD in result.output


def test_get_hello_run_command_roots_the_bundle_at_the_example():
    run_args = RunArguments()
    with run.make_context("run", [HELLO_CMD], obj=None) as ctx:
        cmd = get_hello_run_command(ctx, run_args)

    assert cmd.name == HELLO_CMD
    assert run_args.root_dir == str(TASK_EXAMPLE.directory)
    assert (TASK_EXAMPLE.directory / TASK_EXAMPLE.filename).exists()


def test_hello_help_exposes_task_inputs(runner):
    result = runner.invoke(run, [HELLO_CMD, "--help"])
    assert result.exit_code == 0, result.output
    assert "--x_list" in result.output


def test_hello_runs_locally(runner):
    try:
        result = runner.invoke(run, ["--local", HELLO_CMD])
        assert result.exit_code == 0, result.output
        # mean of 2x + 5 over x in 0..9
        assert "14.0" in result.output
    except ValueError as ve:
        if "I/O operation on closed file" in str(ve):
            # https://github.com/pallets/click/issues/824
            return
        raise


def test_hello_takes_task_inputs(runner):
    try:
        result = runner.invoke(run, ["--local", HELLO_CMD, "--x_list", "[1, 3]"])
        assert result.exit_code == 0, result.output
        # mean of 2x + 5 over [1, 3]
        assert "9.0" in result.output
    except ValueError as ve:
        if "I/O operation on closed file" in str(ve):
            return
        raise


def test_hello_falls_back_to_local_without_an_endpoint(runner, monkeypatch):
    captured = {}
    _patch_with_runcontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello.endpoint_configured", lambda ctx: False)

    result = runner.invoke(run, [HELLO_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "local"
    assert "No Flyte endpoint is configured" in result.output


def test_hello_runs_remotely_when_an_endpoint_is_configured(runner, monkeypatch):
    captured = {}
    _patch_with_runcontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello.endpoint_configured", lambda ctx: True)

    result = runner.invoke(run, ["--project", "p", "--domain", "d", HELLO_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "remote"
    assert "No Flyte endpoint is configured" not in result.output


# --- flyte serve hello -------------------------------------------------------


def test_hello_is_listed_on_serve(runner):
    result = runner.invoke(serve, ["--help"])
    assert result.exit_code == 0, result.output
    assert HELLO_CMD in result.output


def test_get_hello_serve_command_roots_the_bundle_at_the_example():
    serve_args = ServeArguments()
    with serve.make_context("serve", [HELLO_CMD], obj=None) as ctx:
        cmd = get_hello_serve_command(ctx, serve_args)

    assert cmd.name == HELLO_CMD
    assert cmd.obj.name == "hello-app"
    assert serve_args.root_dir == str(APP_EXAMPLE.directory)
    assert (APP_EXAMPLE.directory / "index.html").exists()


def test_hello_serves_the_app_environment_remotely(runner, monkeypatch):
    captured = {}
    _patch_with_servecontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello.endpoint_configured", lambda ctx: True)

    result = runner.invoke(serve, ["--project", "p", "--domain", "d", HELLO_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "remote"
    assert captured["env"].name == "hello-app"
    assert captured["env"].include == ("index.html",)


def test_hello_falls_back_to_local_serving_without_an_endpoint(runner, monkeypatch):
    captured = {}
    _patch_with_servecontext(monkeypatch, captured)
    monkeypatch.setattr("flyte.cli._hello.endpoint_configured", lambda ctx: False)
    # The local path blocks on Ctrl+C once the app is up; stop there.
    monkeypatch.setattr("signal.pause", lambda: None, raising=False)

    result = runner.invoke(serve, [HELLO_CMD])
    assert result.exit_code == 0, result.output
    assert captured.get("mode") == "local"
    assert captured["env"].name == "hello-app"
    assert "No Flyte endpoint is configured" in result.output


# --- the deploy commands are gone --------------------------------------------


@pytest.mark.parametrize("name", ["hello-world-task", "hello-world-app"])
def test_deploy_has_no_built_in_examples(runner, name):
    """The built-in examples live on `run` and `serve`; `deploy` takes user files only."""
    from flyte.cli._deploy import deploy

    result = runner.invoke(deploy, ["--help"])
    assert result.exit_code == 0, result.output
    assert name not in result.output


# --- docs --------------------------------------------------------------------


@pytest.mark.parametrize("path", ["flyte run hello", "flyte serve hello"])
def test_the_built_in_examples_are_documented(path):
    """The doc walker stops at `flyte run`/`flyte serve` except for their static subcommands."""
    import click

    from flyte.cli._gen import walk_commands

    root = click.Group(name="root")
    root.add_command(run, name="run")
    root.add_command(serve, name="serve")

    paths = [p for p, _, _ in walk_commands(click.Context(root, info_name="flyte"))]
    assert path in paths
