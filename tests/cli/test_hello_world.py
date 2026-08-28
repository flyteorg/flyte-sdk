# test/cli/test_hello_world.py
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import flyte
from flyte.cli._hello_world import (
    HELLO_WORLD_CMD,
    HELLO_WORLD_FILENAME,
    HELLO_WORLD_MODULE,
    HELLO_WORLD_SOURCE,
    endpoint_configured,
    get_hello_world_command,
    hello_world_dir,
    load_task,
    materialize,
)
from flyte.cli._run import RunArguments, run


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


def test_hello_world_is_listed_on_run(runner):
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert HELLO_WORLD_CMD in result.output


def test_materialize_writes_the_example(tmp_path):
    path = materialize(tmp_path)
    assert path == tmp_path / HELLO_WORLD_FILENAME
    assert path.read_text() == HELLO_WORLD_SOURCE


def test_materialize_overwrites_a_stale_copy(tmp_path):
    stale = tmp_path / HELLO_WORLD_FILENAME
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("# left behind by an older SDK\n")

    assert materialize(tmp_path).read_text() == HELLO_WORLD_SOURCE


def test_load_task_loads_from_the_materialized_file(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, HELLO_WORLD_MODULE, raising=False)
    path = materialize(tmp_path)

    task = load_task(path)

    assert task.name == "hello_world.main"
    # The bundle resolves a task through its module, so the module has to be the file on
    # disk under the root dir -- not an equivalent copy inside the installed package.
    assert sys.modules[HELLO_WORLD_MODULE].__file__ == str(path)


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


def test_get_hello_world_command_roots_the_bundle_at_the_example(runner):
    run_args = RunArguments()
    with run.make_context("run", [HELLO_WORLD_CMD], obj=None) as ctx:
        cmd = get_hello_world_command(ctx, run_args)

    assert cmd.name == HELLO_WORLD_CMD
    assert run_args.root_dir == str(hello_world_dir())
    assert (hello_world_dir() / HELLO_WORLD_FILENAME).exists()


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


def test_hello_world_is_documented():
    """The doc walker stops at `flyte run` except for the subcommands that are not user files."""
    import click

    from flyte.cli._gen import walk_commands

    root = click.Group(name="root")
    root.add_command(run, name="run")

    paths = [path for path, _, _ in walk_commands(click.Context(root, info_name="flyte"))]
    assert "flyte run hello-world" in paths
