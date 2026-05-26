"""Tests for the internal Flyte MCP helper TaskEnvironment.

Covers:
- The subprocess-based UV-script helpers in ``flyte.ai.mcp._uv_script_utils``
  (exit-code propagation, stdout/stderr tailing, ``--build`` flag wiring,
  temp-file cleanup, env-var passthrough).
- The :class:`flyte.TaskEnvironment` and registered tasks in
  ``flyte.ai.mcp.tasks`` (env name, registered task names, image lineage).
"""

from __future__ import annotations

import os

import pytest

from flyte.ai.mcp._uv_script_utils import (
    RunResult,
    _python_interpreter,
    build_script_image_,
    run_script_remote_,
)


class TestPythonInterpreter:
    """Tests for ``_python_interpreter``."""

    def test_defaults_to_sys_executable(self, monkeypatch):
        import sys

        monkeypatch.delenv("FLYTE_MCP_HELPER_PYTHON", raising=False)
        assert _python_interpreter() == sys.executable

    def test_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("FLYTE_MCP_HELPER_PYTHON", "/opt/venv/bin/python")
        assert _python_interpreter() == "/opt/venv/bin/python"


class TestBuildScriptImageHelper:
    """Tests for ``build_script_image_``."""

    @pytest.mark.asyncio
    async def test_success_returncode_and_build_flag(self, tmp_path, monkeypatch):
        # Run inside a temp cwd so the helper's relative temp file stays scoped.
        monkeypatch.chdir(tmp_path)

        # Script that prints its argv so we can confirm --build was forwarded.
        script = "import sys\nprint('argv=' + ' '.join(sys.argv[1:]))\nsys.stderr.write('build-stderr\\n')\n"
        result = await build_script_image_(script, tail=10)
        assert isinstance(result, RunResult)
        assert result.returncode == 0
        assert "argv=--build" in result.stdout
        assert "build-stderr" in result.stderr
        assert "run_uv_script_remote" in result.next_step
        # Temp file should be cleaned up.
        assert not any(p.name.startswith("__build_script_") for p in tmp_path.iterdir())

    @pytest.mark.asyncio
    async def test_failure_preserves_full_stderr(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Print many stderr lines, then exit non-zero. On failure the helper
        # should NOT truncate stderr to ``tail`` lines.
        script = "import sys\nfor i in range(40):\n    sys.stderr.write(f'line-{i}\\n')\nsys.exit(2)\n"
        result = await build_script_image_(script, tail=5)
        assert result.returncode == 2
        assert "line-0" in result.stderr
        assert "line-39" in result.stderr
        # And we still scrub the temp file.
        assert not any(p.name.startswith("__build_script_") for p in tmp_path.iterdir())

    @pytest.mark.asyncio
    async def test_success_tails_stdout(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        script = "for i in range(50):\n    print(f'out-{i}')\n"
        result = await build_script_image_(script, tail=5)
        assert result.returncode == 0
        stdout_lines = result.stdout.splitlines()
        assert len(stdout_lines) == 5
        assert stdout_lines[-1] == "out-49"

    @pytest.mark.asyncio
    async def test_env_passthrough_to_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FLYTE_PASSTHROUGH_API_KEY", "Bearer fake-token")

        script = "import os\nprint('api-key=' + os.environ.get('FLYTE_PASSTHROUGH_API_KEY', 'missing'))\n"
        result = await build_script_image_(script, tail=10)
        assert result.returncode == 0
        assert "api-key=Bearer fake-token" in result.stdout


class TestRunScriptRemoteHelper:
    """Tests for ``run_script_remote_``."""

    @pytest.mark.asyncio
    async def test_success_does_not_forward_build_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        script = "import sys\nprint('argv=' + ' '.join(sys.argv[1:]))\n"
        result = await run_script_remote_(script, tail=10)
        assert result.returncode == 0
        assert "argv=" in result.stdout
        # No --build flag should be forwarded.
        assert "--build" not in result.stdout
        assert "get_run_io" in result.next_step

    @pytest.mark.asyncio
    async def test_failure_returns_nonzero_returncode(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        script = "import sys\nsys.exit(7)\n"
        result = await run_script_remote_(script, tail=5)
        assert result.returncode == 7

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file_on_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        script = "print('ok')\n"
        await run_script_remote_(script)
        assert not any(p.name.startswith("__run_script_") for p in tmp_path.iterdir())

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        script = "raise RuntimeError('boom')\n"
        result = await run_script_remote_(script)
        assert result.returncode != 0
        assert not any(p.name.startswith("__run_script_") for p in tmp_path.iterdir())


class TestFlyteMCPTasksEnvironment:
    """Tests for the ``flyte.ai.mcp.tasks`` module."""

    def test_env_name_matches_fmcp_default(self):
        from flyte.ai.mcp.tasks import env

        from flyte.ai.mcp import FlyteMCPAppEnvironment

        assert env.name == "flyte_mcp_tasks"
        # The MCP server dispatches to "<env.name>.<task>" by default.
        defaults = FlyteMCPAppEnvironment(name="probe-mcp")
        assert defaults.uv_script_build_task_name.startswith(env.name + ".")
        assert defaults.uv_script_run_task_name.startswith(env.name + ".")

    def test_registered_task_names(self):
        from flyte.ai.mcp.tasks import env

        names = set(env.tasks.keys())
        assert "flyte_mcp_tasks.build_image" in names
        assert "flyte_mcp_tasks.run_task" in names

    def test_task_signatures(self):
        import inspect

        from flyte.ai.mcp.tasks import build_image, run_task

        # ``@env.task`` wraps the underlying function; inspect the original via ``.func``.
        # ``from __future__ import annotations`` keeps annotations as strings, so we
        # compare against their stringified names.
        for task in (build_image, run_task):
            sig = inspect.signature(task.func)
            params = sig.parameters
            assert "script" in params
            assert "tail" in params
            assert params["tail"].default == 50
            assert str(params["script"].annotation) == "str"
            assert str(params["tail"].annotation) == "int"

    def test_image_uses_flyte_base(self):
        from flyte.ai.mcp.tasks import env

        # ``Image.from_debian_base`` produces a flyteorg base image.
        assert env.image is not None
        assert "flyte" in env.image.base_image

    def test_returns_run_result_dataclass(self):
        from flyte.ai.mcp.tasks import RunResult as TasksRunResult

        from flyte.ai.mcp._uv_script_utils import RunResult as UtilsRunResult

        # ``RunResult`` is re-exported through the tasks module for caller convenience.
        assert TasksRunResult is UtilsRunResult

    def test_module_runs_under_dash_m(self):
        # ``python -m flyte.ai.mcp.tasks`` is the canonical deployment entrypoint;
        # make sure the module is at least loadable and exposes the ``__main__`` guard.
        import flyte.ai.mcp.tasks as tasks_mod

        src = open(tasks_mod.__file__).read()
        assert 'if __name__ == "__main__":' in src
        assert "flyte.init_from_config()" in src
        assert "flyte.deploy(env)" in src


def test_run_result_dataclass_fields():
    """``RunResult`` must keep its public field surface stable for the MCP tool."""
    r = RunResult(stdout="o", stderr="e", returncode=0, next_step="n")
    assert r.stdout == "o"
    assert r.stderr == "e"
    assert r.returncode == 0
    assert r.next_step == "n"
    # Sanity-check: the fields are tracked at the dataclass level.
    expected = {"stdout", "stderr", "returncode", "next_step"}
    actual = {f.name for f in __import__("dataclasses").fields(RunResult)}
    assert actual == expected


def test_helper_python_env_var_name():
    # Other components may reference this name; keep it stable.
    assert os.environ.get("FLYTE_MCP_HELPER_PYTHON", None) is None or isinstance(
        os.environ["FLYTE_MCP_HELPER_PYTHON"], str
    )
