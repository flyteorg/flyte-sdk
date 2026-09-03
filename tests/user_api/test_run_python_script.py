"""Tests for flyte.run_python_script (public API) and _build_task."""

import dataclasses
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte
from flyte._run_python_script import (
    _build_dataclass_from_dict,
    _build_script_runner_task,
    _build_task,
    _deserialize_plugin_config,
    _plugin_config_qualname,
    _serialize_plugin_config,
    load_plugin_config,
    run_python_script,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _DummyNodeConfig:
    group_name: str
    replicas: int
    min_replicas: Optional[int] = None


@dataclass
class _DummyPluginConfig:
    nodes: List[_DummyNodeConfig]
    enabled: bool = False
    runtime_env: Optional[Dict[str, str]] = None


@pytest.fixture
def script(tmp_path):
    """Create a temporary .py script and return its path."""
    p = tmp_path / "my_script.py"
    p.write_text("print('hello')")
    return p


@pytest.fixture
def mock_remote():
    """Mock _Runner so no real remote call happens."""
    mock_run = MagicMock()
    mock_runner = MagicMock()
    mock_runner.run.aio = AsyncMock(return_value=mock_run)
    mock_run.wait.aio = AsyncMock()

    with patch("flyte._run._Runner", return_value=mock_runner) as mock_runner_cls:
        yield {
            "run": mock_run,
            "runner": mock_runner,
            "runner_cls": mock_runner_cls,
        }


# ---------------------------------------------------------- -----------------
# _build_task
# ---------------------------------------------------------------------------


class TestBuildTask:
    """Tests for the _build_task helper that creates the execute_script task."""

    def test_task_short_name(self):
        env = flyte.TaskEnvironment(name="test_env")
        task = _build_task(env, script_name="my_script.py", timeout=3600, short_name="my_script")
        assert task.short_name == "my_script"

    def test_task_short_name_custom(self):
        env = flyte.TaskEnvironment(name="test_env2")
        task = _build_task(env, script_name="script.py", timeout=3600, short_name="custom_name")
        assert task.short_name == "custom_name"

    def test_task_timeout(self):
        env = flyte.TaskEnvironment(name="test_env3")
        task = _build_task(env, script_name="script.py", timeout=7200, short_name="t")
        assert task.timeout == timedelta(seconds=7200)

    def test_task_registered_in_env(self):
        env = flyte.TaskEnvironment(name="test_env4")
        task = _build_task(env, script_name="script.py", timeout=3600, short_name="t")
        assert task in env.tasks.values()


# ---------------------------------------------------------------------------
# run_python_script -validation
# ---------------------------------------------------------------------------


class TestRunPythonScriptValidation:
    """Tests for input validation in run_python_script."""

    def test_script_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Script not found"):
            run_python_script(tmp_path / "nonexistent.py")

    def test_script_not_py(self, tmp_path):
        txt = tmp_path / "script.txt"
        txt.write_text("hello")
        with pytest.raises(ValueError, match=r"must be a \.py file"):
            run_python_script(txt)


# ---------------------------------------------------------------------------
# run_python_script -short_name / name
# ---------------------------------------------------------------------------


class TestRunPythonScriptShortName:
    """Tests that the task short_name is derived correctly."""

    def test_short_name_defaults_to_script_stem(self, script, mock_remote):
        """Without name=, short_name should be the script filename stem."""
        run_python_script(script)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.short_name == "my_script"

    def test_short_name_uses_name_param(self, script, mock_remote):
        """With name= provided, short_name should use that name."""
        run_python_script(script, name="custom-run")

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.short_name == "custom-run"

    def test_short_name_for_nested_script(self, tmp_path, mock_remote):
        """Short name should be just the stem, regardless of path depth."""
        nested = tmp_path / "sub" / "dir" / "train_model.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("pass")

        run_python_script(nested)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.short_name == "train_model"


# ---------------------------------------------------------------------------
# run_python_script -code bundle
# ---------------------------------------------------------------------------


class TestRunPythonScriptCodeBundle:
    """Tests that the runner is configured with custom copy_style."""

    def test_runner_uses_custom_copy_style(self, script, mock_remote):
        """Verify _Runner is constructed with copy_style='custom'."""
        run_python_script(script)

        mock_remote["runner_cls"].assert_called_once()
        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["copy_style"] == "custom"

    def test_runner_bundle_relative_paths(self, script, mock_remote):
        """Verify _Runner receives the script filename as bundle_relative_paths."""
        run_python_script(script)

        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["_bundle_relative_paths"] == (script.name,)

    def test_runner_bundle_from_dir(self, script, mock_remote):
        """Verify _Runner receives the script's parent as bundle_from_dir."""
        run_python_script(script)

        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["_bundle_from_dir"] == script.resolve().parent

    def test_task_has_internal_resolver(self, script, mock_remote):
        """Verify the task has an InternalTaskResolver attached."""
        from flyte._internal.resolvers.internal import InternalTaskResolver

        run_python_script(script)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert isinstance(task_arg.task_resolver, InternalTaskResolver)
        assert task_arg.task_resolver._kwargs["script_name"] == script.name

    def test_resolver_output_dir_none_by_default(self, script, mock_remote):
        """Verify the resolver has output_dir=None when not specified."""
        run_python_script(script)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.task_resolver._kwargs.get("output_dir") is None

    def test_resolver_output_dir_passed_through(self, script, mock_remote):
        """Verify the resolver receives the output_dir value."""
        run_python_script(script, output_dir="/tmp/results")

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.task_resolver._kwargs["output_dir"] == "/tmp/results"


# ---------------------------------------------------------------------------
# run_python_script -output_dir
# ---------------------------------------------------------------------------


class TestRunPythonScriptOutputDir:
    """Tests that the output_dir parameter is propagated correctly."""

    def test_output_dir_default_is_none(self, script, mock_remote):
        """Without output_dir=, the resolver should have output_dir=None."""
        from flyte._internal.resolvers.internal import InternalTaskResolver

        run_python_script(script)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        resolver = task_arg.task_resolver
        assert isinstance(resolver, InternalTaskResolver)
        assert resolver._kwargs.get("output_dir") is None

    def test_output_dir_passed_to_resolver(self, script, mock_remote):
        """output_dir= should be stored on the resolver for serialization."""
        run_python_script(script, output_dir="/tmp/output")

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.task_resolver._kwargs["output_dir"] == "/tmp/output"

    def test_resolver_loader_args_includes_output_dir(self):
        """loader_args should include output_dir when set."""
        from flyte._internal.resolvers.internal import InternalTaskResolver

        resolver = InternalTaskResolver(
            "flyte._run_python_script._build_script_runner_task",
            script_name="script.py",
            output_dir="/tmp/out",
            timeout=600,
        )
        args = resolver.loader_args(MagicMock())
        assert "output_dir" in args
        idx = args.index("output_dir")
        assert args[idx + 1] == "/tmp/out"

    def test_resolver_loader_args_excludes_output_dir_when_none(self):
        """loader_args should not include output_dir when None."""
        from flyte._internal.resolvers.internal import InternalTaskResolver

        resolver = InternalTaskResolver(
            "flyte._run_python_script._build_script_runner_task",
            script_name="script.py",
            timeout=600,
        )
        args = resolver.loader_args(MagicMock())
        assert "output_dir" not in args


# ---------------------------------------------------------------------------
# run_python_script -resources
# ---------------------------------------------------------------------------


class TestRunPythonScriptResources:
    """Tests that resource kwargs are constructed correctly."""

    def test_default_resources(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script)
            resources = spy.call_args[1]["resources"]
            assert resources.cpu == 4
            assert resources.memory == "16Gi"
            assert resources.gpu is None

    def test_custom_cpu_memory(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script, cpu=8, memory="32Gi")
            resources = spy.call_args[1]["resources"]
            assert resources.cpu == 8
            assert resources.memory == "32Gi"

    def test_gpu_resources(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script, gpu=2, gpu_type="A100")
            resources = spy.call_args[1]["resources"]
            assert resources.gpu == "A100:2"

    def test_no_gpu_when_zero(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script, gpu=0)
            resources = spy.call_args[1]["resources"]
            assert resources.gpu is None


# ---------------------------------------------------------------------------
# run_python_script -image construction
# ---------------------------------------------------------------------------


class TestRunPythonScriptImage:
    """Tests that the image argument is handled correctly."""

    def test_default_image_is_debian_base(self, script, mock_remote):
        with patch.object(flyte.Image, "from_debian_base", wraps=flyte.Image.from_debian_base) as spy:
            run_python_script(script, image=None)
            spy.assert_called_once_with(name="python-script-runner")

    def test_packages_list_builds_image_with_pip(self, script, mock_remote):
        with patch.object(flyte.Image, "from_debian_base", wraps=flyte.Image.from_debian_base) as spy:
            run_python_script(script, image=["torch", "numpy"])
            spy.assert_called_once_with(name="python-script-runner")

    def test_custom_image_passed_through(self, script, mock_remote):
        custom_img = flyte.Image.from_debian_base(name="custom-img")
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script, image=custom_img)
            assert spy.call_args[1]["image"] is custom_img


# ---------------------------------------------------------------------------
# run_python_script -queue
# ---------------------------------------------------------------------------


class TestRunPythonScriptQueue:
    """Tests that the queue option is passed through."""

    def test_queue_passed(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script, queue="my-queue")
            assert spy.call_args[1]["queue"] == "my-queue"

    def test_no_queue_by_default(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script)
            assert "queue" not in spy.call_args[1]


# ---------------------------------------------------------------------------
# run_python_script -wait
# ---------------------------------------------------------------------------


class TestRunPythonScriptWait:
    """Tests for the wait parameter."""

    def test_wait_true(self, script, mock_remote):
        run_python_script(script, wait=True)
        mock_remote["run"].wait.aio.assert_awaited_once_with(quiet=True)

    def test_wait_false(self, script, mock_remote):
        run_python_script(script, wait=False)
        mock_remote["run"].wait.aio.assert_not_awaited()

    def test_wait_default_is_false(self, script, mock_remote):
        run_python_script(script)
        mock_remote["run"].wait.aio.assert_not_awaited()


# ---------------------------------------------------------------------------
# run_python_script -extra_args
# ---------------------------------------------------------------------------


class TestRunPythonScriptExtraArgs:
    """Tests that extra_args are forwarded correctly."""

    def test_extra_args_passed(self, script, mock_remote):
        run_python_script(script, extra_args=["--lr", "0.01"])

        call_kwargs = mock_remote["runner"].run.aio.call_args[1]
        assert call_kwargs["args"] == ["--lr", "0.01"]

    def test_no_extra_args_defaults_to_empty_list(self, script, mock_remote):
        run_python_script(script)

        call_kwargs = mock_remote["runner"].run.aio.call_args[1]
        assert call_kwargs["args"] == []


# ---------------------------------------------------------------------------
# run_python_script -runcontext
# ---------------------------------------------------------------------------


class TestRunPythonScriptRunContext:
    """Tests that _Runner is constructed with correct parameters."""

    def test_runner_mode_remote(self, script, mock_remote):
        run_python_script(script)
        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["force_mode"] == "remote"
        assert call_kwargs["name"] is None
        assert call_kwargs["debug"] is False

    def test_runner_passes_name(self, script, mock_remote):
        run_python_script(script, name="my-run")
        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["name"] == "my-run"

    def test_runner_passes_debug(self, script, mock_remote):
        run_python_script(script, debug=True)
        call_kwargs = mock_remote["runner_cls"].call_args[1]
        assert call_kwargs["debug"] is True


# ---------------------------------------------------------------------------
# _build_dataclass_from_dict
# ---------------------------------------------------------------------------


class TestBuildDataclassFromDict:
    """Tests for the recursive dict -> dataclass constructor used by plugin config YAML."""

    def test_flat_fields(self):
        result = _build_dataclass_from_dict(_DummyNodeConfig, {"group_name": "workers", "replicas": 2})
        assert result == _DummyNodeConfig(group_name="workers", replicas=2)

    def test_nested_list_of_dataclasses(self):
        data = {
            "nodes": [
                {"group_name": "workers", "replicas": 2},
                {"group_name": "gpu", "replicas": 1, "min_replicas": 1},
            ]
        }
        result = _build_dataclass_from_dict(_DummyPluginConfig, data)
        assert result.nodes == [
            _DummyNodeConfig(group_name="workers", replicas=2),
            _DummyNodeConfig(group_name="gpu", replicas=1, min_replicas=1),
        ]

    def test_dict_field_passthrough(self):
        data = {"nodes": [], "runtime_env": {"pip": "requests"}}
        result = _build_dataclass_from_dict(_DummyPluginConfig, data)
        assert result.runtime_env == {"pip": "requests"}

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="Unknown field"):
            _build_dataclass_from_dict(_DummyNodeConfig, {"group_name": "x", "replicas": 1, "bogus": 1})

    def test_non_dataclass_raises(self):
        with pytest.raises(ValueError, match="not a dataclass"):
            _build_dataclass_from_dict(dict, {"a": 1})


# ---------------------------------------------------------------------------
# load_plugin_config
# ---------------------------------------------------------------------------


class TestLoadPluginConfig:
    """Tests for loading a plugin config instance from a YAML file."""

    def test_loads_yaml_with_nested_config(self, tmp_path):
        qualified_name = f"{_DummyPluginConfig.__module__}.{_DummyPluginConfig.__qualname__}"
        yaml_path = tmp_path / "plugin.yaml"
        yaml_path.write_text(
            f"plugin: {qualified_name}\nconfig:\n  nodes:\n    - group_name: workers\n      replicas: 2\n"
            "  enabled: true\n"
        )
        result = load_plugin_config(yaml_path)
        assert result == _DummyPluginConfig(nodes=[_DummyNodeConfig(group_name="workers", replicas=2)], enabled=True)

    def test_missing_plugin_key_raises(self, tmp_path):
        yaml_path = tmp_path / "plugin.yaml"
        yaml_path.write_text("config:\n  a: 1\n")
        with pytest.raises(ValueError, match="top-level 'plugin' key"):
            load_plugin_config(yaml_path)

    def test_unresolvable_plugin_class_raises(self, tmp_path):
        yaml_path = tmp_path / "plugin.yaml"
        yaml_path.write_text("plugin: nonexistent.module.ClassName\n")
        with pytest.raises(ValueError, match="Could not load plugin config class"):
            load_plugin_config(yaml_path)


# ---------------------------------------------------------------------------
# plugin config serialization round trip (client -> remote container)
# ---------------------------------------------------------------------------


class TestPluginConfigSerializationRoundTrip:
    def test_round_trip(self):
        cfg = _DummyPluginConfig(nodes=[_DummyNodeConfig(group_name="workers", replicas=2)], enabled=True)
        qualified_name = _plugin_config_qualname(cfg)
        data = _serialize_plugin_config(cfg)
        restored = _deserialize_plugin_config(qualified_name, data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# run_python_script - plugin_config
# ---------------------------------------------------------------------------


class TestRunPythonScriptPluginConfig:
    """Tests that plugin_config is threaded through TaskEnvironment and the resolver."""

    def test_plugin_config_passed_to_environment(self, script, mock_remote):
        cfg = _DummyPluginConfig(nodes=[_DummyNodeConfig(group_name="workers", replicas=2)])
        with (
            patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as env_spy,
            patch("flyte._run_python_script._build_task", return_value=MagicMock()),
        ):
            run_python_script(script, plugin_config=cfg)
            assert env_spy.call_args[1]["plugin_config"] is cfg

    def test_no_plugin_config_by_default(self, script, mock_remote):
        with patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy:
            run_python_script(script)
            assert "plugin_config" not in spy.call_args[1]

    def test_plugin_config_forwarded_to_resolver(self, script, mock_remote):
        cfg = _DummyPluginConfig(nodes=[_DummyNodeConfig(group_name="workers", replicas=2)])
        with patch("flyte._run_python_script._build_task", return_value=MagicMock()) as build_task_spy:
            run_python_script(script, plugin_config=cfg)

        resolver = build_task_spy.call_args.kwargs["task_resolver"]
        assert resolver._kwargs["plugin_config_class"] == _plugin_config_qualname(cfg)
        assert json.loads(resolver._kwargs["plugin_config_data"]) == dataclasses.asdict(cfg)

    def test_no_plugin_config_kwargs_on_resolver_by_default(self, script, mock_remote):
        with patch("flyte._run_python_script._build_task", return_value=MagicMock()) as build_task_spy:
            run_python_script(script)

        resolver = build_task_spy.call_args.kwargs["task_resolver"]
        assert "plugin_config_class" not in resolver._kwargs or resolver._kwargs["plugin_config_class"] is None


# ---------------------------------------------------------------------------
# _build_script_runner_task - plugin_config reconstruction
# ---------------------------------------------------------------------------


class TestBuildScriptRunnerTaskPluginConfig:
    """Tests that the remote-side task builder re-hydrates plugin_config."""

    def test_plugin_config_deserialized_and_passed(self):
        cfg = _DummyPluginConfig(nodes=[_DummyNodeConfig(group_name="workers", replicas=2)])
        qualified_name = _plugin_config_qualname(cfg)
        data = _serialize_plugin_config(cfg)
        with (
            patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy,
            patch("flyte._run_python_script._build_task", return_value=MagicMock()),
        ):
            _build_script_runner_task("script.py", plugin_config_class=qualified_name, plugin_config_data=data)
            assert spy.call_args[1]["plugin_config"] == cfg

    def test_no_plugin_config_when_unset(self):
        with (
            patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy,
            patch("flyte._run_python_script._build_task", return_value=MagicMock()),
        ):
            _build_script_runner_task("script.py")
            assert spy.call_args[1]["plugin_config"] is None

    def test_clustered_marker_reconstructs_clustered_plugin(self):
        from flyte.clustered._task import _ClusteredPlugin

        with (
            patch("flyte.TaskEnvironment", wraps=flyte.TaskEnvironment) as spy,
            patch("flyte._run_python_script._build_task", return_value=MagicMock()),
        ):
            _build_script_runner_task("script.py", clustered="1")
            assert isinstance(spy.call_args[1]["plugin_config"], _ClusteredPlugin)


# ---------------------------------------------------------------------------
# run_python_script - clustered
# ---------------------------------------------------------------------------


class TestRunPythonScriptClustered:
    """Tests for clustered=True routing to ClusteredTaskEnvironment."""

    def test_requires_replicas_and_nproc_per_node(self, script):
        with pytest.raises(ValueError, match="requires both replicas and nproc_per_node"):
            run_python_script(script, clustered=True)

    def test_requires_nproc_per_node_even_with_replicas(self, script):
        with pytest.raises(ValueError, match="requires both replicas and nproc_per_node"):
            run_python_script(script, clustered=True, replicas=2)

    def test_plugin_config_mutually_exclusive_with_clustered(self, script):
        cfg = _DummyPluginConfig(nodes=[])
        with pytest.raises(ValueError, match="cannot be combined with clustered=True"):
            run_python_script(script, clustered=True, replicas=2, nproc_per_node=1, plugin_config=cfg)

    def test_clustered_only_kwargs_require_clustered_flag(self, script):
        with pytest.raises(ValueError, match="only apply when clustered=True"):
            run_python_script(script, replicas=2, nproc_per_node=1)

    def test_ttl_requires_clustered_flag(self, script):
        with pytest.raises(ValueError, match="only apply when clustered=True"):
            run_python_script(script, ttl_seconds_after_finished=60)

    def test_clustered_environment_constructed(self, script, mock_remote):
        import flyte.clustered

        with patch("flyte.clustered.ClusteredTaskEnvironment", wraps=flyte.clustered.ClusteredTaskEnvironment) as spy:
            run_python_script(script, clustered=True, replicas=2, nproc_per_node=4, gpu=4)
        kwargs = spy.call_args[1]
        assert kwargs["replicas"] == 2
        assert kwargs["nproc_per_node"] == 4

    def test_clustered_default_runtime_and_failure_policy(self, script, mock_remote):
        import flyte.clustered

        run_python_script(script, clustered=True, replicas=1, nproc_per_node=1)

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        env = task_arg.parent_env()
        assert isinstance(env.runtime, flyte.clustered.TorchRun)
        assert env.runtime.rdzv_backend == "static"
        assert env.failure_policy.max_restarts == 0
        assert env.failure_policy.restart_on_host_maintenance is False

    def test_clustered_custom_runtime_and_failure_policy(self, script, mock_remote):
        import flyte.clustered

        run_python_script(
            script,
            clustered=True,
            replicas=1,
            nproc_per_node=1,
            runtime=flyte.clustered.TorchRun(rdzv_backend="c10d", max_restarts=3),
            failure_policy=flyte.clustered.ClusterFailurePolicy(max_restarts=5, restart_on_host_maintenance=True),
            ttl_seconds_after_finished=120,
        )

        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        env = task_arg.parent_env()
        assert env.runtime.rdzv_backend == "c10d"
        assert env.runtime.max_restarts == 3
        assert env.failure_policy.max_restarts == 5
        assert env.failure_policy.restart_on_host_maintenance is True
        assert env.ttl_seconds_after_finished == 120

    def test_resolver_gets_clustered_marker(self, script, mock_remote):
        run_python_script(script, clustered=True, replicas=1, nproc_per_node=1)
        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.task_resolver._kwargs["clustered"] == "1"

    def test_resolver_no_clustered_marker_by_default(self, script, mock_remote):
        run_python_script(script)
        task_arg = mock_remote["runner"].run.aio.call_args[0][0]
        assert task_arg.task_resolver._kwargs.get("clustered") is None
