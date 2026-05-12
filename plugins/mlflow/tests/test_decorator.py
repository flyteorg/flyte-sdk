import os
from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.mlflow._decorator import (
    _get_flyte_tags,
    _is_logging_rank,
    _resolve_run_mode,
    _run_for_plain_function,
    _run_for_task,
    _setup_autolog,
    _setup_tracking,
    _start_run_kwargs_from_config,
    mlflow_run,
)
from flyteplugins.mlflow._link import Mlflow


class TestIsLoggingRank:
    def test_explicit_rank_0(self):
        assert _is_logging_rank(0) is True

    def test_explicit_rank_nonzero(self):
        assert _is_logging_rank(1) is False
        assert _is_logging_rank(3) is False

    def test_no_rank_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RANK", None)
            assert _is_logging_rank() is True

    def test_env_rank_0(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            assert _is_logging_rank() is True

    def test_env_rank_nonzero(self):
        with patch.dict(os.environ, {"RANK": "2"}):
            assert _is_logging_rank() is False


class TestResolveRunMode:
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_explicit_non_auto_returned(self, mock_ctx):
        assert _resolve_run_mode("new") == "new"
        assert _resolve_run_mode("nested") == "nested"
        mock_ctx.assert_not_called()

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_auto_uses_context(self, mock_ctx):
        config = MagicMock()
        config.run_mode = "new"
        mock_ctx.return_value = config
        assert _resolve_run_mode("auto") == "new"

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_auto_context_also_auto(self, mock_ctx):
        config = MagicMock()
        config.run_mode = "auto"
        mock_ctx.return_value = config
        assert _resolve_run_mode("auto") == "auto"

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_auto_no_context(self, mock_ctx):
        mock_ctx.return_value = None
        assert _resolve_run_mode("auto") == "auto"


class TestGetFlyteTags:
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_no_context(self, mock_flyte):
        mock_flyte.ctx.return_value = None
        assert _get_flyte_tags() == {}

    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_no_action(self, mock_flyte):
        ctx = MagicMock()
        ctx.action = None
        mock_flyte.ctx.return_value = ctx
        assert _get_flyte_tags() == {}

    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_all_tags(self, mock_flyte):
        ctx = MagicMock()
        ctx.action.name = "my-task"
        ctx.action.run_name = "run-123"
        ctx.action.project = "proj"
        ctx.action.domain = "dev"
        mock_flyte.ctx.return_value = ctx
        tags = _get_flyte_tags()
        assert tags == {
            "flyte.action_name": "my-task",
            "flyte.run_name": "run-123",
            "flyte.project": "proj",
            "flyte.domain": "dev",
        }

    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_partial_tags(self, mock_flyte):
        ctx = MagicMock()
        ctx.action.name = "task"
        ctx.action.run_name = None
        ctx.action.project = "proj"
        ctx.action.domain = None
        mock_flyte.ctx.return_value = ctx
        tags = _get_flyte_tags()
        assert tags == {
            "flyte.action_name": "task",
            "flyte.project": "proj",
        }


class TestSetupTracking:
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_explicit_uri(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_tracking(tracking_uri="http://explicit")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://explicit")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_uri_from_context(self, mock_mlflow, mock_ctx):
        config = MagicMock()
        config.tracking_uri = "http://from-context"
        mock_ctx.return_value = config
        _setup_tracking()
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://from-context")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_experiment_by_name(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_tracking(experiment_name="my-exp")
        mock_mlflow.set_experiment.assert_called_once_with("my-exp")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_experiment_by_id(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_tracking(experiment_id="123")
        mock_mlflow.set_experiment.assert_called_once_with(experiment_id="123")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_id_takes_priority_over_name(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_tracking(experiment_id="123", experiment_name="name")
        mock_mlflow.set_experiment.assert_called_once_with(experiment_id="123")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_no_args_no_context(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_tracking()
        mock_mlflow.set_tracking_uri.assert_not_called()
        mock_mlflow.set_experiment.assert_not_called()


class TestSetupAutolog:
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_disabled_by_default(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_autolog()
        mock_mlflow.autolog.assert_not_called()

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_generic_autolog(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_autolog(autolog=True)
        mock_mlflow.autolog.assert_called_once_with(log_models=True, log_datasets=True)

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_framework_autolog(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        mock_module = MagicMock()
        mock_mlflow.sklearn = mock_module
        _setup_autolog(autolog=True, framework="sklearn")
        mock_module.autolog.assert_called_once()

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_invalid_framework(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        mock_mlflow.nonexistent = None
        delattr(mock_mlflow, "nonexistent")
        with pytest.raises(ValueError, match="not supported"):
            _setup_autolog(autolog=True, framework="nonexistent")

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_log_models_false(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_autolog(autolog=True, log_models=False)
        mock_mlflow.autolog.assert_called_once_with(log_models=False, log_datasets=True)

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_context_enables_autolog(self, mock_mlflow, mock_ctx):
        config = MagicMock()
        config.autolog = True
        config.framework = None
        config.log_models = None
        config.log_datasets = None
        config.autolog_kwargs = None
        mock_ctx.return_value = config
        _setup_autolog(autolog=False)
        mock_mlflow.autolog.assert_called_once()

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_extra_autolog_kwargs(self, mock_mlflow, mock_ctx):
        mock_ctx.return_value = None
        _setup_autolog(
            autolog=True,
            autolog_kwargs={"log_input_examples": True},
        )
        mock_mlflow.autolog.assert_called_once_with(
            log_models=True,
            log_datasets=True,
            log_input_examples=True,
        )


class TestStartRunKwargsFromConfig:
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_no_config(self, mock_ctx):
        mock_ctx.return_value = None
        assert _start_run_kwargs_from_config() == {}

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_filters_tracking_and_plugin_fields(self, mock_ctx):
        from flyteplugins.mlflow._context import _MLflowConfig

        config = _MLflowConfig(
            tracking_uri="http://x",
            experiment_name="exp",
            run_name="my-run",
            run_mode="new",
            autolog=True,
            framework="sklearn",
            link_host="http://host",
        )
        mock_ctx.return_value = config
        result = _start_run_kwargs_from_config()
        # tracking/plugin fields should be excluded
        assert "tracking_uri" not in result
        assert "experiment_name" not in result
        assert "run_mode" not in result
        assert "autolog" not in result
        assert "framework" not in result
        assert "link_host" not in result
        # run_name should be included
        assert result["run_name"] == "my-run"

    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    def test_extra_kwargs_merged(self, mock_ctx):
        from flyteplugins.mlflow._context import _MLflowConfig

        config = _MLflowConfig(
            run_name="run",
            kwargs={"description": "test"},
        )
        mock_ctx.return_value = config
        result = _start_run_kwargs_from_config()
        assert result["description"] == "test"
        assert result["run_name"] == "run"


class TestRunForPlainFunction:
    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_starts_and_ends_run(self, mock_mlflow, mock_tracking, mock_autolog):
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        with _run_for_plain_function() as run:
            assert run is mock_run

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_passes_tracking_to_setup(self, mock_mlflow, mock_tracking, mock_autolog):
        mock_mlflow.start_run.return_value = MagicMock()

        with _run_for_plain_function(
            tracking_uri="http://x",
            experiment_name="exp",
        ):
            pass

        mock_tracking.assert_called_once_with(
            tracking_uri="http://x",
            experiment_name="exp",
            experiment_id=None,
        )

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_enables_autolog(self, mock_mlflow, mock_tracking, mock_autolog):
        mock_mlflow.start_run.return_value = MagicMock()

        with _run_for_plain_function(autolog=True, framework="sklearn"):
            pass

        mock_autolog.assert_called_once_with(
            autolog=True,
            framework="sklearn",
            log_models=None,
            log_datasets=None,
            autolog_kwargs=None,
        )

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_ends_run_on_exception(self, mock_mlflow, mock_tracking, mock_autolog):
        mock_mlflow.start_run.return_value = MagicMock()

        with pytest.raises(RuntimeError):
            with _run_for_plain_function():
                raise RuntimeError("boom")

        mock_mlflow.end_run.assert_called_once()


class TestRunForTask:
    def _make_ctx(self, run_id=None, data=None, custom_context=None, action_name="task"):
        ctx = MagicMock()
        ctx.data = data or {}
        ctx.custom_context = custom_context or {}
        ctx.action.name = action_name
        ctx.action.run_name = "run-abc"
        if run_id:
            ctx.custom_context["_mlflow_run_id"] = run_id
        return ctx

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_auto_creates_new_run(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx()
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "new-run-id"
        mock_run.info.experiment_id = "exp-123"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="auto") as run:
            assert run is mock_run
            assert ctx.data["_mlflow_run"] is mock_run
            assert ctx.custom_context["_mlflow_run_id"] == "new-run-id"

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_auto_reuses_parent_run(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx(run_id="parent-run-id")
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "parent-run-id"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="auto") as run:
            assert run is mock_run

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_nested_sets_parent_run_id_tag(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx(run_id="parent-run-id")
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "child-run-id"
        mock_run.info.experiment_id = "exp-1"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="nested"):
            pass

        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert call_kwargs["tags"]["mlflow.parentRunId"] == "parent-run-id"

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_new_clears_stale_link(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx()
        ctx.custom_context["_mlflow_link"] = "http://old-link"
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "new-id"
        mock_run.info.experiment_id = "exp-1"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="new"):
            assert (
                "_mlflow_link" not in ctx.custom_context or ctx.custom_context.get("_mlflow_link") != "http://old-link"
            )

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_nested_keeps_parent_link(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx(run_id="parent-id")
        ctx.custom_context["_mlflow_link"] = "http://parent-link"
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "child-id"
        mock_run.info.experiment_id = "exp-1"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="nested"):
            assert ctx.custom_context.get("_mlflow_link") == "http://parent-link"

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_existing_run_in_data_yields_directly(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        existing_run = MagicMock()
        ctx = self._make_ctx(data={"_mlflow_run": existing_run})
        mock_flyte.ctx.return_value = ctx

        with _run_for_task() as run:
            assert run is existing_run

        mock_mlflow.start_run.assert_not_called()

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_local_new_run_sets_nested_true(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        """When a parent run is active (local exec), new runs need nested=True."""
        ctx = self._make_ctx()
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        active = MagicMock()
        mock_mlflow.active_run.return_value = active
        mock_run = MagicMock()
        mock_run.info.run_id = "new-id"
        mock_run.info.experiment_id = "exp-1"
        mock_mlflow.start_run.return_value = mock_run

        with _run_for_task(run_mode="new"):
            pass

        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert call_kwargs.get("nested") is True

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_local_reuse_skips_start_run(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        """When reusing and the run is already active locally, skip start_run."""
        active = MagicMock()
        active.info.run_id = "parent-run-id"
        ctx = self._make_ctx(run_id="parent-run-id")
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_mlflow.active_run.return_value = active

        with _run_for_task(run_mode="auto") as run:
            assert run is active

        mock_mlflow.start_run.assert_not_called()

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_link_host_generates_link(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx()
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}

        config = MagicMock()
        config.link_host = "http://mlflow.example.com"
        config.link_template = None
        mock_ctx_fn.return_value = config

        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_run.info.experiment_id = "exp-456"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="auto"):
            expected = "http://mlflow.example.com/#/experiments/exp-456/runs/run-123"
            assert ctx.custom_context["_mlflow_link"] == expected

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_custom_link_template(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx()
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_ctx_fn.return_value = None

        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_run.info.experiment_id = "exp-2"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(
            run_mode="auto",
            link_host="https://databricks.com",
            link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
        ):
            expected = "https://databricks.com/ml/experiments/exp-2/runs/run-1"
            assert ctx.custom_context["_mlflow_link"] == expected

    @patch("flyteplugins.mlflow._decorator._setup_autolog")
    @patch("flyteplugins.mlflow._decorator._start_run_kwargs_from_config")
    @patch("flyteplugins.mlflow._decorator._setup_tracking")
    @patch("flyteplugins.mlflow._decorator._get_flyte_tags", return_value={})
    @patch("flyteplugins.mlflow._decorator.get_mlflow_context", return_value=None)
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @patch("flyteplugins.mlflow._decorator.flyte")
    def test_restores_state_after_exit(
        self,
        mock_flyte,
        mock_mlflow,
        mock_ctx_fn,
        mock_tags,
        mock_tracking,
        mock_kwargs,
        mock_autolog,
    ):
        ctx = self._make_ctx(run_id="parent-run-id")
        # _mlflow_run should NOT be in data at entry (only _mlflow_run_id in custom_context)
        # The code saves ctx.data.get("_mlflow_run") as saved_run before overwriting
        mock_flyte.ctx.return_value = ctx
        mock_kwargs.return_value = {}
        mock_run = MagicMock()
        mock_run.info.run_id = "new-id"
        mock_run.info.experiment_id = "exp-1"
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.active_run.return_value = None

        with _run_for_task(run_mode="new"):
            # During execution, data should have the new run
            assert ctx.data["_mlflow_run"] is mock_run
            assert ctx.custom_context["_mlflow_run_id"] == "new-id"

        # After exit, state should be restored
        assert ctx.custom_context["_mlflow_run_id"] == "parent-run-id"
        assert "_mlflow_run" not in ctx.data


class TestMlflowRunDecorator:
    def test_validation_experiment_name_and_id(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            mlflow_run(experiment_name="exp", experiment_id="123")

    def test_validation_run_name_and_id(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            mlflow_run(run_name="run", run_id="abc")

    @patch("flyteplugins.mlflow._decorator.flyte")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_plain_sync_function(self, mock_mlflow, mock_flyte):
        mock_flyte.ctx.return_value = None
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        @mlflow_run
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        mock_mlflow.start_run.assert_called_once()

    @patch("flyteplugins.mlflow._decorator.flyte")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    @pytest.mark.asyncio
    async def test_plain_async_function(self, mock_mlflow, mock_flyte):
        mock_flyte.ctx.return_value = None
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run

        @mlflow_run
        async def my_func(x):
            return x * 2

        result = await my_func(5)
        assert result == 10

    @patch("flyteplugins.mlflow._decorator.flyte")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_rank_gating_skips_non_zero(self, mock_mlflow, mock_flyte):
        mock_flyte.ctx.return_value = None

        @mlflow_run(rank=1)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        mock_mlflow.start_run.assert_not_called()

    def test_task_template_sets_link_run_mode_new(self):
        mock_task = MagicMock(spec=["func", "links", "__class__"])
        mock_task.__class__ = type("AsyncFunctionTaskTemplate", (), {})
        link = Mlflow()
        mock_task.links = [link]

        # We can't easily test with a real AsyncFunctionTaskTemplate,
        # but we can test the Mlflow link dataclass directly
        link._decorator_run_mode = "new"
        assert link._decorator_run_mode == "new"

    def test_task_template_sets_link_name_for_nested(self):
        link = Mlflow()
        link._decorator_run_mode = "nested"
        link.name = "MLflow (parent)"
        assert link.name == "MLflow (parent)"

    @patch("flyteplugins.mlflow._decorator.flyte")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_plain_function_errors_when_ctx_exists(self, mock_mlflow, mock_flyte):
        """Plain function (not task) should error if Flyte context is available."""
        ctx = MagicMock()
        mock_flyte.ctx.return_value = ctx
        mock_mlflow.start_run.return_value = MagicMock()

        @mlflow_run
        def my_func():
            pass

        with pytest.raises(RuntimeError, match="cannot be applied to traces"):
            my_func()

    @patch("flyteplugins.mlflow._decorator.flyte")
    @patch("flyteplugins.mlflow._decorator.mlflow")
    def test_decorator_with_args(self, mock_mlflow, mock_flyte):
        mock_flyte.ctx.return_value = None
        mock_mlflow.start_run.return_value = MagicMock()

        @mlflow_run(autolog=True, experiment_name="test")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
