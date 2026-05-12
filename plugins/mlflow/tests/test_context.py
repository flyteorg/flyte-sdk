import json
from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.mlflow._context import (
    _from_dict_helper,
    _MLflowConfig,
    _to_dict_helper,
    get_mlflow_context,
    mlflow_config,
)


class TestMLflowConfig:
    def test_defaults(self):
        config = _MLflowConfig()
        assert config.tracking_uri is None
        assert config.experiment_name is None
        assert config.experiment_id is None
        assert config.run_name is None
        assert config.run_id is None
        assert config.tags is None
        assert config.run_mode == "auto"
        assert config.autolog is False
        assert config.framework is None
        assert config.log_models is None
        assert config.log_datasets is None
        assert config.autolog_kwargs is None
        assert config.link_host is None
        assert config.link_template is None
        assert config.kwargs is None

    def test_all_fields(self):
        config = _MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test-exp",
            run_name="test-run",
            tags={"env": "test"},
            run_mode="new",
            autolog=True,
            framework="sklearn",
            log_models=True,
            log_datasets=False,
            autolog_kwargs={"log_input_examples": True},
            link_host="http://localhost:5000",
            link_template="{host}/runs/{run_id}",
            kwargs={"description": "test"},
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test-exp"
        assert config.run_mode == "new"
        assert config.autolog is True
        assert config.framework == "sklearn"
        assert config.log_models is True
        assert config.log_datasets is False
        assert config.link_host == "http://localhost:5000"


class TestToDictHelper:
    def test_skips_none_values(self):
        config = _MLflowConfig(tracking_uri="http://localhost:5000")
        d = _to_dict_helper(config)
        assert "mlflow_tracking_uri" in d
        assert "mlflow_experiment_name" not in d

    def test_prefixes_keys(self):
        config = _MLflowConfig(tracking_uri="http://localhost", run_mode="new")
        d = _to_dict_helper(config)
        assert d["mlflow_tracking_uri"] == "http://localhost"
        assert d["mlflow_run_mode"] == "new"

    def test_serializes_dict_as_json(self):
        config = _MLflowConfig(tags={"a": "b"})
        d = _to_dict_helper(config)
        assert d["mlflow_tags"] == json.dumps({"a": "b"})

    def test_serializes_bool_as_json(self):
        config = _MLflowConfig(autolog=True)
        d = _to_dict_helper(config)
        assert d["mlflow_autolog"] == "true"

    def test_serializes_list_as_json(self):
        config = _MLflowConfig(autolog_kwargs={"extras": [1, 2]})
        d = _to_dict_helper(config)
        parsed = json.loads(d["mlflow_autolog_kwargs"])
        assert parsed == {"extras": [1, 2]}

    def test_false_bool_is_included(self):
        config = _MLflowConfig(autolog=False)
        d = _to_dict_helper(config)
        assert d["mlflow_autolog"] == "false"

    def test_log_models_bool(self):
        config = _MLflowConfig(log_models=False)
        d = _to_dict_helper(config)
        assert d["mlflow_log_models"] == "false"


class TestFromDictHelper:
    def test_roundtrip(self):
        original = _MLflowConfig(
            tracking_uri="http://localhost",
            experiment_name="exp",
            tags={"a": "1"},
            run_mode="nested",
            autolog=True,
            framework="sklearn",
            log_models=True,
            log_datasets=False,
            link_host="http://host",
        )
        d = _to_dict_helper(original)
        restored = _from_dict_helper(_MLflowConfig, d)
        assert restored.tracking_uri == original.tracking_uri
        assert restored.experiment_name == original.experiment_name
        assert restored.tags == original.tags
        assert restored.run_mode == original.run_mode
        assert restored.autolog == original.autolog
        assert restored.framework == original.framework
        assert restored.log_models == original.log_models
        assert restored.log_datasets == original.log_datasets
        assert restored.link_host == original.link_host

    def test_ignores_non_mlflow_keys(self):
        d = {"mlflow_tracking_uri": "http://x", "other_key": "ignored"}
        config = _from_dict_helper(_MLflowConfig, d)
        assert config.tracking_uri == "http://x"

    def test_json_deserialize_dict(self):
        d = {"mlflow_tags": json.dumps({"k": "v"})}
        config = _from_dict_helper(_MLflowConfig, d)
        assert config.tags == {"k": "v"}

    def test_plain_string_stays_string(self):
        d = {"mlflow_tracking_uri": "not-json"}
        config = _from_dict_helper(_MLflowConfig, d)
        assert config.tracking_uri == "not-json"


class TestMlflowConfigFactory:
    def test_basic_creation(self):
        config = mlflow_config(tracking_uri="http://localhost:5000")
        assert isinstance(config, _MLflowConfig)
        assert config.tracking_uri == "http://localhost:5000"

    def test_experiment_name_and_id_exclusive(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            mlflow_config(experiment_name="exp", experiment_id="123")

    def test_run_name_and_id_exclusive(self):
        with pytest.raises(ValueError, match="Cannot provide both"):
            mlflow_config(run_name="run", run_id="abc")

    def test_extra_kwargs_stored(self):
        config = mlflow_config(description="test run")
        assert config.kwargs == {"description": "test run"}

    def test_no_extra_kwargs_is_none(self):
        config = mlflow_config()
        assert config.kwargs is None

    def test_autolog_config(self):
        config = mlflow_config(
            autolog=True,
            framework="sklearn",
            log_models=True,
            log_datasets=False,
            autolog_kwargs={"log_input_examples": True},
        )
        assert config.autolog is True
        assert config.framework == "sklearn"
        assert config.log_models is True
        assert config.log_datasets is False
        assert config.autolog_kwargs == {"log_input_examples": True}

    def test_link_config(self):
        config = mlflow_config(
            link_host="http://host",
            link_template="{host}/runs/{run_id}",
        )
        assert config.link_host == "http://host"
        assert config.link_template == "{host}/runs/{run_id}"


class TestConfigDictInterface:
    def test_keys(self):
        config = _MLflowConfig(tracking_uri="http://x")
        keys = list(config.keys())
        assert "mlflow_tracking_uri" in keys

    def test_items(self):
        config = _MLflowConfig(tracking_uri="http://x")
        items = dict(config.items())
        assert items["mlflow_tracking_uri"] == "http://x"

    def test_get(self):
        config = _MLflowConfig(tracking_uri="http://x")
        assert config.get("mlflow_tracking_uri") == "http://x"
        assert config.get("nonexistent", "default") == "default"


class TestGetMlflowContext:
    @patch("flyteplugins.mlflow._context.flyte")
    def test_returns_none_when_no_context(self, mock_flyte):
        mock_flyte.ctx.return_value = None
        assert get_mlflow_context() is None

    @patch("flyteplugins.mlflow._context.flyte")
    def test_returns_none_when_no_custom_context(self, mock_flyte):
        ctx = MagicMock()
        ctx.custom_context = None
        mock_flyte.ctx.return_value = ctx
        assert get_mlflow_context() is None

    @patch("flyteplugins.mlflow._context.flyte")
    def test_returns_none_when_no_mlflow_keys(self, mock_flyte):
        ctx = MagicMock()
        ctx.custom_context = {"other_key": "value"}
        mock_flyte.ctx.return_value = ctx
        assert get_mlflow_context() is None

    @patch("flyteplugins.mlflow._context.flyte")
    def test_returns_config_when_mlflow_keys_present(self, mock_flyte):
        ctx = MagicMock()
        ctx.custom_context = {
            "mlflow_tracking_uri": "http://localhost",
            "mlflow_run_mode": "new",
        }
        mock_flyte.ctx.return_value = ctx
        config = get_mlflow_context()
        assert config is not None
        assert config.tracking_uri == "http://localhost"
        assert config.run_mode == "new"


class TestContextManager:
    @patch("flyteplugins.mlflow._context.flyte")
    def test_context_manager_enter_exit(self, mock_flyte):
        ctx = MagicMock()
        ctx.custom_context = {"mlflow_tracking_uri": "http://original"}
        mock_flyte.ctx.return_value = ctx

        mock_ctx_mgr = MagicMock()
        mock_flyte.custom_context.return_value = mock_ctx_mgr

        config = _MLflowConfig(tracking_uri="http://override")
        config.__enter__()
        mock_flyte.custom_context.assert_called_once()
        mock_ctx_mgr.__enter__.assert_called_once()

        config.__exit__(None, None, None)
        mock_ctx_mgr.__exit__.assert_called_once()
