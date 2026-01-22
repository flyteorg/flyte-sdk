"""Tests for wandb context management."""

import json
from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.wandb import (
    _WandBConfig,
    _WandBSweepConfig,
    get_wandb_context,
    get_wandb_sweep_context,
    wandb_config,
    wandb_sweep_config,
)


class TestWandBConfig:
    """Tests for _WandBConfig class."""

    def test_wandb_config_creation(self):
        """Test creating a WandBConfig instance."""
        config = _WandBConfig(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            config={"learning_rate": 0.01},
        )

        assert config.project == "test-project"
        assert config.entity == "test-entity"
        assert config.tags == ["tag1", "tag2"]
        assert config.config == {"learning_rate": 0.01}

    def test_wandb_config_to_dict(self):
        """Test converting WandBConfig to dict."""
        config = _WandBConfig(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            config={"learning_rate": 0.01},
        )

        result = config.to_dict()

        assert result["wandb_project"] == "test-project"
        assert result["wandb_entity"] == "test-entity"
        assert result["wandb_tags"] == json.dumps(["tag1", "tag2"])
        assert result["wandb_config"] == json.dumps({"learning_rate": 0.01})

    def test_wandb_config_to_dict_with_none_values(self):
        """Test to_dict skips None values."""
        config = _WandBConfig(project="test-project", entity=None)

        result = config.to_dict()

        assert "wandb_project" in result
        assert "wandb_entity" not in result

    def test_wandb_config_from_dict(self):
        """Test creating WandBConfig from dict."""
        input_dict = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "wandb_tags": json.dumps(["tag1", "tag2"]),
            "wandb_config": json.dumps({"learning_rate": 0.01}),
        }

        config = _WandBConfig.from_dict(input_dict)

        assert config.project == "test-project"
        assert config.entity == "test-entity"
        assert config.tags == ["tag1", "tag2"]
        assert config.config == {"learning_rate": 0.01}

    def test_wandb_config_from_dict_excludes_sweep_keys(self):
        """Test from_dict excludes wandb_sweep_* keys."""
        input_dict = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "wandb_sweep_method": "random",  # Should be excluded
            "wandb_sweep_metric": json.dumps({"name": "loss"}),  # Should be excluded
        }

        config = _WandBConfig.from_dict(input_dict)

        assert config.project == "test-project"
        assert config.entity == "test-entity"
        # Sweep keys should not create attributes
        assert not hasattr(config, "sweep_method")
        assert not hasattr(config, "sweep_metric")

    def test_wandb_config_dict_protocol_keys(self):
        """Test keys() method of dict protocol."""
        config = _WandBConfig(project="test-project", entity="test-entity")

        keys = list(config.keys())

        assert "wandb_project" in keys
        assert "wandb_entity" in keys

    def test_wandb_config_dict_protocol_getitem(self):
        """Test __getitem__ method of dict protocol."""
        config = _WandBConfig(project="test-project", entity="test-entity")

        assert config["wandb_project"] == "test-project"
        assert config["wandb_entity"] == "test-entity"

    def test_wandb_config_dict_protocol_items(self):
        """Test items() method of dict protocol."""
        config = _WandBConfig(project="test-project", entity="test-entity")

        items = dict(config.items())

        assert items["wandb_project"] == "test-project"
        assert items["wandb_entity"] == "test-entity"

    def test_wandb_config_dict_protocol_get(self):
        """Test get() method of dict protocol."""
        config = _WandBConfig(project="test-project", entity="test-entity")

        assert config.get("wandb_project") == "test-project"
        assert config.get("wandb_entity") == "test-entity"
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    @patch("flyteplugins.wandb.context.flyte.ctx")
    def test_wandb_config_dict_protocol_setitem(self, mock_ctx):
        """Test __setitem__ method updates Flyte context."""
        mock_context = MagicMock()
        # custom_context needs to be a real dict (not empty) to pass the truthiness check
        mock_context.custom_context = {"existing": "value"}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project")
        config["wandb_entity"] = "new-entity"

        assert mock_context.custom_context["wandb_entity"] == "new-entity"

    @patch("flyte.ctx")
    def test_wandb_config_dict_protocol_delitem(self, mock_ctx):
        """Test __delitem__ method updates Flyte context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"wandb_entity": "test-entity"}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project")
        del config["wandb_entity"]

        assert "wandb_entity" not in mock_context.custom_context

    @patch("flyte.ctx")
    def test_wandb_config_dict_protocol_pop(self, mock_ctx):
        """Test pop() method updates Flyte context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"wandb_entity": "test-entity"}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project")
        result = config.pop("wandb_entity")

        assert result == "test-entity"
        assert "wandb_entity" not in mock_context.custom_context

    @patch("flyte.ctx")
    def test_wandb_config_dict_protocol_pop_with_default(self, mock_ctx):
        """Test pop() with default value."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project")
        result = config.pop("nonexistent", "default-value")

        assert result == "default-value"

    @patch("flyteplugins.wandb.context.flyte.ctx")
    def test_wandb_config_dict_protocol_update(self, mock_ctx):
        """Test update() method updates Flyte context."""
        mock_context = MagicMock()
        # custom_context needs to be a real dict (not empty) to pass the truthiness check
        mock_context.custom_context = {"existing": "value"}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project")
        config.update({"wandb_entity": "new-entity", "wandb_mode": "offline"})

        assert mock_context.custom_context["wandb_entity"] == "new-entity"
        assert mock_context.custom_context["wandb_mode"] == "offline"

    @patch("flyte.ctx")
    def test_wandb_config_context_manager_enter(self, mock_ctx):
        """Test context manager __enter__ sets up context."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project", entity="test-entity")

        with patch("flyte.custom_context") as mock_custom_context:
            mock_cm = MagicMock()
            mock_custom_context.return_value = mock_cm

            config.__enter__()

            # Should call flyte.custom_context with the config dict
            mock_custom_context.assert_called_once()
            mock_cm.__enter__.assert_called_once()

    @patch("flyte.ctx")
    def test_wandb_config_context_manager_exit(self, mock_ctx):
        """Test context manager __exit__ restores previous context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"wandb_project": "old-project"}
        mock_ctx.return_value = mock_context

        config = _WandBConfig(project="test-project", entity="test-entity")

        with patch("flyte.custom_context") as mock_custom_context:
            mock_cm = MagicMock()
            mock_custom_context.return_value = mock_cm

            config.__enter__()
            config.__exit__(None, None, None)

            mock_cm.__exit__.assert_called_once()

    def test_wandb_config_helper_function(self):
        """Test wandb_config() helper function."""
        config = wandb_config(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            config={"lr": 0.01},
            mode="offline",
            custom_param="value",
        )

        assert isinstance(config, _WandBConfig)
        assert config.project == "test-project"
        assert config.entity == "test-entity"
        assert config.tags == ["tag1", "tag2"]
        assert config.config == {"lr": 0.01}
        assert config.mode == "offline"
        assert config.kwargs == {"custom_param": "value"}

    def test_wandb_config_with_run_mode(self):
        """Test wandb_config() with run_mode parameter."""
        config = wandb_config(
            project="test-project",
            entity="test-entity",
            run_mode="shared",
        )

        assert config.run_mode == "shared"

    def test_wandb_config_default_run_mode(self):
        """Test that wandb_config() has 'auto' as default run_mode."""
        config = wandb_config(project="test-project", entity="test-entity")

        assert config.run_mode == "auto"

    def test_wandb_config_run_mode_to_dict(self):
        """Test that run_mode is included in to_dict()."""
        config = _WandBConfig(
            project="test-project",
            run_mode="new",
        )

        result = config.to_dict()

        assert result["wandb_run_mode"] == "new"

    def test_wandb_config_run_mode_from_dict(self):
        """Test that run_mode is parsed from dict."""
        input_dict = {
            "wandb_project": "test-project",
            "wandb_run_mode": "shared",
        }

        config = _WandBConfig.from_dict(input_dict)

        assert config.project == "test-project"
        assert config.run_mode == "shared"

    def test_wandb_config_download_logs_serialization_true(self):
        """Test that download_logs=True serializes correctly as JSON boolean."""
        config = _WandBConfig(
            project="test-project",
            download_logs=True,
        )

        result = config.to_dict()

        # Should be JSON "true" (lowercase), not Python "True"
        assert result["wandb_download_logs"] == "true"
        assert result["wandb_download_logs"] == json.dumps(True)

    def test_wandb_config_download_logs_serialization_false(self):
        """Test that download_logs=False serializes correctly as JSON boolean."""
        config = _WandBConfig(
            project="test-project",
            download_logs=False,
        )

        result = config.to_dict()

        # Should be JSON "false" (lowercase), not Python "False"
        assert result["wandb_download_logs"] == "false"
        assert result["wandb_download_logs"] == json.dumps(False)

    def test_wandb_config_download_logs_deserialization_true(self):
        """Test that download_logs deserializes correctly from JSON true."""
        input_dict = {
            "wandb_project": "test-project",
            "wandb_download_logs": "true",  # JSON format
        }

        config = _WandBConfig.from_dict(input_dict)

        assert config.project == "test-project"
        assert config.download_logs is True
        assert isinstance(config.download_logs, bool)

    def test_wandb_config_download_logs_deserialization_false(self):
        """Test that download_logs deserializes correctly from JSON false."""
        input_dict = {
            "wandb_project": "test-project",
            "wandb_download_logs": "false",  # JSON format
        }

        config = _WandBConfig.from_dict(input_dict)

        assert config.project == "test-project"
        assert config.download_logs is False
        assert isinstance(config.download_logs, bool)

    def test_wandb_config_download_logs_roundtrip(self):
        """Test that download_logs survives a roundtrip through serialization."""
        # Test with True
        config_true = _WandBConfig(project="test-project", download_logs=True)
        dict_true = config_true.to_dict()
        restored_true = _WandBConfig.from_dict(dict_true)
        assert restored_true.download_logs is True
        assert isinstance(restored_true.download_logs, bool)

        # Test with False
        config_false = _WandBConfig(project="test-project", download_logs=False)
        dict_false = config_false.to_dict()
        restored_false = _WandBConfig.from_dict(dict_false)
        assert restored_false.download_logs is False
        assert isinstance(restored_false.download_logs, bool)


class TestWandBSweepConfig:
    """Tests for _WandBSweepConfig class."""

    def test_wandb_sweep_config_creation(self):
        """Test creating a WandBSweepConfig instance."""
        config = _WandBSweepConfig(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
            parameters={"lr": {"min": 0.001, "max": 0.1}},
            project="test-project",
            entity="test-entity",
        )

        assert config.method == "random"
        assert config.metric == {"name": "loss", "goal": "minimize"}
        assert config.parameters == {"lr": {"min": 0.001, "max": 0.1}}
        assert config.project == "test-project"
        assert config.entity == "test-entity"

    def test_wandb_sweep_config_to_sweep_config(self):
        """Test converting to wandb.sweep() compatible dict."""
        config = _WandBSweepConfig(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
            parameters={"lr": {"min": 0.001, "max": 0.1}},
            project="test-project",
            entity="test-entity",
            prior_runs=["run1", "run2"],
            kwargs={"early_terminate": {"type": "hyperband"}},
        )

        sweep_config = config.to_sweep_config()

        # Should include sweep config fields
        assert sweep_config["method"] == "random"
        assert sweep_config["metric"] == {"name": "loss", "goal": "minimize"}
        assert sweep_config["parameters"] == {"lr": {"min": 0.001, "max": 0.1}}

        # Should merge kwargs
        assert sweep_config["early_terminate"] == {"type": "hyperband"}

        # Should exclude metadata fields
        assert "project" not in sweep_config
        assert "entity" not in sweep_config
        assert "prior_runs" not in sweep_config
        assert "kwargs" not in sweep_config

    def test_wandb_sweep_config_to_dict(self):
        """Test converting WandBSweepConfig to dict."""
        config = _WandBSweepConfig(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
            parameters={"lr": {"min": 0.001, "max": 0.1}},
        )

        result = config.to_dict()

        assert result["wandb_sweep_method"] == "random"
        assert result["wandb_sweep_metric"] == json.dumps(
            {"name": "loss", "goal": "minimize"}
        )
        assert result["wandb_sweep_parameters"] == json.dumps(
            {"lr": {"min": 0.001, "max": 0.1}}
        )

    def test_wandb_sweep_config_from_dict(self):
        """Test creating WandBSweepConfig from dict."""
        input_dict = {
            "wandb_sweep_method": "random",
            "wandb_sweep_metric": json.dumps({"name": "loss", "goal": "minimize"}),
            "wandb_sweep_parameters": json.dumps({"lr": {"min": 0.001, "max": 0.1}}),
        }

        config = _WandBSweepConfig.from_dict(input_dict)

        assert config.method == "random"
        assert config.metric == {"name": "loss", "goal": "minimize"}
        assert config.parameters == {"lr": {"min": 0.001, "max": 0.1}}

    def test_wandb_sweep_config_dict_protocol_get(self):
        """Test get() method of dict protocol."""
        config = _WandBSweepConfig(method="random", project="test-project")

        assert config.get("wandb_sweep_method") == "random"
        assert config.get("wandb_sweep_project") == "test-project"
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    @patch("flyte.ctx")
    def test_wandb_sweep_config_dict_protocol_pop(self, mock_ctx):
        """Test pop() method updates Flyte context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"wandb_sweep_method": "random"}
        mock_ctx.return_value = mock_context

        config = _WandBSweepConfig(method="random")
        result = config.pop("wandb_sweep_method")

        assert result == "random"
        assert "wandb_sweep_method" not in mock_context.custom_context

    def test_wandb_sweep_config_helper_function(self):
        """Test wandb_sweep_config() helper function."""
        config = wandb_sweep_config(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
            parameters={"lr": {"min": 0.001, "max": 0.1}},
            project="test-project",
            entity="test-entity",
            prior_runs=["run1"],
            early_terminate={"type": "hyperband"},
        )

        assert isinstance(config, _WandBSweepConfig)
        assert config.method == "random"
        assert config.metric == {"name": "loss", "goal": "minimize"}
        assert config.parameters == {"lr": {"min": 0.001, "max": 0.1}}
        assert config.project == "test-project"
        assert config.entity == "test-entity"
        assert config.prior_runs == ["run1"]
        assert config.kwargs == {"early_terminate": {"type": "hyperband"}}

    def test_wandb_sweep_config_download_logs_serialization_true(self):
        """Test that download_logs=True serializes correctly as JSON boolean for sweep config."""
        config = _WandBSweepConfig(
            method="random",
            download_logs=True,
        )

        result = config.to_dict()

        # Should be JSON "true" (lowercase), not Python "True"
        assert result["wandb_sweep_download_logs"] == "true"
        assert result["wandb_sweep_download_logs"] == json.dumps(True)

    def test_wandb_sweep_config_download_logs_serialization_false(self):
        """Test that download_logs=False serializes correctly as JSON boolean for sweep config."""
        config = _WandBSweepConfig(
            method="random",
            download_logs=False,
        )

        result = config.to_dict()

        # Should be JSON "false" (lowercase), not Python "False"
        assert result["wandb_sweep_download_logs"] == "false"
        assert result["wandb_sweep_download_logs"] == json.dumps(False)

    def test_wandb_sweep_config_download_logs_deserialization_true(self):
        """Test that download_logs deserializes correctly from JSON true for sweep config."""
        input_dict = {
            "wandb_sweep_method": "random",
            "wandb_sweep_download_logs": "true",  # JSON format
        }

        config = _WandBSweepConfig.from_dict(input_dict)

        assert config.method == "random"
        assert config.download_logs is True
        assert isinstance(config.download_logs, bool)

    def test_wandb_sweep_config_download_logs_deserialization_false(self):
        """Test that download_logs deserializes correctly from JSON false for sweep config."""
        input_dict = {
            "wandb_sweep_method": "random",
            "wandb_sweep_download_logs": "false",  # JSON format
        }

        config = _WandBSweepConfig.from_dict(input_dict)

        assert config.method == "random"
        assert config.download_logs is False
        assert isinstance(config.download_logs, bool)

    def test_wandb_sweep_config_download_logs_roundtrip(self):
        """Test that download_logs survives a roundtrip through serialization for sweep config."""
        # Test with True
        config_true = _WandBSweepConfig(method="random", download_logs=True)
        dict_true = config_true.to_dict()
        restored_true = _WandBSweepConfig.from_dict(dict_true)
        assert restored_true.download_logs is True
        assert isinstance(restored_true.download_logs, bool)

        # Test with False
        config_false = _WandBSweepConfig(method="random", download_logs=False)
        dict_false = config_false.to_dict()
        restored_false = _WandBSweepConfig.from_dict(dict_false)
        assert restored_false.download_logs is False
        assert isinstance(restored_false.download_logs, bool)


class TestGetWandBContext:
    """Tests for get_wandb_context() function."""

    @patch("flyte.ctx")
    def test_get_wandb_context_with_config(self, mock_ctx):
        """Test getting wandb context when config exists."""
        mock_context = MagicMock()
        mock_context.custom_context = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
        }
        mock_ctx.return_value = mock_context

        config = get_wandb_context()

        assert config is not None
        assert config.project == "test-project"
        assert config.entity == "test-entity"

    @patch("flyte.ctx")
    def test_get_wandb_context_no_wandb_keys(self, mock_ctx):
        """Test getting wandb context when no wandb keys exist."""
        mock_context = MagicMock()
        mock_context.custom_context = {"other_key": "value"}
        mock_ctx.return_value = mock_context

        config = get_wandb_context()

        assert config is None

    @patch("flyte.ctx")
    def test_get_wandb_context_no_ctx(self, mock_ctx):
        """Test getting wandb context when flyte ctx is None."""
        mock_ctx.return_value = None

        config = get_wandb_context()

        assert config is None

    @patch("flyte.ctx")
    def test_get_wandb_context_no_custom_context(self, mock_ctx):
        """Test getting wandb context when custom_context is None."""
        mock_context = MagicMock()
        mock_context.custom_context = None
        mock_ctx.return_value = mock_context

        config = get_wandb_context()

        assert config is None


class TestGetWandBSweepContext:
    """Tests for get_wandb_sweep_context() function."""

    @patch("flyte.ctx")
    def test_get_wandb_sweep_context_with_config(self, mock_ctx):
        """Test getting wandb sweep context when config exists."""
        mock_context = MagicMock()
        mock_context.custom_context = {
            "wandb_sweep_method": "random",
            "wandb_sweep_metric": json.dumps({"name": "loss"}),
        }
        mock_ctx.return_value = mock_context

        config = get_wandb_sweep_context()

        assert config is not None
        assert config.method == "random"
        assert config.metric == {"name": "loss"}

    @patch("flyte.ctx")
    def test_get_wandb_sweep_context_no_sweep_keys(self, mock_ctx):
        """Test getting sweep context when no sweep keys exist."""
        mock_context = MagicMock()
        mock_context.custom_context = {
            "wandb_project": "test-project",  # Regular wandb key, not sweep
        }
        mock_ctx.return_value = mock_context

        config = get_wandb_sweep_context()

        assert config is None

    @patch("flyte.ctx")
    def test_get_wandb_sweep_context_no_ctx(self, mock_ctx):
        """Test getting sweep context when flyte ctx is None."""
        mock_ctx.return_value = None

        config = get_wandb_sweep_context()

        assert config is None


class TestContextManagerIntegration:
    """Integration tests for context manager behavior."""

    @patch("flyte.ctx")
    @patch("flyte.custom_context")
    def test_wandb_config_as_context_manager(self, mock_custom_context, mock_ctx):
        """Test using wandb_config as a context manager."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_cm = MagicMock()
        mock_custom_context.return_value = mock_cm

        config = wandb_config(project="test-project", entity="test-entity")

        with config:
            pass  # Context manager should work

        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()

    @patch("flyte.ctx")
    @patch("flyte.custom_context")
    def test_wandb_sweep_config_as_context_manager(self, mock_custom_context, mock_ctx):
        """Test using wandb_sweep_config as a context manager."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_cm = MagicMock()
        mock_custom_context.return_value = mock_cm

        config = wandb_sweep_config(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
        )

        with config:
            pass  # Context manager should work

        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_wandb_config_with_non_serializable_value(self):
        """Test that non-JSON-serializable values in config raise error."""
        config = _WandBConfig(project="test", config={"func": lambda x: x})

        with pytest.raises(ValueError, match="must be JSON-serializable"):
            config.to_dict()

    def test_wandb_config_unpacking(self):
        """Test that config can be unpacked with ** operator."""
        config = wandb_config(project="test-project", entity="test-entity")

        # Should be able to unpack as dict
        unpacked = {**config}

        assert "wandb_project" in unpacked
        assert "wandb_entity" in unpacked

    def test_wandb_sweep_config_unpacking(self):
        """Test that sweep config can be unpacked with ** operator."""
        config = wandb_sweep_config(method="random")

        # Should be able to unpack as dict
        unpacked = {**config}

        assert "wandb_sweep_method" in unpacked
