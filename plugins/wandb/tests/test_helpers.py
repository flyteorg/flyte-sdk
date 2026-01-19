"""Tests for wandb helper functions."""

from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.wandb import (
    download_wandb_run_dir,
    download_wandb_sweep_dirs,
    get_wandb_run,
    get_wandb_run_dir,
    get_wandb_sweep_id,
)


class TestGetWandbRun:
    """Tests for get_wandb_run() helper function."""

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_run_returns_run_when_available(self, mock_ctx):
        """Test that get_wandb_run returns the run when it exists."""
        mock_run = MagicMock()
        mock_run.id = "test-run-id"

        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": mock_run}
        mock_ctx.return_value = mock_context

        result = get_wandb_run()

        assert result == mock_run

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_run_returns_none_when_no_run(self, mock_ctx):
        """Test that get_wandb_run returns None when no run exists."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_ctx.return_value = mock_context

        result = get_wandb_run()

        assert result is None

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_run_returns_none_when_no_data(self, mock_ctx):
        """Test that get_wandb_run returns None when data is None."""
        mock_context = MagicMock()
        mock_context.data = None
        mock_ctx.return_value = mock_context

        result = get_wandb_run()

        assert result is None

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_run_returns_none_when_no_context(self, mock_ctx):
        """Test that get_wandb_run returns None when context is None."""
        mock_ctx.return_value = None

        result = get_wandb_run()

        assert result is None


class TestGetWandbSweepId:
    """Tests for get_wandb_sweep_id() helper function."""

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_sweep_id_returns_id_when_available(self, mock_ctx):
        """Test that get_wandb_sweep_id returns the ID when it exists."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_sweep_id": "sweep-123"}
        mock_ctx.return_value = mock_context

        result = get_wandb_sweep_id()

        assert result == "sweep-123"

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_sweep_id_returns_none_when_no_id(self, mock_ctx):
        """Test that get_wandb_sweep_id returns None when no ID exists."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        result = get_wandb_sweep_id()

        assert result is None

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_sweep_id_returns_none_when_no_custom_context(self, mock_ctx):
        """Test that get_wandb_sweep_id returns None when custom_context is None."""
        mock_context = MagicMock()
        mock_context.custom_context = None
        mock_ctx.return_value = mock_context

        result = get_wandb_sweep_id()

        assert result is None

    @patch("flyteplugins.wandb.flyte.ctx")
    def test_get_wandb_sweep_id_returns_none_when_no_context(self, mock_ctx):
        """Test that get_wandb_sweep_id returns None when context is None."""
        mock_ctx.return_value = None

        result = get_wandb_sweep_id()

        assert result is None


class TestGetWandbRunDir:
    """Tests for get_wandb_run_dir() helper function."""

    @patch("flyteplugins.wandb.get_wandb_run")
    def test_get_wandb_run_dir_returns_dir_when_run_exists(self, mock_get_run):
        """Test that get_wandb_run_dir returns run.dir when run exists."""
        mock_run = MagicMock()
        mock_run.dir = "/tmp/wandb/run-123"
        mock_get_run.return_value = mock_run

        result = get_wandb_run_dir()

        assert result == "/tmp/wandb/run-123"

    @patch("flyteplugins.wandb.get_wandb_run")
    def test_get_wandb_run_dir_returns_none_when_no_run(self, mock_get_run):
        """Test that get_wandb_run_dir returns None when no run exists."""
        mock_get_run.return_value = None

        result = get_wandb_run_dir()

        assert result is None


class TestDownloadWandbRunDir:
    """Tests for download_wandb_run_dir() helper function."""

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_run")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_with_explicit_run_id(
        self, mock_ctx, mock_get_run, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir with explicit run_id."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_file = MagicMock()
        mock_api_run.files.return_value = [mock_file]
        mock_api_run.summary = {}
        mock_api_run.history.return_value = []
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        result = download_wandb_run_dir(run_id="explicit-run-id")

        assert result == "/tmp/wandb_runs/explicit-run-id"
        mock_api.run.assert_called_once_with("test-entity/test-project/explicit-run-id")
        mock_file.download.assert_called_once()

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_run")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_uses_context_run_id(
        self, mock_ctx, mock_get_run, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir uses run_id from context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_run_id": "context-run-id"}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.files.return_value = []
        mock_api_run.summary = {}
        mock_api_run.history.return_value = []
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        result = download_wandb_run_dir()

        assert result == "/tmp/wandb_runs/context-run-id"
        mock_api.run.assert_called_once_with("test-entity/test-project/context-run-id")

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_run")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_uses_current_run(
        self, mock_ctx, mock_get_run, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir uses current run when no context run_id."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_run = MagicMock()
        mock_run.id = "current-run-id"
        mock_get_run.return_value = mock_run

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.files.return_value = []
        mock_api_run.summary = {}
        mock_api_run.history.return_value = []
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        result = download_wandb_run_dir()

        assert result == "/tmp/wandb_runs/current-run-id"

    @patch("flyteplugins.wandb.get_wandb_run")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_raises_error_when_no_run_id(self, mock_ctx, mock_get_run):
        """Test that download_wandb_run_dir raises error when no run_id available."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context
        mock_get_run.return_value = None

        with pytest.raises(RuntimeError, match="No run_id provided"):
            download_wandb_run_dir()

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_with_custom_path(
        self, mock_ctx, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir to custom path."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_run_id": "test-run-id"}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_file = MagicMock()
        mock_api_run.files.return_value = [mock_file]
        mock_api_run.summary = {}
        mock_api_run.history.return_value = []
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        result = download_wandb_run_dir(path="/custom/path")

        assert result == "/custom/path"
        mock_file.download.assert_called_once_with(root="/custom/path", replace=True)

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_exports_history(
        self, mock_ctx, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir exports metrics history to JSON."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_run_id": "test-run-id"}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.files.return_value = []
        mock_api_run.summary = {}
        mock_api_run.history.return_value = [{"step": 0, "loss": 0.5}, {"step": 1, "loss": 0.3}]
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        mock_path_join.return_value = "/tmp/wandb_runs/test-run-id/metrics_history.json"

        result = download_wandb_run_dir()

        assert result == "/tmp/wandb_runs/test-run-id"
        # Verify open was called for history export
        mock_open.assert_called()

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_exports_summary(
        self, mock_ctx, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir exports summary to JSON."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_run_id": "test-run-id"}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.files.return_value = []
        mock_api_run.summary = {"loss": 0.5, "accuracy": 0.95}
        mock_history = MagicMock()
        mock_history.empty = True
        mock_api_run.history.return_value = mock_history
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        mock_path_join.return_value = "/tmp/wandb_runs/test-run-id/summary.json"

        result = download_wandb_run_dir()

        assert result == "/tmp/wandb_runs/test-run-id"
        # Verify open was called for summary.json
        mock_open.assert_called()

    @patch("builtins.open", create=True)
    @patch("os.path.join")
    @patch("os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.flyte.ctx")
    def test_download_wandb_run_dir_skip_history(
        self, mock_ctx, mock_get_context, mock_api_class, mock_makedirs, mock_path_join, mock_open
    ):
        """Test downloading run dir can skip history export."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_run_id": "test-run-id"}
        mock_ctx.return_value = mock_context

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.files.return_value = []
        mock_api_run.summary = {}  # Empty summary
        mock_api.run.return_value = mock_api_run
        mock_api_class.return_value = mock_api

        result = download_wandb_run_dir(include_history=False)

        assert result == "/tmp/wandb_runs/test-run-id"
        mock_api_run.history.assert_not_called()


class TestDownloadWandbSweepDirs:
    """Tests for download_wandb_sweep_dirs() helper function."""

    @patch("flyteplugins.wandb.download_wandb_run_dir")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_sweep_id")
    def test_download_wandb_sweep_dirs_with_explicit_sweep_id(
        self, mock_get_sweep_id, mock_get_context, mock_api_class, mock_download_run_dir
    ):
        """Test downloading sweep dirs with explicit sweep_id."""
        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_sweep = MagicMock()
        mock_run1 = MagicMock()
        mock_run1.id = "run-1"
        mock_run2 = MagicMock()
        mock_run2.id = "run-2"
        mock_sweep.runs = [mock_run1, mock_run2]
        mock_api.sweep.return_value = mock_sweep
        mock_api_class.return_value = mock_api

        mock_download_run_dir.side_effect = lambda run_id, path, include_history: path

        result = download_wandb_sweep_dirs(sweep_id="test-sweep-id")

        assert len(result) == 2
        assert "/tmp/wandb_runs/run-1" in result
        assert "/tmp/wandb_runs/run-2" in result
        mock_api.sweep.assert_called_once_with("test-entity/test-project/test-sweep-id")

    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_sweep_id")
    def test_download_wandb_sweep_dirs_raises_error_when_no_sweep_id(self, mock_get_sweep_id, mock_get_context):
        """Test that download_wandb_sweep_dirs raises error when no sweep_id."""
        mock_get_sweep_id.return_value = None

        with pytest.raises(RuntimeError, match="No sweep_id provided"):
            download_wandb_sweep_dirs()

    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_sweep_id")
    def test_download_wandb_sweep_dirs_raises_error_when_no_entity_project(self, mock_get_sweep_id, mock_get_context):
        """Test that download_wandb_sweep_dirs raises error when no entity/project."""
        mock_get_sweep_id.return_value = "test-sweep-id"
        mock_get_context.return_value = None

        with pytest.raises(RuntimeError, match="Cannot query sweep without entity and project"):
            download_wandb_sweep_dirs()

    @patch("flyteplugins.wandb.download_wandb_run_dir")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_sweep_id")
    def test_download_wandb_sweep_dirs_uses_context_sweep_id(
        self, mock_get_sweep_id, mock_get_context, mock_api_class, mock_download_run_dir
    ):
        """Test downloading sweep dirs uses sweep_id from context."""
        mock_get_sweep_id.return_value = "context-sweep-id"

        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_sweep = MagicMock()
        mock_sweep.runs = []
        mock_api.sweep.return_value = mock_sweep
        mock_api_class.return_value = mock_api

        result = download_wandb_sweep_dirs()

        assert result == []
        mock_api.sweep.assert_called_once_with("test-entity/test-project/context-sweep-id")

    @patch("flyteplugins.wandb.download_wandb_run_dir")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.get_wandb_sweep_id")
    def test_download_wandb_sweep_dirs_with_custom_base_path(
        self, mock_get_sweep_id, mock_get_context, mock_api_class, mock_download_run_dir
    ):
        """Test downloading sweep dirs to custom base path."""
        mock_wandb_ctx = MagicMock()
        mock_wandb_ctx.entity = "test-entity"
        mock_wandb_ctx.project = "test-project"
        mock_get_context.return_value = mock_wandb_ctx

        mock_api = MagicMock()
        mock_sweep = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "run-1"
        mock_sweep.runs = [mock_run]
        mock_api.sweep.return_value = mock_sweep
        mock_api_class.return_value = mock_api

        mock_download_run_dir.side_effect = lambda run_id, path, include_history: path

        result = download_wandb_sweep_dirs(sweep_id="test-sweep", base_path="/custom/base")

        assert result == ["/custom/base/run-1"]
        mock_download_run_dir.assert_called_once_with(run_id="run-1", path="/custom/base/run-1", include_history=True)
