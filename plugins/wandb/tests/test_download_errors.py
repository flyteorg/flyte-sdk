"""Tests for error handling in wandb download functions."""

from unittest.mock import MagicMock, patch

import pytest
import wandb.errors

from flyteplugins.wandb import (
    download_wandb_run_dir,
    download_wandb_sweep_dirs,
)


class TestDownloadWandbRunDirErrors:
    """Tests for error handling in download_wandb_run_dir()."""

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_run_not_found_raises_error(
        self, mock_get_context, mock_api_class
    ):
        """Test that downloading a non-existent run raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to raise CommError (run not found)
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.run.side_effect = wandb.errors.CommError("Run not found")

        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="Failed to fetch wandb run"):
            download_wandb_run_dir(run_id="non-existent-run")

    @patch("flyteplugins.wandb.os.makedirs")
    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_run_auth_failure_raises_error(
        self, mock_get_context, mock_api_class, mock_makedirs
    ):
        """Test that authentication failure raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to raise AuthenticationError
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.run.side_effect = wandb.errors.AuthenticationError("Invalid API key")

        # Should raise RuntimeError with auth-specific message
        with pytest.raises(RuntimeError, match="Authentication failed"):
            download_wandb_run_dir(run_id="test-run")

    @patch("flyteplugins.wandb.os.makedirs")
    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_run_directory_creation_failure_raises_error(
        self, mock_get_context, mock_makedirs
    ):
        """Test that directory creation failure raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock makedirs to raise OSError
        mock_makedirs.side_effect = OSError("Permission denied")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to create download directory"):
            download_wandb_run_dir(run_id="test-run")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.os.makedirs")
    def test_download_run_file_download_failure_raises_error(
        self, mock_makedirs, mock_get_context, mock_api_class
    ):
        """Test that file download failure raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to return a run with files
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_run = MagicMock()
        mock_api.run.return_value = mock_run

        # Mock file that raises error on download
        mock_file = MagicMock()
        mock_file.download.side_effect = Exception("Network timeout")
        mock_run.files.return_value = [mock_file]

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to download files"):
            download_wandb_run_dir(run_id="test-run")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.os.makedirs")
    @patch("builtins.open", side_effect=IOError("Disk full"))
    def test_download_run_summary_write_failure_raises_error(
        self, mock_open, mock_makedirs, mock_get_context, mock_api_class
    ):
        """Test that summary.json write failure raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to return a run
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_run = MagicMock()
        mock_api.run.return_value = mock_run
        mock_run.files.return_value = []
        mock_run.summary = {"loss": 0.5}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match=r"Failed to write summary\.json"):
            download_wandb_run_dir(run_id="test-run")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.os.makedirs")
    def test_download_run_unexpected_api_error_raises_error(
        self, mock_makedirs, mock_get_context, mock_api_class
    ):
        """Test that unexpected API errors raise RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to raise unexpected error
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.run.side_effect = ValueError("Unexpected error")

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Unexpected error fetching wandb run"):
            download_wandb_run_dir(run_id="test-run")


class TestDownloadWandbSweepDirsErrors:
    """Tests for error handling in download_wandb_sweep_dirs()."""

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_sweep_not_found_raises_error(
        self, mock_get_context, mock_api_class
    ):
        """Test that downloading a non-existent sweep raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to raise CommError (sweep not found)
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.sweep.side_effect = wandb.errors.CommError("Sweep not found")

        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="Failed to fetch wandb sweep"):
            download_wandb_sweep_dirs(sweep_id="non-existent-sweep")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_sweep_auth_failure_raises_error(
        self, mock_get_context, mock_api_class
    ):
        """Test that authentication failure raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to raise AuthenticationError
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.sweep.side_effect = wandb.errors.AuthenticationError("Invalid API key")

        # Should raise RuntimeError with auth-specific message
        with pytest.raises(RuntimeError, match="Authentication failed"):
            download_wandb_sweep_dirs(sweep_id="test-sweep")

    @patch("flyteplugins.wandb.get_wandb_context")
    def test_download_sweep_missing_entity_raises_error(self, mock_get_context):
        """Test that missing entity/project raises RuntimeError."""
        # Setup context without entity/project
        mock_context = MagicMock()
        mock_context.entity = None
        mock_context.project = None
        mock_get_context.return_value = mock_context

        # Should raise RuntimeError
        with pytest.raises(
            RuntimeError, match="Cannot query sweep without entity and project"
        ):
            download_wandb_sweep_dirs(sweep_id="test-sweep")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.download_wandb_run_dir")
    def test_download_sweep_all_runs_fail_raises_error(
        self, mock_download_run, mock_get_context, mock_api_class
    ):
        """Test that if all runs fail to download, raises RuntimeError."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to return a sweep with 2 runs
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_sweep = MagicMock()
        mock_run1 = MagicMock()
        mock_run1.id = "run-1"
        mock_run2 = MagicMock()
        mock_run2.id = "run-2"
        mock_sweep.runs = [mock_run1, mock_run2]
        mock_api.sweep.return_value = mock_sweep

        # Mock download_wandb_run_dir to fail for all runs
        mock_download_run.side_effect = RuntimeError("Download failed")

        # Should raise RuntimeError indicating all runs failed
        with pytest.raises(RuntimeError, match="Failed to download all 2 runs"):
            download_wandb_sweep_dirs(sweep_id="test-sweep")

    @patch("flyteplugins.wandb.wandb.Api")
    @patch("flyteplugins.wandb.get_wandb_context")
    @patch("flyteplugins.wandb.download_wandb_run_dir")
    def test_download_sweep_partial_failure_succeeds_with_warning(
        self, mock_download_run, mock_get_context, mock_api_class, caplog
    ):
        """Test that if some runs fail, function succeeds with warning."""
        # Setup context
        mock_context = MagicMock()
        mock_context.entity = "test-entity"
        mock_context.project = "test-project"
        mock_get_context.return_value = mock_context

        # Mock API to return a sweep with 3 runs
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_sweep = MagicMock()
        mock_run1 = MagicMock()
        mock_run1.id = "run-1"
        mock_run2 = MagicMock()
        mock_run2.id = "run-2"
        mock_run3 = MagicMock()
        mock_run3.id = "run-3"
        mock_sweep.runs = [mock_run1, mock_run2, mock_run3]
        mock_api.sweep.return_value = mock_sweep

        # Mock download_wandb_run_dir to fail for run-2 only
        def download_side_effect(run_id, **kwargs):
            if run_id == "run-2":
                raise RuntimeError("Download failed for run-2")
            return f"/tmp/wandb_runs/{run_id}"

        mock_download_run.side_effect = download_side_effect

        # Should succeed and return paths for successful runs
        with caplog.at_level("WARNING"):
            paths = download_wandb_sweep_dirs(sweep_id="test-sweep")

            # Should return 2 paths (for run-1 and run-3)
            assert len(paths) == 2
            assert "/tmp/wandb_runs/run-1" in paths
            assert "/tmp/wandb_runs/run-3" in paths

            # Should log warning about failed run
            assert len(caplog.records) == 1
            assert "Failed to download 1/3 runs" in caplog.text
            assert "run-2" in caplog.text


class TestTracedDownloadFunctionsErrors:
    """Tests for error handling in traced download functions."""

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_dir")
    async def test_download_run_logs_propagates_errors(self, mock_download_run):
        """Test that download_wandb_run_logs propagates errors from download_wandb_run_dir."""
        from flyteplugins.wandb import download_wandb_run_logs

        # Mock download_wandb_run_dir to raise RuntimeError
        mock_download_run.side_effect = RuntimeError("Run not found")

        # Should propagate the RuntimeError with original message
        with pytest.raises(RuntimeError, match="Run not found"):
            await download_wandb_run_logs("test-run")

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_dirs")
    async def test_download_sweep_logs_propagates_errors(self, mock_download_sweep):
        """Test that download_wandb_sweep_logs propagates errors from download_wandb_sweep_dirs."""
        from flyteplugins.wandb import download_wandb_sweep_logs

        # Mock download_wandb_sweep_dirs to raise RuntimeError
        mock_download_sweep.side_effect = RuntimeError("Sweep not found")

        # Should propagate the RuntimeError with original message
        with pytest.raises(RuntimeError, match="Sweep not found"):
            await download_wandb_sweep_logs("test-sweep")
