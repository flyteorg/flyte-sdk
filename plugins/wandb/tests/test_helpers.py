"""Tests for wandb helper functions."""

from unittest.mock import MagicMock, patch

from flyteplugins.wandb import get_wandb_run, get_wandb_sweep_id


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
