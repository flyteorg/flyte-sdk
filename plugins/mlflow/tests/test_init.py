from unittest.mock import MagicMock, patch

from flyteplugins.mlflow import get_mlflow_run


class TestGetMlflowRun:
    @patch("flyteplugins.mlflow.flyte")
    def test_returns_run_from_data(self, mock_flyte):
        mock_run = MagicMock()
        ctx = MagicMock()
        ctx.data = {"_mlflow_run": mock_run}
        mock_flyte.ctx.return_value = ctx
        assert get_mlflow_run() is mock_run

    @patch("flyteplugins.mlflow.flyte")
    def test_returns_none_when_no_run(self, mock_flyte):
        ctx = MagicMock()
        ctx.data = {}
        mock_flyte.ctx.return_value = ctx
        assert get_mlflow_run() is None

    @patch("flyteplugins.mlflow.flyte")
    def test_returns_none_when_no_context(self, mock_flyte):
        mock_flyte.ctx.return_value = None
        assert get_mlflow_run() is None

    @patch("flyteplugins.mlflow.flyte")
    def test_returns_none_when_no_data(self, mock_flyte):
        ctx = MagicMock()
        ctx.data = None
        mock_flyte.ctx.return_value = ctx
        assert get_mlflow_run() is None
