"""Tests for download_logs functionality in wandb decorators."""

from unittest.mock import MagicMock, patch

import flyte
import pytest

from flyteplugins.wandb.decorator import wandb_init, wandb_sweep


class TestWandbInitDownloadLogsExecution:
    """Tests for download_logs parameter execution in @wandb_init decorator."""

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_true_calls_trace_function(self, mock_wandb_init, mock_download_logs):
        """Test that download_logs=True actually calls the download trace function."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-123"
        mock_run.dir = "/tmp/wandb/run-123"
        mock_wandb_init.return_value = mock_run

        @wandb_init(download_logs=True, project="test-project")
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute task
        await flyte.init.aio()
        result = await flyte.run.aio(test_task)

        # Verify download was called with correct run_id
        mock_download_logs.assert_called_once_with("test-run-123")
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_false_does_not_call_trace(self, mock_wandb_init, mock_download_logs):
        """Test that download_logs=False does NOT call the download trace function."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-456"
        mock_run.dir = "/tmp/wandb/run-456"
        mock_wandb_init.return_value = mock_run

        @wandb_init(download_logs=False, project="test-project")
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute task
        await flyte.init.aio()
        result = await flyte.run.aio(test_task)

        # Verify download was NOT called
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_none_defaults_to_false(self, mock_wandb_init, mock_download_logs):
        """Test that download_logs=None defaults to False (no download)."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-789"
        mock_run.dir = "/tmp/wandb/run-789"
        mock_wandb_init.return_value = mock_run

        @wandb_init(project="test-project")  # No download_logs parameter
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute task
        await flyte.init.aio()
        result = await flyte.run.aio(test_task)

        # Verify download was NOT called (default is False)
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_context_fallback(self, mock_wandb_init, mock_download_logs):
        """Test that decorator falls back to context config when download_logs=None."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-ctx"
        mock_run.dir = "/tmp/wandb/run-ctx"
        mock_wandb_init.return_value = mock_run

        @wandb_init  # No download_logs - should use context
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with context containing download_logs=True
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_config

        result = await flyte.with_runcontext(
            custom_context=wandb_config(project="test-project", download_logs=True)
        ).run.aio(test_task)

        # Verify download WAS called because context has download_logs=True
        mock_download_logs.assert_called_once_with("test-run-ctx")
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_decorator_overrides_context(self, mock_wandb_init, mock_download_logs):
        """Test that decorator parameter overrides context config."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-override"
        mock_run.dir = "/tmp/wandb/run-override"
        mock_wandb_init.return_value = mock_run

        @wandb_init(download_logs=False)  # Explicit False in decorator
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with context containing download_logs=True
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_config

        result = await flyte.with_runcontext(
            custom_context=wandb_config(
                project="test-project",
                download_logs=True,  # Context says True
            )
        ).run.aio(test_task)

        # Verify download was NOT called (decorator False overrides context True)
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_sync_task_true_calls_trace(self, mock_wandb_init, mock_download_logs):
        """Test that download_logs=True works with sync tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-sync"
        mock_run.dir = "/tmp/wandb/run-sync"
        mock_wandb_init.return_value = mock_run

        @wandb_init(download_logs=True, project="test-project")
        @env.task
        def sync_task() -> str:
            return "result"

        # Execute task
        await flyte.init.aio()
        result = await flyte.run.aio(sync_task)

        # Verify download was called even for sync task
        mock_download_logs.assert_called_once_with("test-run-sync")
        assert result.outputs()[0] == "result"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_download_logs_sync_task_false_no_trace(self, mock_wandb_init, mock_download_logs):
        """Test that download_logs=False works with sync tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb run
        mock_run = MagicMock()
        mock_run.id = "test-run-sync-ok"
        mock_run.dir = "/tmp/wandb/run-sync-ok"
        mock_wandb_init.return_value = mock_run

        @wandb_init(download_logs=False, project="test-project")
        @env.task
        def sync_task() -> str:
            return "result"

        # Execute task
        await flyte.init.aio()
        result = await flyte.run.aio(sync_task)

        # Verify download was NOT called
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "result"


class TestWandbSweepDownloadLogsExecution:
    """Tests for download_logs parameter execution in @wandb_sweep decorator."""

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_true_calls_trace_function(self, mock_wandb_sweep, mock_download_logs):
        """Test that download_logs=True actually calls the download trace function."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-123"

        @wandb_sweep(download_logs=True, project="test-project")
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with sweep config
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(test_task)

        # Verify download was called with correct sweep_id
        mock_download_logs.assert_called_once_with("sweep-123")
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_false_does_not_call_trace(self, mock_wandb_sweep, mock_download_logs):
        """Test that download_logs=False does NOT call the download trace function."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-456"

        @wandb_sweep(download_logs=False, project="test-project")
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with sweep config
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(test_task)

        # Verify download was NOT called
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_none_defaults_to_false(self, mock_wandb_sweep, mock_download_logs):
        """Test that download_logs=None defaults to False (no download)."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-789"

        @wandb_sweep(project="test-project")  # No download_logs parameter
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with sweep config
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(test_task)

        # Verify download was NOT called (default is False)
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_context_fallback(self, mock_wandb_sweep, mock_download_logs):
        """Test that decorator falls back to sweep context config when download_logs=None."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-ctx"

        @wandb_sweep(project="test-project")  # No download_logs - should use context
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with sweep config containing download_logs=True
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
                download_logs=True,  # Context says True
            )
        ).run.aio(test_task)

        # Verify download WAS called because sweep context has download_logs=True
        mock_download_logs.assert_called_once_with("sweep-ctx")
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_decorator_overrides_context(self, mock_wandb_sweep, mock_download_logs):
        """Test that decorator parameter overrides context config."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-override"

        @wandb_sweep(download_logs=False, project="test-project")  # Explicit False
        @env.task
        async def test_task() -> str:
            return "success"

        # Execute with sweep config containing download_logs=True
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
                download_logs=True,  # Context says True
            )
        ).run.aio(test_task)

        # Verify download was NOT called (decorator False overrides context True)
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "success"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_sync_task_true_calls_trace(self, mock_wandb_sweep, mock_download_logs):
        """Test that download_logs=True works with sync sweep tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-sync"

        @wandb_sweep(download_logs=True, project="test-project")
        @env.task
        def sync_task() -> str:
            return "result"

        # Execute with sweep config
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(sync_task)

        # Verify download was called even for sync task
        mock_download_logs.assert_called_once_with("sweep-sync")
        assert result.outputs()[0] == "result"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_download_logs_sync_task_false_no_trace(self, mock_wandb_sweep, mock_download_logs):
        """Test that download_logs=False works with sync sweep tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "sweep-sync-ok"

        @wandb_sweep(download_logs=False, project="test-project")
        @env.task
        def sync_task() -> str:
            return "result"

        # Execute with sweep config
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(sync_task)

        # Verify download was NOT called
        mock_download_logs.assert_not_called()
        assert result.outputs()[0] == "result"


class TestDownloadLogsEdgeCases:
    """Edge case tests for download_logs functionality."""

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_run_logs")
    @patch("wandb.init")
    async def test_multiple_tasks_independent_download_settings(self, mock_wandb_init, mock_download_logs):
        """Test that multiple tasks with different download_logs settings work independently."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock wandb runs
        mock_run_1 = MagicMock()
        mock_run_1.id = "run-with-download"
        mock_run_1.dir = "/tmp/wandb/run-1"

        mock_run_2 = MagicMock()
        mock_run_2.id = "run-without-download"
        mock_run_2.dir = "/tmp/wandb/run-2"

        mock_wandb_init.side_effect = [mock_run_1, mock_run_2]

        @wandb_init(download_logs=True, project="test-project", run_mode="new")
        @env.task
        async def task_with_download() -> str:
            return "task1"

        @wandb_init(download_logs=False, project="test-project", run_mode="new")
        @env.task
        async def task_without_download() -> str:
            return "task2"

        @env.task
        async def parent() -> str:
            result1 = await task_with_download()
            result2 = await task_without_download()
            return f"{result1}-{result2}"

        # Execute
        await flyte.init.aio()
        result = await flyte.run.aio(parent)

        # Verify download was called only once for task_with_download
        assert mock_download_logs.call_count == 1
        mock_download_logs.assert_called_with("run-with-download")
        assert result.outputs()[0] == "task1-task2"

    @pytest.mark.asyncio
    @patch("flyteplugins.wandb.download_wandb_sweep_logs")
    @patch("wandb.sweep")
    async def test_sweep_download_logs_with_empty_sweep(self, mock_wandb_sweep, mock_download_logs):
        """Test that sweep download works even with no trials."""
        env = flyte.TaskEnvironment(name="test-env")

        # Mock sweep creation
        mock_wandb_sweep.return_value = "empty-sweep"

        @wandb_sweep(download_logs=True, project="test-project")
        @env.task
        async def sweep_task() -> str:
            # Don't run any agents
            return "no trials"

        # Execute
        await flyte.init.aio()
        from flyteplugins.wandb import wandb_sweep_config

        result = await flyte.with_runcontext(
            custom_context=wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={"lr": {"min": 0.001, "max": 0.1}},
            )
        ).run.aio(sweep_task)

        # Download should still be called
        mock_download_logs.assert_called_once_with("empty-sweep")
        assert result.outputs()[0] == "no trials"
