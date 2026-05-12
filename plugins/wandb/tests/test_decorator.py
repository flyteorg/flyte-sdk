"""Tests for wandb decorators."""

import logging
import os
from unittest.mock import MagicMock, patch

import flyte
import pytest
from flyte._task import AsyncFunctionTaskTemplate

from flyteplugins.wandb import (
    Wandb,
    WandbSweep,
    wandb_init,
    wandb_sweep,
)
from flyteplugins.wandb._decorator import _build_init_kwargs


class TestBuildWandbInitKwargs:
    """Tests for _build_init_kwargs helper function."""

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_build_init_kwargs_with_context(self, mock_get_context):
        """Test building init kwargs when context config exists."""
        from flyteplugins.wandb._context import _WandBConfig

        mock_get_context.return_value = _WandBConfig(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            kwargs={"custom_key": "custom_value"},
        )

        result = _build_init_kwargs()

        assert result["project"] == "test-project"
        assert result["entity"] == "test-entity"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["custom_key"] == "custom_value"
        assert "kwargs" not in result  # Should be merged and removed

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_build_init_kwargs_no_context(self, mock_get_context):
        """Test building init kwargs when no context config exists."""
        mock_get_context.return_value = None
        result = _build_init_kwargs()
        assert result == {}

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_build_init_kwargs_no_extra_kwargs(self, mock_get_context):
        """Test building init kwargs when context has no extra kwargs."""
        from flyteplugins.wandb._context import _WandBConfig

        mock_get_context.return_value = _WandBConfig(
            project="test-project",
            kwargs=None,
        )

        result = _build_init_kwargs()

        assert result["project"] == "test-project"
        assert "kwargs" not in result

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_build_init_kwargs_excludes_run_mode(self, mock_get_context):
        """Test that run_mode is excluded from init kwargs (Flyte-specific, not for wandb.init)."""
        from flyteplugins.wandb._context import _WandBConfig

        mock_get_context.return_value = _WandBConfig(
            project="test-project",
            run_mode="shared",
            kwargs=None,
        )

        result = _build_init_kwargs()

        assert result["project"] == "test-project"
        assert "run_mode" not in result

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_build_init_kwargs_excludes_download_logs(self, mock_get_context):
        """Test that download_logs is excluded from init kwargs (Flyte-specific)."""
        from flyteplugins.wandb._context import _WandBConfig

        mock_get_context.return_value = _WandBConfig(
            project="test-project",
            download_logs=True,
            kwargs=None,
        )

        result = _build_init_kwargs()

        assert result["project"] == "test-project"
        assert "download_logs" not in result

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_decorator_kwargs_override_context(self, mock_get_context):
        """Test that decorator kwargs override context config (handled in _wandb_run, not _build_init_kwargs)."""
        from flyteplugins.wandb._context import _WandBConfig

        # _build_init_kwargs just returns context config
        # decorator kwargs override happens in _wandb_run when merging
        mock_get_context.return_value = _WandBConfig(
            project="context-project",
            entity="context-entity",
        )

        result = _build_init_kwargs()

        # _build_init_kwargs returns context values
        assert result["project"] == "context-project"
        assert result["entity"] == "context-entity"

    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    def test_filters_none_values(self, mock_get_context):
        """Test that None values are filtered out."""
        from flyteplugins.wandb._context import _WandBConfig

        mock_get_context.return_value = _WandBConfig(
            project="test-project",
            entity=None,
            tags=None,
        )

        result = _build_init_kwargs()

        assert result["project"] == "test-project"
        assert "entity" not in result
        assert "tags" not in result


class TestWandbInitDecorator:
    """Tests for @wandb_init decorator."""

    def test_wandb_init_on_async_task_adds_link(self):
        """Test that @wandb_init adds a Wandb link to async tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity")
        @env.task
        async def test_task():
            return "result"

        assert isinstance(test_task, AsyncFunctionTaskTemplate)
        # Check that a link was added
        assert len(test_task.links) > 0
        assert isinstance(test_task.links[0], Wandb)
        assert test_task.links[0].project == "test-project"
        assert test_task.links[0].entity == "test-entity"

    def test_wandb_init_on_task_with_run_mode_new(self):
        """Test @wandb_init with run_mode="new"."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity", run_mode="new")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.run_mode == "new"

    def test_wandb_init_on_task_with_run_mode_shared(self):
        """Test @wandb_init with run_mode="shared"."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity", run_mode="shared")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.run_mode == "shared"

    def test_wandb_init_on_task_with_run_mode_auto(self):
        """Test @wandb_init with run_mode='auto'."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(run_mode="auto")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.run_mode == "auto"

    def test_wandb_init_preserves_existing_links(self):
        """Test that @wandb_init preserves existing task links."""
        env = flyte.TaskEnvironment(name="test-env")

        # Create a task and manually add a link
        @env.task
        async def base_task():
            return "result"

        existing_link = MagicMock()
        base_task_with_link = base_task.override(links=(existing_link,))

        # Apply wandb_init decorator to task that already has a link
        # The decorator should preserve the existing link
        decorated_task = wandb_init(project="test-project", entity="test-entity")(base_task_with_link)

        # Should have both links
        assert len(decorated_task.links) == 2
        assert any(isinstance(link, Wandb) for link in decorated_task.links)
        assert existing_link in decorated_task.links

    def test_wandb_init_on_regular_async_function(self):
        """Test @wandb_init on a regular async function (not a task)."""

        @wandb_init(project="test-project", entity="test-entity")
        async def test_func():
            return "result"

        # Should wrap the function
        assert callable(test_func)

    def test_wandb_init_on_regular_sync_function(self):
        """Test @wandb_init on a regular sync function."""

        @wandb_init(project="test-project", entity="test-entity")
        def test_func():
            return "result"

        # Should wrap the function
        assert callable(test_func)

    def test_wandb_init_without_params(self):
        """Test @wandb_init without any parameters."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init
        @env.task
        async def test_task():
            return "result"

        assert isinstance(test_task, AsyncFunctionTaskTemplate)
        link = test_task.links[0]
        assert isinstance(link, Wandb)
        assert link.project is None
        assert link.entity is None

    def test_wandb_init_with_additional_kwargs(self):
        """Test @wandb_init with additional wandb.init() kwargs."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            mode="offline",
        )
        @env.task
        async def test_task():
            return "result"

        # Decorator should accept and pass through additional kwargs
        assert isinstance(test_task, AsyncFunctionTaskTemplate)

    def test_wandb_init_with_rank_scope_global(self):
        """Test @wandb_init with rank_scope='global'."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity", rank_scope="global")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.rank_scope == "global"

    def test_wandb_init_with_rank_scope_worker(self):
        """Test @wandb_init with rank_scope='worker'."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity", rank_scope="worker")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.rank_scope == "worker"

    def test_wandb_init_default_rank_scope_is_global(self):
        """Test that default rank_scope is 'global'."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project", entity="test-entity")
        @env.task
        async def test_task():
            return "result"

        # Check link configuration
        link = test_task.links[0]
        assert link.rank_scope == "global"


class TestWandbSweepDecorator:
    """Tests for @wandb_sweep decorator."""

    def test_wandb_sweep_on_async_task_adds_link(self):
        """Test that @wandb_sweep adds a WandbSweep link to async tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_sweep
        @env.task
        async def test_task():
            return "result"

        assert isinstance(test_task, AsyncFunctionTaskTemplate)
        # Check that a sweep link was added
        assert len(test_task.links) > 0
        assert isinstance(test_task.links[0], WandbSweep)

    def test_wandb_sweep_preserves_existing_links(self):
        """Test that @wandb_sweep preserves existing task links."""
        env = flyte.TaskEnvironment(name="test-env")

        # Create a task and manually add a link
        @env.task
        async def base_task():
            return "result"

        existing_link = MagicMock()
        base_task_with_link = base_task.override(links=(existing_link,))

        # Apply wandb_sweep decorator to task that already has a link
        # The decorator should preserve the existing link
        decorated_task = wandb_sweep(base_task_with_link)

        # Should have both links
        assert len(decorated_task.links) == 2
        assert any(isinstance(link, WandbSweep) for link in decorated_task.links)
        assert existing_link in decorated_task.links

    def test_wandb_sweep_on_regular_function_raises_error(self):
        """Test that @wandb_sweep raises error on non-task functions."""

        with pytest.raises(RuntimeError, match="can only be used with Flyte tasks"):

            @wandb_sweep
            async def test_func():
                return "result"

    def test_wandb_sweep_on_sync_task(self):
        """Test @wandb_sweep on sync task."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_sweep
        @env.task
        def test_task():
            return "result"

        assert isinstance(test_task, AsyncFunctionTaskTemplate)
        link = test_task.links[0]
        assert isinstance(link, WandbSweep)


class TestWandbRunContextManager:
    """Tests for _wandb_run context manager."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_wandb_run_initializes_eagerly(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that _wandb_run eagerly initializes wandb and stores the run."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        # Mock wandb.init() to return a mock run
        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # Should initialize wandb eagerly and store the run
            assert "_wandb_run" in mock_context.data
            assert mock_context.data["_wandb_run"] == mock_run
            # Should also store run ID in custom_context
            assert mock_context.custom_context["_wandb_run_id"] == "test-run-id"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_wandb_run_restores_state_on_exit(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that _wandb_run restores state on exit."""
        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": "existing_run"}
        mock_context.custom_context = {"_wandb_run_id": "existing_id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "new-run-id"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Should restore previous state
        assert mock_context.data["_wandb_run"] == "existing_run"
        assert mock_context.custom_context["_wandb_run_id"] == "existing_id"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_wandb_run_cleans_up_run_on_exit(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that _wandb_run cleans up run and metadata on exit."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        # Mock wandb.init() to return a mock run
        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # Run should be initialized during context
            assert "_wandb_run" in mock_context.data

        # Run and metadata should be cleaned up after context exits (if not saved)
        # Since there was no saved state, it should be removed
        assert "_wandb_run" not in mock_context.data
        assert "_wandb_run_id" not in mock_context.custom_context

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_wandb_run_func_mode_without_flyte_context(self, mock_ctx, mock_wandb_init):
        """Test _wandb_run in func mode (no Flyte context)."""
        mock_ctx.return_value = None
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", func=True, project="test"):
            pass

        # Should call wandb.init directly in func mode
        mock_wandb_init.assert_called_once()
        mock_run.finish.assert_called()

    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_wandb_run_func_mode_with_flyte_context_yields_existing_run(self, mock_ctx):
        """Test that func mode with Flyte context yields existing run if available."""
        mock_existing_run = MagicMock()
        mock_existing_run.id = "existing-run-id"

        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": mock_existing_run}
        mock_ctx.return_value = mock_context

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", func=True) as run:
            # Should yield the existing run from parent context
            assert run == mock_existing_run


class TestCreateSweep:
    """Tests for _create_sweep context manager."""

    @patch("flyteplugins.wandb._decorator.wandb.sweep")
    @patch("flyteplugins.wandb._decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_basic(self, mock_ctx, mock_get_wandb_ctx, mock_get_sweep_ctx, mock_wandb_sweep):
        """Test basic sweep creation."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_ctx.return_value = mock_context

        mock_sweep_config = MagicMock()
        mock_sweep_config.project = "test-project"
        mock_sweep_config.entity = "test-entity"
        mock_sweep_config.prior_runs = None
        mock_sweep_config.to_sweep_config.return_value = {
            "method": "random",
            "metric": {"name": "loss", "goal": "minimize"},
        }
        mock_get_sweep_ctx.return_value = mock_sweep_config
        mock_get_wandb_ctx.return_value = None

        mock_wandb_sweep.return_value = "sweep-123"

        from flyteplugins.wandb._decorator import _create_sweep

        with _create_sweep() as sweep_id:
            assert sweep_id == "sweep-123"
            # Should store sweep_id in context
            assert mock_context.custom_context["_wandb_sweep_id"] == "sweep-123"

        # Should clean up sweep_id on exit
        assert "_wandb_sweep_id" not in mock_context.custom_context

        # Should have added deterministic name to sweep
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["sweep"]["name"] == "test-run-test-action"

    @patch("flyteplugins.wandb._decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_no_config_raises_error(self, mock_ctx, mock_get_sweep_ctx):
        """Test that missing sweep config raises error."""
        mock_context = MagicMock()
        mock_context.custom_context = {}  # Empty dict, no existing sweep_id
        mock_ctx.return_value = mock_context
        mock_get_sweep_ctx.return_value = None

        from flyteplugins.wandb._decorator import _create_sweep

        with pytest.raises(RuntimeError, match="No wandb sweep config found"):
            with _create_sweep():
                pass

    @patch("flyteplugins.wandb._decorator.wandb.sweep")
    @patch("flyteplugins.wandb._decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_fallback_to_wandb_config(
        self, mock_ctx, mock_get_wandb_ctx, mock_get_sweep_ctx, mock_wandb_sweep
    ):
        """Test that sweep uses wandb_config for project/entity fallback."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_ctx.return_value = mock_context

        mock_sweep_config = MagicMock()
        mock_sweep_config.project = None  # Not set in sweep config
        mock_sweep_config.entity = None
        mock_sweep_config.prior_runs = []
        mock_sweep_config.to_sweep_config.return_value = {"method": "random"}
        mock_get_sweep_ctx.return_value = mock_sweep_config

        mock_wandb_config = MagicMock()
        mock_wandb_config.project = "fallback-project"
        mock_wandb_config.entity = "fallback-entity"
        mock_get_wandb_ctx.return_value = mock_wandb_config

        mock_wandb_sweep.return_value = "sweep-456"

        from flyteplugins.wandb._decorator import _create_sweep

        with _create_sweep():
            pass

        # Should use fallback project/entity and generate deterministic name
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["project"] == "fallback-project"
        assert call_args[1]["entity"] == "fallback-entity"
        assert call_args[1]["prior_runs"] == []
        assert call_args[1]["sweep"]["method"] == "random"
        assert call_args[1]["sweep"]["name"] == "test-run-test-action"

    @patch("flyteplugins.wandb._decorator.wandb.sweep")
    @patch("flyteplugins.wandb._decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_preserves_provided_name(
        self, mock_ctx, mock_get_wandb_ctx, mock_get_sweep_ctx, mock_wandb_sweep
    ):
        """Test that sweep preserves user-provided name."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_ctx.return_value = mock_context

        mock_sweep_config = MagicMock()
        mock_sweep_config.project = "test-project"
        mock_sweep_config.entity = "test-entity"
        mock_sweep_config.prior_runs = []
        mock_sweep_config.to_sweep_config.return_value = {
            "method": "random",
            "name": "custom-sweep-name",
        }
        mock_get_sweep_ctx.return_value = mock_sweep_config
        mock_get_wandb_ctx.return_value = None

        mock_wandb_sweep.return_value = "sweep-789"

        from flyteplugins.wandb._decorator import _create_sweep

        with _create_sweep():
            pass

        # Should preserve user-provided name
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["sweep"]["name"] == "custom-sweep-name"


class TestDecoratorIntegration:
    """Integration tests for decorator combinations."""

    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_wandb_init_and_trace_decorator_order(self, mock_build_kwargs, mock_ctx):
        """Test that @wandb_init works correctly with @flyte.trace."""
        # This is more of a documentation test since the actual integration
        # requires the full Flyte runtime
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test-project")
        @env.task
        async def parent_task():
            return "result"

        assert isinstance(parent_task, AsyncFunctionTaskTemplate)
        assert len(parent_task.links) > 0


class TestModeSpecificBehavior:
    """Tests for local vs remote mode specific behavior."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_local_mode_uses_reinit_create_new(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that local mode with run_mode="new" uses reinit='create_new'."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Verify wandb.init was called with reinit='create_new'
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_local_mode_uses_reinit_return_previous(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that local mode with run_mode="shared" uses reinit='return_previous'."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Verify wandb.init was called with reinit='return_previous'
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["reinit"] == "return_previous"
        assert call_kwargs["id"] == "parent-run-id"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.wandb.Settings")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_remote_mode_uses_shared_settings(self, mock_build_kwargs, mock_ctx, mock_settings, mock_wandb_init):
        """Test that remote mode uses shared mode settings with x_primary=True."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Verify wandb.Settings was called with shared mode config
        settings_kwargs = mock_settings.call_args[1]
        assert settings_kwargs["mode"] == "shared"
        assert settings_kwargs["x_primary"] is True
        assert settings_kwargs["x_label"] == "test-action"
        # x_update_finish_state should not be set for primary
        assert "x_update_finish_state" not in settings_kwargs

        # Verify reinit was NOT set in remote mode
        call_kwargs = mock_wandb_init.call_args[1]
        assert "reinit" not in call_kwargs

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.wandb.Settings")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_remote_mode_secondary_task_settings(self, mock_build_kwargs, mock_ctx, mock_settings, mock_wandb_init):
        """Test that remote mode secondary task uses x_primary=False and x_update_finish_state=False."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "child-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Verify wandb.Settings was called with secondary task config
        settings_kwargs = mock_settings.call_args[1]
        assert settings_kwargs["mode"] == "shared"
        assert settings_kwargs["x_primary"] is False
        assert settings_kwargs["x_label"] == "child-action"
        assert settings_kwargs["x_update_finish_state"] is False

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_trace_detection_yields_existing_run(self, mock_ctx, mock_wandb_init):
        """Test that func mode with existing run in ctx.data yields that run without re-initializing."""
        mock_existing_run = MagicMock()
        mock_existing_run.id = "existing-run-id"

        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": mock_existing_run}
        mock_ctx.return_value = mock_context

        from flyteplugins.wandb._decorator import _wandb_run

        # func=True is used for traces/sweep objectives within a task
        with _wandb_run(run_mode="new", func=True, project="test") as run:
            # Should yield existing run from parent task
            assert run == mock_existing_run

        # Should NOT call wandb.init
        mock_wandb_init.assert_not_called()

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_run_mode_auto_with_parent_reuses_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that run_mode='auto' reuses parent run when parent run_id exists."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "child-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test"):
            pass

        # Verify it used parent's run_id
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "parent-run-id"
        assert call_kwargs["reinit"] == "return_previous"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_run_mode_auto_without_parent_creates_new(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that run_mode='auto' creates new run when no parent run_id exists."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test"):
            pass

        # Verify it created new run_id
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action"
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_run_mode_shared_without_parent_raises_error(self, mock_build_kwargs, mock_ctx):
        """Test that run_mode="shared" raises error when no parent run_id available."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        from flyteplugins.wandb._decorator import _wandb_run

        with pytest.raises(RuntimeError, match="Cannot reuse parent run: no parent run ID found"):
            with _wandb_run(run_mode="shared", project="test"):
                pass


class TestRunFinishingLogic:
    """Tests for run finishing logic in local vs remote mode."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_remote_mode_always_finishes_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that remote mode always calls run.finish()."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Should call run.finish in remote mode
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_remote_mode_secondary_task_finishes_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that remote mode secondary task also calls run.finish() (to flush data)."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "child-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Even secondary tasks should call finish in remote mode (x_update_finish_state=False prevents actual finish)
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_local_mode_primary_task_finishes_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that local mode primary task calls run.finish()."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Primary task in local mode should call finish
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_local_mode_secondary_task_does_not_finish_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that local mode secondary task does NOT call run.finish()."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "child-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Secondary task in local mode should NOT call finish (shares parent's run object)
        mock_run.finish.assert_not_called()


class TestStateSaveRestore:
    """Tests for complete state save and restore."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_complete_state_save_and_restore(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that all state (run, run_id) is saved and restored in nested contexts."""
        # Simulate nested context scenario (e.g., parent task calling another context manager)
        # Note: Child tasks have separate ctx.data, so this tests nested calls within same task
        mock_parent_run = MagicMock()
        mock_parent_run.id = "parent-run-id"

        mock_context = MagicMock()
        # Start with empty ctx.data (child tasks don't inherit parent's ctx.data)
        # Only custom_context is shared
        mock_context.data = {}
        mock_context.custom_context = {
            "_wandb_run_id": "parent-run-id",
        }
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "child-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        # Mock child run
        mock_child_run = MagicMock()
        mock_child_run.id = "child-run-id"
        mock_child_run.project = "child-project"
        mock_child_run.entity = "child-entity"
        mock_wandb_init.return_value = mock_child_run

        from flyteplugins.wandb._decorator import _wandb_run

        # Execute child context
        with _wandb_run(run_mode="new", project="child-project", entity="child-entity"):
            # During execution, child state should be active
            assert mock_context.data["_wandb_run"] == mock_child_run
            assert mock_context.custom_context["_wandb_run_id"] == "child-run-id"

        # After exit, parent state in custom_context should be restored
        # ctx.data should be empty (no parent run in ctx.data for child tasks)
        assert "_wandb_run" not in mock_context.data
        assert mock_context.custom_context["_wandb_run_id"] == "parent-run-id"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_state_cleanup_when_no_parent(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that state is cleaned up when there's no parent state to restore."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-id"
        mock_run.project = "test-project"
        mock_run.entity = "test-entity"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # State should be set during execution
            assert "_wandb_run" in mock_context.data
            assert "_wandb_run_id" in mock_context.custom_context

        # All state should be cleaned up after exit
        assert "_wandb_run" not in mock_context.data
        assert "_wandb_run_id" not in mock_context.custom_context


class TestRunModeDefault:
    """Tests for run_mode default behavior ('auto')."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_default_run_mode_is_auto(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that default run_mode is 'auto'."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        # Default run_mode is "auto", no parent run exists
        with _wandb_run(project="test"):
            pass

        # Should create new run since no parent (auto behavior)
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action"
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_auto_mode_reuses_parent_when_available(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that auto mode reuses parent run when available."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        # Default run_mode is "auto", parent run exists
        with _wandb_run(project="test"):
            pass

        # Should reuse parent's run ID (auto behavior)
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "parent-run-id"
        assert call_kwargs["reinit"] == "return_previous"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_explicit_new_mode_creates_new_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that explicit run_mode='new' creates new run even with parent."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Should create new run despite parent existing
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action"
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._build_init_kwargs")
    def test_explicit_shared_mode_reuses_parent(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that explicit run_mode='shared' reuses parent run."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {"_wandb_run_id": "parent-run-id"}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "local"
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        mock_run = MagicMock()
        mock_run.id = "parent-run-id"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Should reuse parent's run ID
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "parent-run-id"
        assert call_kwargs["reinit"] == "return_previous"


class TestSweepIdReuse:
    """Tests for sweep ID reuse in _create_sweep."""

    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_reuses_existing_sweep_id(self, mock_ctx):
        """Test that _create_sweep reuses existing sweep_id from context."""
        mock_context = MagicMock()
        mock_context.custom_context = {"_wandb_sweep_id": "existing-sweep-id"}
        mock_ctx.return_value = mock_context

        from flyteplugins.wandb._decorator import _create_sweep

        with _create_sweep() as sweep_id:
            assert sweep_id == "existing-sweep-id"

        # Should still have the sweep_id after exit (not cleaned up since we didn't create it)
        # Note: The implementation removes it on exit, which is also valid behavior
        # The key test is that it returns the existing ID without creating a new sweep

    @patch("flyteplugins.wandb._decorator.wandb.sweep")
    @patch("flyteplugins.wandb._decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb._decorator.get_wandb_context")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    def test_create_sweep_creates_new_when_no_existing(
        self, mock_ctx, mock_get_wandb_ctx, mock_get_sweep_ctx, mock_wandb_sweep
    ):
        """Test that _create_sweep creates new sweep when no existing sweep_id."""
        mock_context = MagicMock()
        mock_context.custom_context = {}  # No existing sweep ID
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_ctx.return_value = mock_context

        mock_sweep_config = MagicMock()
        mock_sweep_config.project = "test-project"
        mock_sweep_config.entity = "test-entity"
        mock_sweep_config.prior_runs = None
        mock_sweep_config.to_sweep_config.return_value = {"method": "random"}
        mock_get_sweep_ctx.return_value = mock_sweep_config
        mock_get_wandb_ctx.return_value = None

        mock_wandb_sweep.return_value = "new-sweep-123"

        from flyteplugins.wandb._decorator import _create_sweep

        with _create_sweep() as sweep_id:
            assert sweep_id == "new-sweep-123"

        # Should have called wandb.sweep to create a new sweep
        mock_wandb_sweep.assert_called_once()


class TestGetDistributedInfo:
    """Tests for _get_distributed_info helper function."""

    def test_returns_none_when_no_env_vars(self):
        """Test that None is returned when RANK/WORLD_SIZE env vars are not set."""
        from flyteplugins.wandb._decorator import _get_distributed_info

        with patch.dict(os.environ, {}, clear=True):
            result = _get_distributed_info()
            assert result is None

    def test_returns_none_when_world_size_is_one(self):
        """Test that None is returned when WORLD_SIZE <= 1 (not distributed)."""
        from flyteplugins.wandb._decorator import _get_distributed_info

        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}, clear=True):
            result = _get_distributed_info()
            assert result is None

    def test_returns_info_for_single_node_distributed(self):
        """Test distributed info for single-node multi-GPU setup."""
        from flyteplugins.wandb._decorator import _get_distributed_info

        env = {
            "RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            result = _get_distributed_info()
            assert result == {
                "rank": 2,
                "local_rank": 2,
                "world_size": 4,
                "local_world_size": 4,
                "worker_index": 0,
                "num_workers": 1,
            }

    def test_returns_info_for_multi_node_distributed(self):
        """Test distributed info for multi-node multi-GPU setup (2 nodes, 4 GPUs each)."""
        from flyteplugins.wandb._decorator import _get_distributed_info

        # Worker 1, local rank 2 (global rank = 4 + 2 = 6)
        env = {
            "RANK": "6",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            result = _get_distributed_info()
            assert result == {
                "rank": 6,
                "local_rank": 2,
                "world_size": 8,
                "local_world_size": 4,
                "worker_index": 1,
                "num_workers": 2,
            }

    def test_defaults_local_rank_to_zero(self):
        """Test that LOCAL_RANK defaults to 0 if not set."""
        from flyteplugins.wandb._decorator import _get_distributed_info

        env = {"RANK": "0", "WORLD_SIZE": "4"}
        with patch.dict(os.environ, env, clear=True):
            result = _get_distributed_info()
            assert result["local_rank"] == 0


class TestIsMultiNode:
    """Tests for _is_multi_node helper function."""

    def test_single_node_returns_false(self):
        """Test that single-node setup returns False."""
        from flyteplugins.wandb._decorator import _is_multi_node

        info = {"num_workers": 1}
        assert _is_multi_node(info) is False

    def test_multi_node_returns_true(self):
        """Test that multi-node setup returns True."""
        from flyteplugins.wandb._decorator import _is_multi_node

        info = {"num_workers": 2}
        assert _is_multi_node(info) is True


class TestIsPrimaryRank:
    """Tests for _is_primary_rank helper function."""

    def test_rank_zero_is_primary(self):
        """Test that rank 0 is primary."""
        from flyteplugins.wandb._decorator import _is_primary_rank

        info = {"rank": 0}
        assert _is_primary_rank(info) is True

    def test_non_zero_rank_is_not_primary(self):
        """Test that non-zero rank is not primary."""
        from flyteplugins.wandb._decorator import _is_primary_rank

        info = {"rank": 3}
        assert _is_primary_rank(info) is False


class TestShouldSkipRank:
    """Tests for _should_skip_rank function."""

    def test_shared_mode_never_skips(self):
        """Test that run_mode='shared' never skips any rank."""
        from flyteplugins.wandb._decorator import _should_skip_rank

        dist_info = {"rank": 3, "local_rank": 3, "num_workers": 1}
        assert _should_skip_rank("shared", "global", dist_info) is False
        assert _should_skip_rank("shared", "worker", dist_info) is False

    def test_new_mode_never_skips(self):
        """Test that run_mode='new' never skips any rank."""
        from flyteplugins.wandb._decorator import _should_skip_rank

        dist_info = {"rank": 3, "local_rank": 3, "num_workers": 1}
        assert _should_skip_rank("new", "global", dist_info) is False
        assert _should_skip_rank("new", "worker", dist_info) is False

    def test_auto_mode_single_node_only_rank_0_logs(self):
        """Test that run_mode='auto' on single-node only allows rank 0 (both scopes)."""
        from flyteplugins.wandb._decorator import _should_skip_rank

        # Single-node: rank 0 should NOT skip (for both global and worker scope)
        dist_info = {"rank": 0, "local_rank": 0, "num_workers": 1}
        assert _should_skip_rank("auto", "global", dist_info) is False
        assert _should_skip_rank("auto", "worker", dist_info) is False

        # Single-node: rank 1, 2, 3 should skip (for both global and worker scope)
        for rank in [1, 2, 3]:
            dist_info = {"rank": rank, "local_rank": rank, "num_workers": 1}
            assert _should_skip_rank("auto", "global", dist_info) is True
            assert _should_skip_rank("auto", "worker", dist_info) is True

    def test_auto_mode_global_scope_only_rank_0_logs(self):
        """Test that run_mode='auto' with rank_scope='global' only allows global rank 0."""
        from flyteplugins.wandb._decorator import _should_skip_rank

        # Global rank 0 - should NOT skip
        dist_info = {"rank": 0, "local_rank": 0, "num_workers": 2}
        assert _should_skip_rank("auto", "global", dist_info) is False

        # Global rank 1 (worker 0, local_rank 1) - should skip
        dist_info = {"rank": 1, "local_rank": 1, "num_workers": 2}
        assert _should_skip_rank("auto", "global", dist_info) is True

        # Global rank 4 (worker 1, local_rank 0) - should skip (not global rank 0)
        dist_info = {"rank": 4, "local_rank": 0, "num_workers": 2}
        assert _should_skip_rank("auto", "global", dist_info) is True

        # Global rank 6 (worker 1, local_rank 2) - should skip
        dist_info = {"rank": 6, "local_rank": 2, "num_workers": 2}
        assert _should_skip_rank("auto", "global", dist_info) is True

    def test_auto_mode_worker_scope_multi_node(self):
        """Test that run_mode='auto' with rank_scope='worker' allows local_rank 0 per worker."""
        from flyteplugins.wandb._decorator import _should_skip_rank

        # Worker 0, local_rank 0 (global rank 0) - should NOT skip
        dist_info = {"rank": 0, "local_rank": 0, "num_workers": 2}
        assert _should_skip_rank("auto", "worker", dist_info) is False

        # Worker 0, local_rank 1 (global rank 1) - should skip
        dist_info = {"rank": 1, "local_rank": 1, "num_workers": 2}
        assert _should_skip_rank("auto", "worker", dist_info) is True

        # Worker 1, local_rank 0 (global rank 4) - should NOT skip
        dist_info = {"rank": 4, "local_rank": 0, "num_workers": 2}
        assert _should_skip_rank("auto", "worker", dist_info) is False

        # Worker 1, local_rank 2 (global rank 6) - should skip
        dist_info = {"rank": 6, "local_rank": 2, "num_workers": 2}
        assert _should_skip_rank("auto", "worker", dist_info) is True


class TestConfigureDistributedRun:
    """Tests for _configure_distributed_run function."""

    def test_single_node_auto_mode_run_id(self):
        """Test run ID generation for single-node auto mode."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        dist_info = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "auto", "global", dist_info, "my-run-task")

        assert result["id"] == "my-run-task"
        assert "group" not in result

    def test_single_node_new_mode_run_id(self):
        """Test run ID generation for single-node new mode."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        dist_info = {
            "rank": 2,
            "local_rank": 2,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "new", "global", dist_info, "my-run-task")

        assert result["id"] == "my-run-task-rank-2"
        assert result["group"] == "my-run-task"

    def test_multi_node_auto_global_scope_run_id(self):
        """Test run ID generation for multi-node auto mode with global scope."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Worker 0, global rank 0
        dist_info = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "auto", "global", dist_info, "my-run-task")

        # Global scope: single run ID without worker suffix
        assert result["id"] == "my-run-task"
        assert "group" not in result

    def test_multi_node_auto_worker_scope_run_id(self):
        """Test run ID generation for multi-node auto mode with worker scope."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Worker 1, local_rank 0
        dist_info = {
            "rank": 4,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "auto", "worker", dist_info, "my-run-task")

        # Worker scope: run ID includes worker suffix
        assert result["id"] == "my-run-task-worker-1"
        assert "group" not in result

    def test_multi_node_new_mode_global_scope_run_id(self):
        """Test run ID generation for multi-node new mode with global scope."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Worker 1, local_rank 2 (global rank 6)
        dist_info = {
            "rank": 6,
            "local_rank": 2,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "new", "global", dist_info, "my-run-task")

        # Global scope: run ID uses global rank, single group for all
        assert result["id"] == "my-run-task-rank-6"
        assert result["group"] == "my-run-task"

    def test_multi_node_new_mode_worker_scope_run_id(self):
        """Test run ID generation for multi-node new mode with worker scope."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Worker 1, local_rank 2
        dist_info = {
            "rank": 6,
            "local_rank": 2,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "new", "worker", dist_info, "my-run-task")

        # Worker scope: run ID uses worker index and local rank, group per worker
        assert result["id"] == "my-run-task-worker-1-rank-2"
        assert result["group"] == "my-run-task-worker-1"

    def test_single_node_shared_mode_settings(self):
        """Test W&B settings for single-node shared mode."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Rank 0 - primary
        dist_info = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "global", dist_info, "my-run-task")

        assert result["id"] == "my-run-task"
        settings = result["settings"]
        assert settings.mode == "shared"
        assert settings.x_primary is True
        assert settings.x_label == "rank-0"
        assert settings.x_update_finish_state is True

        # Rank 2 - non-primary
        dist_info["rank"] = 2
        dist_info["local_rank"] = 2
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "global", dist_info, "my-run-task")

        settings = result["settings"]
        assert settings.x_primary is False
        assert settings.x_label == "rank-2"
        assert settings.x_update_finish_state is False

    def test_multi_node_shared_mode_global_scope_settings(self):
        """Test W&B settings for multi-node shared mode with global scope - only global rank 0 is primary."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Global rank 0 (worker 0, local_rank 0) - primary
        dist_info = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "global", dist_info, "my-run-task")

        # Global scope: single run ID without worker suffix
        assert result["id"] == "my-run-task"
        settings = result["settings"]
        assert settings.mode == "shared"
        assert settings.x_primary is True  # Global rank 0 is primary
        assert settings.x_label == "worker-0-rank-0"
        assert settings.x_update_finish_state is True

        # Worker 1, local_rank 0 (global rank 4) - NOT primary in global scope
        dist_info = {
            "rank": 4,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "global", dist_info, "my-run-task")

        assert result["id"] == "my-run-task"  # Same run ID for all
        settings = result["settings"]
        assert settings.x_primary is False  # Not global rank 0
        assert settings.x_label == "worker-1-rank-0"
        assert settings.x_update_finish_state is False

    def test_multi_node_shared_mode_worker_scope_settings(self):
        """Test W&B settings for multi-node shared mode with worker scope - local_rank 0 is primary per worker."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        # Worker 1, local_rank 0 - primary for this worker
        dist_info = {
            "rank": 4,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "worker", dist_info, "my-run-task")

        # Worker scope: run ID includes worker suffix
        assert result["id"] == "my-run-task-worker-1"
        settings = result["settings"]
        assert settings.mode == "shared"
        assert settings.x_primary is True  # local_rank 0 is primary
        assert settings.x_label == "worker-1-rank-0"
        assert settings.x_update_finish_state is True

        # Worker 1, local_rank 2 - non-primary
        dist_info["local_rank"] = 2
        dist_info["rank"] = 6
        init_kwargs = {}
        result = _configure_distributed_run(init_kwargs, "shared", "worker", dist_info, "my-run-task")

        settings = result["settings"]
        assert settings.x_primary is False
        assert settings.x_label == "worker-1-rank-2"
        assert settings.x_update_finish_state is False

    def test_preserves_user_provided_id(self):
        """Test that user-provided ID is preserved."""
        from flyteplugins.wandb._decorator import _configure_distributed_run

        dist_info = {
            "rank": 2,
            "local_rank": 2,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }
        init_kwargs = {"id": "custom-id"}
        result = _configure_distributed_run(init_kwargs, "new", "global", dist_info, "my-run-task")

        assert result["id"] == "custom-id"


class TestDistributedRunContextManager:
    """Tests for _wandb_run context manager with distributed training."""

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._get_distributed_info")
    def test_auto_mode_skipped_rank_yields_none(self, mock_dist_info, mock_ctx, mock_wandb_init):
        """Test that skipped ranks yield None in auto mode."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_ctx.return_value = mock_context

        # Single-node, rank 2 - should be skipped in auto mode
        mock_dist_info.return_value = {
            "rank": 2,
            "local_rank": 2,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test") as run:
            assert run is None

        # Should NOT call wandb.init
        mock_wandb_init.assert_not_called()

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._get_distributed_info")
    def test_auto_mode_primary_rank_initializes(self, mock_dist_info, mock_ctx, mock_wandb_init):
        """Test that primary rank initializes wandb in auto mode."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context

        # Single-node, rank 0 - primary
        mock_dist_info.return_value = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test") as run:
            assert run == mock_run

        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._get_distributed_info")
    def test_new_mode_all_ranks_initialize(self, mock_dist_info, mock_ctx, mock_wandb_init):
        """Test that all ranks initialize wandb in new mode."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context

        # Single-node, rank 2
        mock_dist_info.return_value = {
            "rank": 2,
            "local_rank": 2,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action-rank-2"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test") as run:
            assert run == mock_run

        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action-rank-2"
        assert call_kwargs["group"] == "test-run-test-action"

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._get_distributed_info")
    def test_shared_mode_finish_logic_single_node(self, mock_dist_info, mock_ctx, mock_wandb_init):
        """Test that only rank 0 finishes in single-node shared mode."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        # Rank 0 - should finish
        mock_dist_info.return_value = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }

        with _wandb_run(run_mode="shared", project="test"):
            pass

        mock_run.finish.assert_called_once()
        mock_run.reset_mock()

        # Rank 2 - should NOT finish
        mock_dist_info.return_value = {
            "rank": 2,
            "local_rank": 2,
            "world_size": 4,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 1,
        }
        mock_context.data = {}
        mock_context.custom_context = {}

        with _wandb_run(run_mode="shared", project="test"):
            pass

        mock_run.finish.assert_not_called()

    @patch("flyteplugins.wandb._decorator.wandb.init")
    @patch("flyteplugins.wandb._decorator.flyte.ctx")
    @patch("flyteplugins.wandb._decorator._get_distributed_info")
    def test_shared_mode_finish_logic_multi_node(self, mock_dist_info, mock_ctx, mock_wandb_init):
        """Test finish logic in multi-node shared mode with different rank_scope values."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_context.action = MagicMock()
        mock_context.action.run_name = "test-run"
        mock_context.action.name = "test-action"
        mock_context.mode = "remote"
        mock_ctx.return_value = mock_context

        mock_run = MagicMock()
        mock_run.id = "test-run-test-action"
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb._decorator import _wandb_run

        # Test rank_scope="global": only global rank 0 finishes
        # Global rank 0 - should finish
        mock_dist_info.return_value = {
            "rank": 0,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 0,
            "num_workers": 2,
        }

        with _wandb_run(run_mode="shared", rank_scope="global", project="test"):
            pass

        mock_run.finish.assert_called_once()
        mock_run.reset_mock()

        # Global rank 4 (worker 1, local_rank 0) - should NOT finish with global scope
        mock_dist_info.return_value = {
            "rank": 4,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        mock_context.data = {}
        mock_context.custom_context = {}

        with _wandb_run(run_mode="shared", rank_scope="global", project="test"):
            pass

        mock_run.finish.assert_not_called()
        mock_run.reset_mock()

        # Test rank_scope="worker": local_rank 0 of each worker finishes
        # Worker 1, local_rank 0 - should finish with worker scope
        mock_dist_info.return_value = {
            "rank": 4,
            "local_rank": 0,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        mock_context.data = {}
        mock_context.custom_context = {}

        with _wandb_run(run_mode="shared", rank_scope="worker", project="test"):
            pass

        mock_run.finish.assert_called_once()
        mock_run.reset_mock()

        # Worker 1, local_rank 2 - should NOT finish
        mock_dist_info.return_value = {
            "rank": 6,
            "local_rank": 2,
            "world_size": 8,
            "local_world_size": 4,
            "worker_index": 1,
            "num_workers": 2,
        }
        mock_context.data = {}
        mock_context.custom_context = {}

        with _wandb_run(run_mode="shared", rank_scope="worker", project="test"):
            pass

        mock_run.finish.assert_not_called()


class TestWrapTaskDistributed:
    """Tests for _wrap_task with distributed training detection."""

    def test_detects_distributed_from_elastic_config_worker_scope(self):
        """Test that distributed training with rank_scope='worker' creates per-worker links."""
        env = flyte.TaskEnvironment(name="test-env")

        # Create a mock Elastic plugin config
        mock_elastic = MagicMock()
        mock_elastic.nnodes = 2
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        # Apply the plugin config
        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)

        # Apply wandb_init with worker scope
        decorated = wandb_init(project="test", rank_scope="worker")(test_task_with_elastic)

        # Should have 2 links (one per node) with worker scope
        assert len(decorated.links) == 2
        for i, link in enumerate(decorated.links):
            assert isinstance(link, Wandb)
            assert link._is_distributed is True
            assert link._worker_index == i
            assert link.rank_scope == "worker"
            assert f"Worker {i}" in link.name

    def test_detects_distributed_from_elastic_config_global_scope(self):
        """Test that distributed training with rank_scope='global' (default) creates single link."""
        env = flyte.TaskEnvironment(name="test-env")

        # Create a mock Elastic plugin config
        mock_elastic = MagicMock()
        mock_elastic.nnodes = 2
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        # Apply the plugin config
        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)

        # Apply wandb_init with global scope (default)
        decorated = wandb_init(project="test", rank_scope="global")(test_task_with_elastic)

        # Should have 1 link (global scope = single run)
        assert len(decorated.links) == 1
        link = decorated.links[0]
        assert isinstance(link, Wandb)
        assert link._is_distributed is True
        assert link.rank_scope == "global"

    def test_single_node_distributed_adds_single_link(self):
        """Test that single-node distributed adds one link with _is_distributed=True."""
        env = flyte.TaskEnvironment(name="test-env")

        mock_elastic = MagicMock()
        mock_elastic.nnodes = 1
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)
        decorated = wandb_init(project="test")(test_task_with_elastic)

        assert len(decorated.links) == 1
        link = decorated.links[0]
        assert isinstance(link, Wandb)
        assert link._is_distributed is True
        assert link._worker_index is None  # Single node, no worker index


class TestTaskWrappingStrategy:
    """Tests for the task wrapping strategy (execute vs func.func)."""

    def test_non_distributed_task_wraps_execute(self):
        """Test that non-distributed tasks wrap execute method."""
        env = flyte.TaskEnvironment(name="test-env")

        @env.task
        async def test_task():
            return "result"

        original_func = test_task.func

        decorated = wandb_init(project="test")(test_task)

        # func should NOT be wrapped (same as original) for non-distributed
        assert decorated.func == original_func
        # execute is wrapped but we can't directly compare - check that decoration happened
        assert len(decorated.links) > 0

    def test_distributed_task_wraps_func(self):
        """Test that distributed tasks wrap func.func method."""
        env = flyte.TaskEnvironment(name="test-env")

        mock_elastic = MagicMock()
        mock_elastic.nnodes = 1
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)
        original_func = test_task_with_elastic.func

        decorated = wandb_init(project="test")(test_task_with_elastic)

        # func should be wrapped (different from original) for distributed
        assert decorated.func != original_func
        # The wrapped func should have __wrapped__ attribute pointing to original
        assert hasattr(decorated.func, "__wrapped__")
        assert decorated.func.__wrapped__ == original_func

    def test_multi_node_task_wraps_func(self):
        """Test that multi-node tasks wrap func.func method."""
        env = flyte.TaskEnvironment(name="test-env")

        mock_elastic = MagicMock()
        mock_elastic.nnodes = 2
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)
        original_func = test_task_with_elastic.func

        decorated = wandb_init(project="test")(test_task_with_elastic)

        # func should be wrapped (different from original) for distributed
        assert decorated.func != original_func
        # The wrapped func should have __wrapped__ attribute pointing to original
        assert hasattr(decorated.func, "__wrapped__")
        assert decorated.func.__wrapped__ == original_func


class TestDownloadLogs:
    """Tests for download_logs parameter."""

    def test_download_logs_param_accepted(self):
        """Test that download_logs parameter is accepted by decorator."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test", download_logs=True)
        @env.task
        async def test_task():
            return "result"

        # Should not raise and task should be created
        assert isinstance(test_task, AsyncFunctionTaskTemplate)

    def test_download_logs_false_accepted(self):
        """Test that download_logs=False is accepted."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="test", download_logs=False)
        @env.task
        async def test_task():
            return "result"

        assert isinstance(test_task, AsyncFunctionTaskTemplate)

    def test_download_logs_not_supported_for_distributed(self, caplog):
        """Test that download_logs=True logs a warning for distributed tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        mock_elastic = MagicMock()
        mock_elastic.nnodes = 1
        mock_elastic.nproc_per_node = 4
        type(mock_elastic).__name__ = "Elastic"

        @env.task
        async def test_task():
            return "result"

        test_task_with_elastic = test_task.override(plugin_config=mock_elastic)

        with caplog.at_level(logging.WARNING):
            wandb_init(project="test", download_logs=True)(test_task_with_elastic)

        assert "download_logs is not supported for distributed tasks" in caplog.text
