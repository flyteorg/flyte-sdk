"""Tests for wandb decorators."""

from unittest.mock import MagicMock, patch

import flyte
import pytest
from flyte._task import AsyncFunctionTaskTemplate

from flyteplugins.wandb.decorator import _build_init_kwargs, wandb_init, wandb_sweep
from flyteplugins.wandb.link import Wandb, WandbSweep


class TestBuildInitKwargs:
    """Tests for _build_init_kwargs helper function."""

    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    def test_build_init_kwargs_with_context(self, mock_get_context):
        """Test building init kwargs when context exists."""
        mock_config = MagicMock()
        mock_config.project = "test-project"
        mock_config.entity = "test-entity"
        mock_config.tags = ["tag1", "tag2"]
        mock_config.kwargs = {"custom_key": "custom_value"}
        mock_get_context.return_value = mock_config

        with patch("flyteplugins.wandb.decorator.asdict") as mock_asdict:
            mock_asdict.return_value = {
                "project": "test-project",
                "entity": "test-entity",
                "tags": ["tag1", "tag2"],
                "kwargs": {"custom_key": "custom_value"},
            }

            result = _build_init_kwargs()

            assert result["project"] == "test-project"
            assert result["entity"] == "test-entity"
            assert result["tags"] == ["tag1", "tag2"]
            assert result["custom_key"] == "custom_value"
            assert "kwargs" not in result  # Should be merged and removed

    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    def test_build_init_kwargs_no_context(self, mock_get_context):
        """Test building init kwargs when no context exists."""
        mock_get_context.return_value = None

        result = _build_init_kwargs()

        assert result == {}

    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    def test_build_init_kwargs_no_extra_kwargs(self, mock_get_context):
        """Test building init kwargs when context has no extra kwargs."""
        mock_config = MagicMock()
        mock_config.project = "test-project"
        mock_config.kwargs = None
        mock_get_context.return_value = mock_config

        with patch("flyteplugins.wandb.decorator.asdict") as mock_asdict:
            mock_asdict.return_value = {
                "project": "test-project",
                "kwargs": None,
            }

            result = _build_init_kwargs()

            assert result["project"] == "test-project"
            assert "kwargs" not in result


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

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # Should initialize wandb eagerly and store the run
            assert "_wandb_run" in mock_context.data
            assert mock_context.data["_wandb_run"] == mock_run
            # Should also store run ID in custom_context
            assert mock_context.custom_context["_wandb_run_id"] == "test-run-id"

    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
    def test_wandb_run_restores_state_on_exit(self, mock_build_kwargs, mock_ctx):
        """Test that _wandb_run restores state on exit."""
        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": "existing_run"}
        mock_context.custom_context = {"_wandb_run_id": "existing_id"}
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Should restore previous state
        assert mock_context.data["_wandb_run"] == "existing_run"
        assert mock_context.custom_context["_wandb_run_id"] == "existing_id"

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # Run should be initialized during context
            assert "_wandb_run" in mock_context.data

        # Run and metadata should be cleaned up after context exits (if not saved)
        # Since there was no saved state, it should be removed
        assert "_wandb_run" not in mock_context.data
        assert "_wandb_run_id" not in mock_context.custom_context

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    def test_wandb_run_func_mode_without_flyte_context(self, mock_ctx, mock_wandb_init):
        """Test _wandb_run in func mode (no Flyte context)."""
        mock_ctx.return_value = None
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", func=True, project="test"):
            pass

        # Should call wandb.init directly in func mode
        mock_wandb_init.assert_called_once()
        mock_run.finish.assert_called()

    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    def test_wandb_run_func_mode_with_flyte_context_raises_error(self, mock_ctx):
        """Test that func mode with Flyte context raises error."""
        mock_context = MagicMock()
        mock_ctx.return_value = mock_context

        from flyteplugins.wandb.decorator import _wandb_run

        with pytest.raises(RuntimeError, match="cannot be applied to traces"):
            with _wandb_run(run_mode="new", func=True):
                pass


class TestCreateSweep:
    """Tests for _create_sweep context manager."""

    @patch("flyteplugins.wandb.decorator.wandb.sweep")
    @patch("flyteplugins.wandb.decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
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

        from flyteplugins.wandb.decorator import _create_sweep

        with _create_sweep() as sweep_id:
            assert sweep_id == "sweep-123"
            # Should store sweep_id in context
            assert mock_context.custom_context["_wandb_sweep_id"] == "sweep-123"

        # Should clean up sweep_id on exit
        assert "_wandb_sweep_id" not in mock_context.custom_context

        # Should have added deterministic name to sweep
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["sweep"]["name"] == "test-run-test-action"

    @patch("flyteplugins.wandb.decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    def test_create_sweep_no_config_raises_error(self, mock_ctx, mock_get_sweep_ctx):
        """Test that missing sweep config raises error."""
        mock_context = MagicMock()
        mock_ctx.return_value = mock_context
        mock_get_sweep_ctx.return_value = None

        from flyteplugins.wandb.decorator import _create_sweep

        with pytest.raises(RuntimeError, match="No wandb sweep config found"):
            with _create_sweep():
                pass

    @patch("flyteplugins.wandb.decorator.wandb.sweep")
    @patch("flyteplugins.wandb.decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
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

        from flyteplugins.wandb.decorator import _create_sweep

        with _create_sweep():
            pass

        # Should use fallback project/entity and generate deterministic name
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["project"] == "fallback-project"
        assert call_args[1]["entity"] == "fallback-entity"
        assert call_args[1]["prior_runs"] == []
        assert call_args[1]["sweep"]["method"] == "random"
        assert call_args[1]["sweep"]["name"] == "test-run-test-action"

    @patch("flyteplugins.wandb.decorator.wandb.sweep")
    @patch("flyteplugins.wandb.decorator.get_wandb_sweep_context")
    @patch("flyteplugins.wandb.decorator.get_wandb_context")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
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

        from flyteplugins.wandb.decorator import _create_sweep

        with _create_sweep():
            pass

        # Should preserve user-provided name
        call_args = mock_wandb_sweep.call_args
        assert call_args[1]["sweep"]["name"] == "custom-sweep-name"


class TestDecoratorIntegration:
    """Integration tests for decorator combinations."""

    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Verify wandb.init was called with reinit='create_new'
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Verify wandb.init was called with reinit='return_previous'
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["reinit"] == "return_previous"
        assert call_kwargs["id"] == "parent-run-id"

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.wandb.Settings")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

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

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.wandb.Settings")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Verify wandb.Settings was called with secondary task config
        settings_kwargs = mock_settings.call_args[1]
        assert settings_kwargs["mode"] == "shared"
        assert settings_kwargs["x_primary"] is False
        assert settings_kwargs["x_label"] == "child-action"
        assert settings_kwargs["x_update_finish_state"] is False

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
    def test_trace_detection_yields_existing_run(self, mock_build_kwargs, mock_ctx, mock_wandb_init):
        """Test that when a run exists in ctx.data, it yields that run without re-initializing."""
        mock_existing_run = MagicMock()
        mock_existing_run.id = "existing-run-id"

        mock_context = MagicMock()
        mock_context.data = {"_wandb_run": mock_existing_run}
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context
        mock_build_kwargs.return_value = {}

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test") as run:
            # Should yield existing run
            assert run == mock_existing_run

        # Should NOT call wandb.init
        mock_wandb_init.assert_not_called()

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test"):
            pass

        # Verify it used parent's run_id
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "parent-run-id"
        assert call_kwargs["reinit"] == "return_previous"

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="auto", project="test"):
            pass

        # Verify it created new run_id
        call_kwargs = mock_wandb_init.call_args[1]
        assert call_kwargs["id"] == "test-run-test-action"
        assert call_kwargs["reinit"] == "create_new"

    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with pytest.raises(RuntimeError, match="Cannot reuse parent run: no parent run ID found"):
            with _wandb_run(run_mode="shared", project="test"):
                pass


class TestRunFinishingLogic:
    """Tests for run finishing logic in local vs remote mode."""

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Should call run.finish in remote mode
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Even secondary tasks should call finish in remote mode (x_update_finish_state=False prevents actual finish)
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            pass

        # Primary task in local mode should call finish
        mock_run.finish.assert_called_once_with(exit_code=0)

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="shared", project="test"):
            pass

        # Secondary task in local mode should NOT call finish (shares parent's run object)
        mock_run.finish.assert_not_called()


class TestStateSaveRestore:
    """Tests for complete state save and restore."""

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        # Execute child context
        with _wandb_run(run_mode="new", project="child-project", entity="child-entity"):
            # During execution, child state should be active
            assert mock_context.data["_wandb_run"] == mock_child_run
            assert mock_context.custom_context["_wandb_run_id"] == "child-run-id"

        # After exit, parent state in custom_context should be restored
        # ctx.data should be empty (no parent run in ctx.data for child tasks)
        assert "_wandb_run" not in mock_context.data
        assert mock_context.custom_context["_wandb_run_id"] == "parent-run-id"

    @patch("flyteplugins.wandb.decorator.wandb.init")
    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
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

        from flyteplugins.wandb.decorator import _wandb_run

        with _wandb_run(run_mode="new", project="test"):
            # State should be set during execution
            assert "_wandb_run" in mock_context.data
            assert "_wandb_run_id" in mock_context.custom_context

        # All state should be cleaned up after exit
        assert "_wandb_run" not in mock_context.data
        assert "_wandb_run_id" not in mock_context.custom_context
