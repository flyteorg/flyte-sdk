"""Integration tests for wandb plugin end-to-end scenarios."""

from unittest.mock import MagicMock, patch

import pytest
from flyteplugins.wandb import (
    Wandb,
    WandbSweep,
    wandb_config,
    wandb_init,
    wandb_sweep,
    wandb_sweep_config,
)

import flyte


class TestWandbInitIntegration:
    """Integration tests for @wandb_init decorator with tasks."""

    def test_wandb_init_task_has_correct_link(self):
        """Test that a task decorated with @wandb_init has the correct link."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(
            project="integration-project", entity="integration-entity", run_mode="new"
        )
        @env.task
        async def integration_task():
            return "result"

        # Verify task has Wandb link
        assert len(integration_task.links) == 1
        link = integration_task.links[0]
        assert isinstance(link, Wandb)
        assert link.project == "integration-project"
        assert link.entity == "integration-entity"
        assert link.run_mode == "new"

    def test_multiple_wandb_init_tasks_independent(self):
        """Test that multiple tasks with @wandb_init are independent."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="project-1", entity="entity-1")
        @env.task
        async def task_1():
            return "result-1"

        @wandb_init(project="project-2", entity="entity-2", run_mode="shared")
        @env.task
        async def task_2():
            return "result-2"

        # Each task should have its own link configuration
        link_1 = task_1.links[0]
        link_2 = task_2.links[0]

        assert link_1.project == "project-1"
        assert link_1.entity == "entity-1"
        assert link_1.run_mode == "auto"  # default

        assert link_2.project == "project-2"
        assert link_2.entity == "entity-2"
        assert link_2.run_mode == "shared"

    @patch("flyteplugins.wandb.decorator.flyte.ctx")
    @patch("flyteplugins.wandb.decorator._build_init_kwargs")
    def test_wandb_init_with_context_config(self, mock_build_kwargs, mock_ctx):
        """Test @wandb_init with wandb_config from context."""
        mock_context = MagicMock()
        mock_context.data = {}
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        # Simulate context config
        mock_build_kwargs.return_value = {
            "project": "context-project",
            "entity": "context-entity",
            "tags": ["context-tag"],
        }

        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init  # No params, should use context
        @env.task
        async def task_with_context():
            return "result"

        # Link should not have project/entity set (gets from context at runtime)
        link = task_with_context.links[0]
        assert link.project is None
        assert link.entity is None


class TestWandbSweepIntegration:
    """Integration tests for @wandb_sweep decorator with tasks."""

    def test_wandb_sweep_task_has_correct_link(self):
        """Test that a task decorated with @wandb_sweep has the correct link."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_sweep
        @env.task
        async def sweep_task():
            return "sweep-result"

        # Verify task has WandbSweep link
        assert len(sweep_task.links) == 1
        link = sweep_task.links[0]
        assert isinstance(link, WandbSweep)

    def test_wandb_sweep_with_wandb_init_child_task(self):
        """Test sweep task that calls child tasks with @wandb_init."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(run_mode="new")
        @env.task
        async def objective_task(lr: float):
            return f"trained with lr={lr}"

        @wandb_sweep
        @env.task
        async def sweep_controller():
            return "sweep-complete"

        # Both should have their respective links
        assert isinstance(sweep_controller.links[0], WandbSweep)
        assert isinstance(objective_task.links[0], Wandb)


class TestContextManagerIntegration:
    """Integration tests for wandb config context managers."""

    @patch("flyte.ctx")
    @patch("flyte.custom_context")
    def test_wandb_config_context_manager_workflow(self, mock_custom_context, mock_ctx):
        """Test complete workflow with wandb_config as context manager."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_cm = MagicMock()
        mock_custom_context.return_value = mock_cm

        # Simulate using wandb_config in with block
        config = wandb_config(
            project="test-project", entity="test-entity", tags=["test"]
        )

        with config:
            # Inside the context, config should be active
            pass

        # Context manager should have been used
        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()

    @patch("flyte.ctx")
    @patch("flyte.custom_context")
    def test_wandb_sweep_config_context_manager_workflow(
        self, mock_custom_context, mock_ctx
    ):
        """Test complete workflow with wandb_sweep_config as context manager."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_cm = MagicMock()
        mock_custom_context.return_value = mock_cm

        # Simulate using wandb_sweep_config in with block
        config = wandb_sweep_config(
            method="random",
            metric={"name": "loss", "goal": "minimize"},
            parameters={"lr": {"min": 0.001, "max": 0.1}},
        )

        with config:
            # Inside the context, config should be active
            pass

        # Context manager should have been used
        mock_cm.__enter__.assert_called_once()
        mock_cm.__exit__.assert_called_once()


class TestLinkGenerationIntegration:
    """Integration tests for link generation in realistic scenarios."""

    def test_link_generation_with_full_context(self):
        """Test link generation with complete context."""
        link = Wandb(project="test-project", entity="test-entity")

        context = {
            "wandb_project": "context-project",
            "wandb_entity": "context-entity",
            "_wandb_run_id": "parent-run-123",
        }

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context=context,
            parent_action_name="parent_action",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use decorator params (highest priority)
        assert "test-entity" in uri
        assert "test-project" in uri

    def test_sweep_link_generation_with_full_context(self):
        """Test sweep link generation with complete context."""
        link = WandbSweep(project="test-project", entity="test-entity")

        context = {
            "wandb_project": "context-project",
            "wandb_entity": "context-entity",
        }

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context=context,
            parent_action_name="parent_action",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use decorator params and return sweeps list URL
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps"

    def test_link_generation_priority_chain(self):
        """Test that link parameter priority works correctly."""
        # Priority: decorator params > runtime keys > config keys

        # Case 1: Only config keys
        link1 = Wandb()
        uri1 = link1.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={"wandb_project": "proj1", "wandb_entity": "ent1"},
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )
        assert "ent1" in uri1 and "proj1" in uri1

        # Case 2: Decorator params override config keys
        link2 = Wandb(project="proj2", entity="ent2")
        uri2 = link2.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={
                "wandb_project": "proj1",
                "wandb_entity": "ent1",
            },
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )
        assert "ent2" in uri2 and "proj2" in uri2

        # Case 3: Decorator params override everything
        link3 = Wandb(project="proj3", entity="ent3")
        uri3 = link3.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={
                "wandb_project": "proj1",
                "wandb_entity": "ent1",
            },
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )
        assert "ent3" in uri3 and "proj3" in uri3


class TestRunModeBehaviorIntegration:
    """Integration tests for run_mode parameter behavior."""

    def test_run_mode_new_always_creates_new(self):
        """Test that run_mode="new" always creates a new run ID."""
        link = Wandb(project="proj", entity="ent", run_mode="new")

        # Even with parent run ID in context, should create new
        uri = link.get_link(
            run_name="test-run",
            project="proj",
            domain="dev",
            context={"_wandb_run_id": "parent-123"},
            parent_action_name="pa",
            action_name="child-action",
            pod_name="{{.podName}}",
        )

        # Should not use parent ID
        assert "parent-123" not in uri
        assert "child-action" in uri

    def test_run_mode_shared_reuses_parent(self):
        """Test that run_mode="shared" reuses parent's run ID."""
        link = Wandb(project="proj", entity="ent", run_mode="shared")

        # Should reuse parent run ID
        uri = link.get_link(
            run_name="test-run",
            project="proj",
            domain="dev",
            context={"_wandb_run_id": "parent-123"},
            parent_action_name="pa",
            action_name="child-action",
            pod_name="{{.podName}}",
        )

        assert "parent-123" in uri

    def test_run_mode_auto_with_parent(self):
        """Test that run_mode='auto' reuses parent when available."""
        link = Wandb(project="proj", entity="ent", run_mode="auto")

        # Should reuse parent when available
        uri = link.get_link(
            run_name="test-run",
            project="proj",
            domain="dev",
            context={"_wandb_run_id": "parent-123"},
            parent_action_name="pa",
            action_name="child-action",
            pod_name="{{.podName}}",
        )

        assert "parent-123" in uri

    def test_run_mode_auto_without_parent(self):
        """Test that run_mode='auto' creates new when no parent."""
        link = Wandb(project="proj", entity="ent", run_mode="auto")

        # Should create new run when no parent
        uri = link.get_link(
            run_name="test-run",
            project="proj",
            domain="dev",
            context={},
            parent_action_name="pa",
            action_name="child-action",
            pod_name="{{.podName}}",
        )

        assert "child-action" in uri


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_link_with_missing_required_params(self):
        """Test link generation with missing required parameters."""
        link = Wandb()  # No project/entity

        uri = link.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={},
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when params missing
        assert uri == "https://wandb.ai"

    def test_sweep_link_with_missing_required_params(self):
        """Test sweep link generation with missing required parameters."""
        link = WandbSweep()  # No project/entity

        uri = link.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={},
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when params missing
        assert uri == "https://wandb.ai"

    def test_sweep_decorator_on_non_task_raises_error(self):
        """Test that @wandb_sweep on non-task raises appropriate error."""
        with pytest.raises(RuntimeError, match="can only be used with Flyte tasks"):

            @wandb_sweep
            async def not_a_task():
                pass


class TestCustomHostIntegration:
    """Integration tests for custom wandb host."""

    def test_custom_host_in_link(self):
        """Test using custom wandb host."""
        link = Wandb(
            host="https://wandb.example.com",
            project="proj",
            entity="ent",
        )

        uri = link.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={},
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri.startswith("https://wandb.example.com")

    def test_custom_host_in_sweep_link(self):
        """Test using custom wandb host for sweeps."""
        link = WandbSweep(
            host="https://wandb.example.com",
            project="proj",
            entity="ent",
        )

        uri = link.get_link(
            run_name="run",
            project="proj",
            domain="dev",
            context={},
            parent_action_name="pa",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri.startswith("https://wandb.example.com")
        assert uri == "https://wandb.example.com/ent/proj/sweeps"


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    def test_parent_child_task_hierarchy(self):
        """Test parent task with child tasks using different run_mode settings."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init(project="proj", entity="ent", run_mode="auto")
        @env.task
        async def parent():
            return "parent-result"

        @wandb_init(project="proj", entity="ent", run_mode="new")
        @env.task
        async def child_new_run():
            return "child-result"

        @wandb_init(project="proj", entity="ent", run_mode="shared")
        @env.task
        async def child_reuse_run():
            return "child-result"

        # Each should have correct link configuration
        assert parent.links[0].run_mode == "auto"
        assert child_new_run.links[0].run_mode == "new"
        assert child_reuse_run.links[0].run_mode == "shared"

    def test_sweep_with_parallel_agents(self):
        """Test sweep task that spawns multiple agent tasks."""
        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init
        @env.task
        async def agent_task(agent_id: int, sweep_id: str):
            return f"agent-{agent_id}-complete"

        @wandb_sweep
        @env.task
        async def sweep_coordinator():
            return "sweep-started"

        # Both should have appropriate links
        assert isinstance(agent_task.links[0], Wandb)
        assert isinstance(sweep_coordinator.links[0], WandbSweep)

    @patch("flyte.ctx")
    @patch("flyte.custom_context")
    def test_config_override_pattern(self, mock_custom_context, mock_ctx):
        """Test pattern where child task overrides parent's config."""
        mock_context = MagicMock()
        mock_context.custom_context = {}
        mock_ctx.return_value = mock_context

        mock_cm = MagicMock()
        mock_custom_context.return_value = mock_cm

        env = flyte.TaskEnvironment(name="test-env")

        @wandb_init
        @env.task
        async def parent():
            # Parent uses base config
            return "parent"

        @wandb_init
        @env.task
        async def child():
            # Child can override with context manager
            return "child"

        # Simulate using config override
        override_config = wandb_config(tags=["child-override"])

        with override_config:
            pass  # Child task would run here with override

        # Both tasks should exist independently
        assert isinstance(parent.links[0], Wandb)
        assert isinstance(child.links[0], Wandb)
