"""Tests for wandb link generation."""

from flyteplugins.wandb import Wandb, WandbSweep


class TestWandbLink:
    """Tests for Wandb link class."""

    def test_wandb_link_with_decorator_params(self):
        """Test link generation when project/entity are provided at decoration time."""
        link = Wandb(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use decorator params and generate run ID
        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_with_context_runtime_keys(self):
        """Test link generation when project/entity are in context (config keys only)."""
        link = Wandb()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "context-project",
                "wandb_entity": "context-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/context-entity/context-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_with_context_config_keys(self):
        """Test link generation when project/entity are in context with config keys."""
        link = Wandb()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "config-project",
                "wandb_entity": "config-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/config-entity/config-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_uses_config_keys(self):
        """Test that config keys (wandb_*) are used when available."""
        link = Wandb()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "config-project",
                "wandb_entity": "config-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/config-entity/config-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_decorator_params_take_precedence_over_context(self):
        """Test that decorator params take precedence over context."""
        link = Wandb(project="decorator-project", entity="decorator-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "context-project",
                "wandb_entity": "context-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/decorator-entity/decorator-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_missing_project(self):
        """Test link generation when project is missing."""
        link = Wandb(entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when project is missing
        assert uri == "https://wandb.ai"

    def test_wandb_link_missing_entity(self):
        """Test link generation when entity is missing."""
        link = Wandb(project="test-project")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when entity is missing
        assert uri == "https://wandb.ai"

    def test_wandb_link_with_parent_run_id_run_mode_new(self):
        """Test link generation with run_mode="new" (always creates new run)."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="new")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_run_id": "parent-run-id",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should always create new run ID even when parent exists
        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_with_parent_run_id_run_mode_shared(self):
        """Test link generation with run_mode="shared" (always reuses parent)."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="shared")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_run_id": "parent-run-id",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should reuse parent's run ID
        assert uri == "https://wandb.ai/test-entity/test-project/runs/parent-run-id"

    def test_wandb_link_run_mode_shared_no_parent(self):
        """Test link generation with run_mode="shared" but no parent run."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="shared")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return project URL without run ID when can't reuse parent
        assert uri == "https://wandb.ai/test-entity/test-project"

    def test_wandb_link_run_mode_auto_with_parent(self):
        """Test link generation with run_mode='auto' and parent exists."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="auto")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_run_id": "parent-run-id",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should reuse parent's run ID when available
        assert uri == "https://wandb.ai/test-entity/test-project/runs/parent-run-id"

    def test_wandb_link_run_mode_auto_without_parent(self):
        """Test link generation with run_mode='auto' and no parent."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="auto")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should create new run ID when no parent available
        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_custom_host(self):
        """Test link generation with custom wandb host."""
        link = Wandb(
            host="https://custom.wandb.io",
            project="test-project",
            entity="test-entity",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://custom.wandb.io/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_empty_context(self):
        """Test link generation with empty context dict."""
        link = Wandb(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_none_context(self):
        """Test link generation with None context."""
        link = Wandb(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context=None,
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_with_custom_id_in_link(self):
        """Test link generation with custom run ID provided to Wandb link."""
        link = Wandb(
            project="test-project",
            entity="test-entity",
            run_mode="new",
            id="my-custom-run-id",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use custom ID from link parameter
        assert uri == "https://wandb.ai/test-entity/test-project/runs/my-custom-run-id"

    def test_wandb_link_with_custom_id_in_context(self):
        """Test link generation with custom run ID from context."""
        link = Wandb(project="test-project", entity="test-entity", run_mode="new")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={"wandb_id": "context-custom-id"},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use custom ID from context
        assert uri == "https://wandb.ai/test-entity/test-project/runs/context-custom-id"

    def test_wandb_link_custom_id_priority(self):
        """Test that link.id takes precedence over context wandb_id."""
        link = Wandb(
            project="test-project",
            entity="test-entity",
            run_mode="new",
            id="link-custom-id",  # This should win
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={"wandb_id": "context-custom-id"},  # This should be ignored
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Link ID should take precedence over context ID
        assert uri == "https://wandb.ai/test-entity/test-project/runs/link-custom-id"

    def test_wandb_link_custom_id_with_run_mode_auto(self):
        """Test custom ID works with run_mode='auto' when no parent run exists."""
        link = Wandb(
            project="test-project",
            entity="test-entity",
            run_mode="auto",
            id="auto-custom-id",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},  # No parent run ID
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use custom ID since no parent run exists
        assert uri == "https://wandb.ai/test-entity/test-project/runs/auto-custom-id"

    def test_wandb_link_default_run_mode_is_auto(self):
        """Test that default run_mode is 'auto'."""
        link = Wandb(project="test-project", entity="test-entity")  # No run_mode set

        assert link.run_mode == "auto"

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_run_id": "parent-run-id",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # With auto mode and parent run ID, should use parent's run
        assert uri == "https://wandb.ai/test-entity/test-project/runs/parent-run-id"

    def test_wandb_link_auto_mode_without_parent(self):
        """Test that auto mode creates new run when no parent."""
        link = Wandb(project="test-project", entity="test-entity")  # Defaults to auto

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},  # No parent run ID
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should create new run ID since no parent
        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )

    def test_wandb_link_explicit_run_mode_new(self):
        """Test that explicit run_mode='new' always creates new run."""
        link = Wandb(
            project="test-project",
            entity="test-entity",
            run_mode="new",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_run_id": "parent-run-id",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should create new run ID even though parent exists
        assert (
            uri
            == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"
        )


class TestWandbSweepLink:
    """Tests for WandbSweep link class."""

    def test_wandb_sweep_link_with_decorator_params(self):
        """Test sweep link generation when project/entity are provided at decoration time."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return sweeps list URL when no sweep_id in context
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps"

    def test_wandb_sweep_link_with_context_runtime_keys(self):
        """Test sweep link generation with context (config keys only)."""
        link = WandbSweep()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "context-project",
                "wandb_entity": "context-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri == "https://wandb.ai/context-entity/context-project/sweeps"

    def test_wandb_sweep_link_with_context_config_keys(self):
        """Test sweep link generation with config context keys."""
        link = WandbSweep()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "config-project",
                "wandb_entity": "config-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri == "https://wandb.ai/config-entity/config-project/sweeps"

    def test_wandb_sweep_link_with_deterministic_name(self):
        """Test sweep link generation when no sweep_id available."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="my-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="sweep-task",
            pod_name="{{.podName}}",
        )

        # Should return sweeps list URL when no sweep_id in context
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps"

    def test_wandb_sweep_link_uses_config_keys(self):
        """Test that config keys are used when available."""
        link = WandbSweep()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "config-project",
                "wandb_entity": "config-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri == "https://wandb.ai/config-entity/config-project/sweeps"

    def test_wandb_sweep_link_missing_project(self):
        """Test sweep link generation when project is missing."""
        link = WandbSweep(entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when project is missing
        assert uri == "https://wandb.ai"

    def test_wandb_sweep_link_missing_entity(self):
        """Test sweep link generation when entity is missing."""
        link = WandbSweep(project="test-project")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should return base host when entity is missing
        assert uri == "https://wandb.ai"

    def test_wandb_sweep_link_custom_host(self):
        """Test sweep link generation with custom wandb host."""
        link = WandbSweep(
            host="https://custom.wandb.io",
            project="test-project",
            entity="test-entity",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri == "https://custom.wandb.io/test-entity/test-project/sweeps"

    def test_wandb_sweep_link_empty_context(self):
        """Test sweep link generation with empty context dict."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        assert uri == "https://wandb.ai/test-entity/test-project/sweeps"

    def test_wandb_sweep_link_none_context(self):
        """Test sweep link generation with None context."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context=None,
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should handle None context gracefully
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps"

    def test_wandb_sweep_link_with_all_context_keys(self):
        """Test sweep link with config context keys."""
        link = WandbSweep()

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "wandb_project": "config-project",
                "wandb_entity": "config-entity",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use config keys and link to sweeps list
        assert uri == "https://wandb.ai/config-entity/config-project/sweeps"

    def test_wandb_sweep_link_with_sweep_id_in_context(self):
        """Test sweep link generation with sweep_id from parent context."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={
                "_wandb_sweep_id": "sweep-abc123",
            },
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should link to specific sweep when sweep_id available
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps/sweep-abc123"

    def test_wandb_sweep_link_with_custom_id_in_link(self):
        """Test sweep link generation with custom sweep ID provided to WandbSweep link."""
        link = WandbSweep(
            project="test-project",
            entity="test-entity",
            id="my-custom-sweep-id",
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use custom sweep ID from link parameter
        assert (
            uri == "https://wandb.ai/test-entity/test-project/sweeps/my-custom-sweep-id"
        )

    def test_wandb_sweep_link_with_custom_id_in_context(self):
        """Test sweep link generation with custom sweep ID from context."""
        link = WandbSweep(project="test-project", entity="test-entity")

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={"_wandb_sweep_id": "context-sweep-id"},
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Should use sweep ID from context
        assert (
            uri == "https://wandb.ai/test-entity/test-project/sweeps/context-sweep-id"
        )

    def test_wandb_sweep_link_custom_id_priority(self):
        """Test that link.id takes precedence over context _wandb_sweep_id."""
        link = WandbSweep(
            project="test-project",
            entity="test-entity",
            id="link-sweep-id",  # This should win
        )

        uri = link.get_link(
            run_name="test-run",
            project="flyte-project",
            domain="development",
            context={"_wandb_sweep_id": "context-sweep-id"},  # This should be ignored
            parent_action_name="parent",
            action_name="{{.actionName}}",
            pod_name="{{.podName}}",
        )

        # Link ID should take precedence over context ID
        assert uri == "https://wandb.ai/test-entity/test-project/sweeps/link-sweep-id"
