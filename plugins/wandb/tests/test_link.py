"""Tests for wandb link generation."""


from flyteplugins.wandb.link import Wandb, WandbSweep


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
        assert uri == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/context-entity/context-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/config-entity/config-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/config-entity/config-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/decorator-entity/decorator-project/runs/test-run-{{.actionName}}"

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

    def test_wandb_link_with_parent_run_id_new_run_true(self):
        """Test link generation with new_run=True (always creates new run)."""
        link = Wandb(project="test-project", entity="test-entity", new_run=True)

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
        assert uri == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"

    def test_wandb_link_with_parent_run_id_new_run_false(self):
        """Test link generation with new_run=False (always reuses parent)."""
        link = Wandb(project="test-project", entity="test-entity", new_run=False)

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

    def test_wandb_link_new_run_false_no_parent(self):
        """Test link generation with new_run=False but no parent run."""
        link = Wandb(project="test-project", entity="test-entity", new_run=False)

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

    def test_wandb_link_new_run_auto_with_parent(self):
        """Test link generation with new_run='auto' and parent exists."""
        link = Wandb(project="test-project", entity="test-entity", new_run="auto")

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

    def test_wandb_link_new_run_auto_without_parent(self):
        """Test link generation with new_run='auto' and no parent."""
        link = Wandb(project="test-project", entity="test-entity", new_run="auto")

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
        assert uri == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://custom.wandb.io/test-entity/test-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"

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

        assert uri == "https://wandb.ai/test-entity/test-project/runs/test-run-{{.actionName}}"


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
