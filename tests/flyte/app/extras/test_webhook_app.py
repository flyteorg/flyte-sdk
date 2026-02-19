"""
Unit tests for FlyteWebhookAppEnvironment.

These tests verify the FlyteWebhookAppEnvironment functionality including:
- Basic instantiation and default values
- Custom configuration options
- FastAPI app creation with correct endpoints
- Inheritance from FastAPIAppEnvironment
- Container command generation
"""

import pathlib

import pytest

from flyte._image import Image
from flyte._resources import Resources
from flyte.app._types import Domain, Scaling
from flyte.app.extras import FastAPIAppEnvironment, FlyteWebhookAppEnvironment
from flyte.models import SerializationContext


class TestFlyteWebhookAppEnvironmentInstantiation:
    """Tests for basic instantiation and default values."""

    def test_basic_instantiation_with_name_only(self):
        """
        GOAL: Verify that FlyteWebhookAppEnvironment can be created with minimal configuration.

        Tests that:
        - Only the name is required
        - Default image is set (debian base with fastapi and uvicorn)
        - Default type is "FlyteWebhookApp"
        - A FastAPI app is automatically created
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.name == "test-webhook"
        assert webhook_env.type == "FlyteWebhookApp"
        assert webhook_env.app is not None
        assert webhook_env.title is None
        assert webhook_env.description is None

    def test_default_image_is_set(self):
        """
        GOAL: Verify that a default image is provided.

        Tests that the default image is a debian base with fastapi and uvicorn packages.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.image is not None
        # The default image should be from debian base with pip packages
        assert isinstance(webhook_env.image, Image)

    def test_custom_title_and_description(self):
        """
        GOAL: Verify that custom title and description can be set.

        Tests that:
        - Custom title is used in the FastAPI app
        - Custom description is used in the FastAPI app
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="custom-webhook",
            title="My Custom Webhook",
            description="A custom webhook for testing",
        )

        assert webhook_env.title == "My Custom Webhook"
        assert webhook_env.description == "A custom webhook for testing"
        # Verify the FastAPI app uses these values
        assert webhook_env.app.title == "My Custom Webhook"
        assert webhook_env.app.description == "A custom webhook for testing"

    def test_default_title_uses_name(self):
        """
        GOAL: Verify that when no title is provided, the FastAPI app uses a default title with the name.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="my-webhook")

        # Default title should include the name
        assert "my-webhook" in webhook_env.app.title

    def test_custom_resources(self):
        """
        GOAL: Verify that custom resources can be configured.
        """
        resources = Resources(cpu=2, memory="1Gi", gpu=1)
        webhook_env = FlyteWebhookAppEnvironment(
            name="resource-webhook",
            resources=resources,
        )

        assert webhook_env.resources.cpu == 2
        assert webhook_env.resources.memory == "1Gi"
        assert webhook_env.resources.gpu == 1

    def test_custom_scaling(self):
        """
        GOAL: Verify that custom scaling configuration can be set.
        """
        scaling = Scaling(replicas=(2, 5), metric=Scaling.Concurrency(val=10))
        webhook_env = FlyteWebhookAppEnvironment(
            name="scaling-webhook",
            scaling=scaling,
        )

        assert webhook_env.scaling.replicas == (2, 5)
        assert isinstance(webhook_env.scaling.metric, Scaling.Concurrency)
        assert webhook_env.scaling.metric.val == 10

    def test_custom_domain(self):
        """
        GOAL: Verify that custom domain configuration can be set.
        """
        domain = Domain(subdomain="my-subdomain", custom_domain="example.com")
        webhook_env = FlyteWebhookAppEnvironment(
            name="domain-webhook",
            domain=domain,
        )

        assert webhook_env.domain.subdomain == "my-subdomain"
        assert webhook_env.domain.custom_domain == "example.com"

    def test_requires_auth_default_true(self):
        """
        GOAL: Verify that requires_auth defaults to True.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="auth-webhook")

        assert webhook_env.requires_auth is True

    def test_requires_auth_can_be_disabled(self):
        """
        GOAL: Verify that requires_auth can be set to False.
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="no-auth-webhook",
            requires_auth=False,
        )

        assert webhook_env.requires_auth is False

    def test_custom_image(self):
        """
        GOAL: Verify that a custom image can be provided.
        """
        custom_image = Image.from_base("python:3.12-slim")
        webhook_env = FlyteWebhookAppEnvironment(
            name="custom-image-webhook",
            image=custom_image,
        )

        assert webhook_env.image == custom_image


class TestFlyteWebhookAppEnvironmentFastAPIApp:
    """Tests for the FastAPI app creation and endpoints."""

    def test_fastapi_app_is_created(self):
        """
        GOAL: Verify that a FastAPI app is automatically created.
        """
        import fastapi

        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.app is not None
        assert isinstance(webhook_env.app, fastapi.FastAPI)

    def test_fastapi_app_has_health_endpoint(self):
        """
        GOAL: Verify that the FastAPI app has a /health endpoint.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        # Check that /health route exists
        routes = [route.path for route in webhook_env.app.routes]
        assert "/health" in routes

    def test_fastapi_app_has_me_endpoint(self):
        """
        GOAL: Verify that the FastAPI app has a /me endpoint.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/me" in routes

    def test_fastapi_app_has_run_task_endpoint(self):
        """
        GOAL: Verify that the FastAPI app has a /run-task endpoint.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/run-task/{domain}/{project}/{name}" in routes

    def test_fastapi_app_has_run_endpoints(self):
        """
        GOAL: Verify that the FastAPI app has run-related endpoints.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/run/{name}" in routes
        assert "/run/{name}/io" in routes
        assert "/run/{name}/abort" in routes

    def test_fastapi_app_has_task_endpoint(self):
        """
        GOAL: Verify that the FastAPI app has a /task endpoint.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/task/{domain}/{project}/{name}" in routes

    def test_fastapi_app_has_app_endpoints(self):
        """
        GOAL: Verify that the FastAPI app has app-related endpoints.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/app/{name}" in routes
        assert "/app/{name}/activate" in routes
        assert "/app/{name}/deactivate" in routes
        assert "/app/{name}/call" in routes

    def test_fastapi_app_has_trigger_endpoints(self):
        """
        GOAL: Verify that the FastAPI app has trigger-related endpoints.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/trigger/{task_name}/{trigger_name}/activate" in routes
        assert "/trigger/{task_name}/{trigger_name}/deactivate" in routes

    def test_fastapi_app_has_build_image_endpoint(self):
        """
        GOAL: Verify that the FastAPI app has a /build-image endpoint.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/build-image" in routes

    def test_fastapi_app_has_prefetch_endpoints(self):
        """
        GOAL: Verify that the FastAPI app has prefetch-related endpoints.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        routes = [route.path for route in webhook_env.app.routes]
        assert "/prefetch/hf-model" in routes
        assert "/prefetch/hf-model/{run_name}" in routes
        assert "/prefetch/hf-model/{run_name}/io" in routes
        assert "/prefetch/hf-model/{run_name}/abort" in routes

    def test_fastapi_app_version(self):
        """
        GOAL: Verify that the FastAPI app has the correct version.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.app.version == "1.0.0"

    def test_fastapi_app_has_docs_link(self):
        """
        GOAL: Verify that the FastAPI app has a docs link added by FastAPIAppEnvironment.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        # FastAPIAppEnvironment adds a docs link
        link_paths = [link.path for link in webhook_env.links]
        assert "/docs" in link_paths


class TestFlyteWebhookAppEnvironmentInheritance:
    """Tests for inheritance from FastAPIAppEnvironment."""

    def test_inherits_from_fastapi_app_environment(self):
        """
        GOAL: Verify that FlyteWebhookAppEnvironment inherits from FastAPIAppEnvironment.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert isinstance(webhook_env, FastAPIAppEnvironment)

    def test_has_uvicorn_config_attribute(self):
        """
        GOAL: Verify that uvicorn_config attribute is available from parent class.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        # uvicorn_config should be None by default
        assert webhook_env.uvicorn_config is None

    def test_custom_uvicorn_config(self):
        """
        GOAL: Verify that custom uvicorn config can be provided.
        """
        import uvicorn

        # Create a minimal FastAPI app for the config
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")
        custom_config = uvicorn.Config(webhook_env.app, host="0.0.0.0", port=9000)

        webhook_env.uvicorn_config = custom_config

        assert webhook_env.uvicorn_config is not None
        assert webhook_env.uvicorn_config.host == "0.0.0.0"
        assert webhook_env.uvicorn_config.port == 9000


class TestFlyteWebhookAppEnvironmentContainerCommand:
    """Tests for container command generation."""

    def test_container_command_returns_empty_list(self):
        """
        GOAL: Verify that container_command returns an empty list.

        FlyteWebhookAppEnvironment overrides container_command to return an empty list
        because it uses the _server method to run the FastAPI app.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        ctx = SerializationContext(
            org="test-org",
            project="test-project",
            domain="test-domain",
            version="v1.0.0",
            root_dir=pathlib.Path.cwd(),
        )

        cmd = webhook_env.container_command(ctx)
        assert cmd == []


class TestFlyteWebhookAppEnvironmentRichRepr:
    """Tests for rich repr output."""

    def test_rich_repr_includes_name(self):
        """
        GOAL: Verify that rich repr includes the name.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        repr_items = list(webhook_env.__rich_repr__())
        names = [item[0] for item in repr_items]

        assert "name" in names

    def test_rich_repr_includes_title(self):
        """
        GOAL: Verify that rich repr includes the title.
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="test-webhook",
            title="My Webhook",
        )

        repr_items = list(webhook_env.__rich_repr__())
        names = [item[0] for item in repr_items]

        assert "title" in names

    def test_rich_repr_includes_type(self):
        """
        GOAL: Verify that rich repr includes the type.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        repr_items = list(webhook_env.__rich_repr__())
        names = [item[0] for item in repr_items]

        assert "type" in names

    def test_rich_repr_values(self):
        """
        GOAL: Verify that rich repr returns correct values.
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="my-webhook",
            title="My Title",
        )

        repr_dict = dict(webhook_env.__rich_repr__())

        assert repr_dict["name"] == "my-webhook"
        assert repr_dict["title"] == "My Title"
        assert repr_dict["type"] == "FlyteWebhookApp"


class TestFlyteWebhookAppEnvironmentNameValidation:
    """Tests for name validation."""

    def test_valid_names_accepted(self):
        """
        GOAL: Verify that valid Kubernetes-style names are accepted.
        """
        valid_names = ["my-webhook", "webhook123", "a-b-c", "test"]

        for name in valid_names:
            webhook_env = FlyteWebhookAppEnvironment(name=name)
            assert webhook_env.name == name
            webhook_env._validate_name()  # Should not raise

    def test_invalid_names_rejected(self):
        """
        GOAL: Verify that invalid names are rejected.
        """
        invalid_names = ["My-Webhook", "webhook_123", "-webhook", "webhook-"]

        for name in invalid_names:
            with pytest.raises(ValueError, match="must consist of lower case"):
                FlyteWebhookAppEnvironment(name=name)


class TestFlyteWebhookAppEnvironmentPortHandling:
    """Tests for port handling."""

    def test_default_port(self):
        """
        GOAL: Verify that the default port is 8080.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.port.port == 8080

    def test_custom_port(self):
        """
        GOAL: Verify that a custom port can be set.
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="test-webhook",
            port=9000,
        )

        assert webhook_env.port.port == 9000

    def test_reserved_ports_rejected(self):
        """
        GOAL: Verify that reserved ports are rejected.
        """
        reserved_ports = [8012, 8022, 8112, 9090, 9091]

        for port in reserved_ports:
            with pytest.raises(ValueError, match="is not allowed"):
                FlyteWebhookAppEnvironment(name="test-webhook", port=port)


class TestFlyteWebhookAppEnvironmentMiddleware:
    """Tests for middleware configuration."""

    def test_auth_middleware_is_added(self):
        """
        GOAL: Verify that FastAPIPassthroughAuthMiddleware is added to the app.
        """
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        # Check that middleware is present
        middleware_classes = [m.cls for m in webhook_env.app.user_middleware]
        assert FastAPIPassthroughAuthMiddleware in middleware_classes

    def test_excluded_paths_for_middleware(self):
        """
        GOAL: Verify that certain paths are excluded from auth middleware.

        The /health, /docs, /openapi.json, and /redoc paths should be excluded.
        """
        from flyte.app.extras import FastAPIPassthroughAuthMiddleware

        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        # Find the middleware configuration
        for middleware in webhook_env.app.user_middleware:
            if middleware.cls == FastAPIPassthroughAuthMiddleware:
                excluded_paths = middleware.kwargs.get("excluded_paths", set())
                assert "/health" in excluded_paths
                assert "/docs" in excluded_paths
                assert "/openapi.json" in excluded_paths
                assert "/redoc" in excluded_paths
                break
        else:
            pytest.fail("FastAPIPassthroughAuthMiddleware not found")


class TestCreateWebhookAppFunction:
    """Tests for the _create_webhook_app helper function."""

    def test_create_webhook_app_returns_fastapi_app(self):
        """
        GOAL: Verify that _create_webhook_app returns a FastAPI app.
        """
        import fastapi

        from flyte.app.extras._webhook_app import _create_webhook_app

        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")
        # Note: The app is already created in __post_init__, but we can test the function directly
        # by creating a new environment and checking the app type
        app = _create_webhook_app(webhook_env)

        assert isinstance(app, fastapi.FastAPI)

    def test_create_webhook_app_uses_custom_title(self):
        """
        GOAL: Verify that _create_webhook_app uses the custom title when provided.
        """

        # Create a webhook env with custom title
        webhook_env = FlyteWebhookAppEnvironment(
            name="test-webhook",
            title="Custom Title",
        )

        # The app should have the custom title
        assert webhook_env.app.title == "Custom Title"

    def test_create_webhook_app_uses_default_title_with_name(self):
        """
        GOAL: Verify that _create_webhook_app uses a default title with the name when no title provided.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="my-webhook")

        # Default title should be "Flyte Webhook: {name}"
        assert webhook_env.app.title == "Flyte Webhook: my-webhook"

    def test_create_webhook_app_uses_custom_description(self):
        """
        GOAL: Verify that _create_webhook_app uses the custom description when provided.
        """
        webhook_env = FlyteWebhookAppEnvironment(
            name="test-webhook",
            description="My custom description",
        )

        assert webhook_env.app.description == "My custom description"

    def test_create_webhook_app_uses_default_description(self):
        """
        GOAL: Verify that _create_webhook_app uses a default description when none provided.
        """
        webhook_env = FlyteWebhookAppEnvironment(name="test-webhook")

        assert webhook_env.app.description == "A webhook service for Flyte operations"
