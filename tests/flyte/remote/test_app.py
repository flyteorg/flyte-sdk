from unittest.mock import AsyncMock, patch

import pytest
from flyteidl2.app import app_definition_pb2
from flyteidl2.common import identity_pb2

from flyte.remote._app import App


class TestAppSpecsAreEqual:
    """Test suite for the _app_specs_are_equal static method."""

    @pytest.mark.asyncio
    async def test_identical_specs_are_equal(self):
        """Test that identical specs return True."""
        spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))

        result = await App._app_specs_are_equal(spec, spec)
        assert result is True

    @pytest.mark.asyncio
    async def test_different_specs_are_not_equal(self):
        """Test that different specs return False."""
        old_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        old_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))

        new_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )
        new_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))

        result = await App._app_specs_are_equal(old_spec, new_spec)
        assert result is False

    @pytest.mark.asyncio
    async def test_updated_spec_without_ingress_uses_old_ingress(self):
        """Test that when updated_app_spec has no ingress, old_app_spec's ingress is used for comparison."""
        old_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        old_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False, subdomain="test"))

        # New spec without ingress but same desired_state
        new_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        # Don't set ingress on new_spec

        result = await App._app_specs_are_equal(old_spec, new_spec)
        assert result is True

    @pytest.mark.asyncio
    async def test_updated_spec_with_different_ingress_not_equal(self):
        """Test that specs with different ingress settings are not equal."""
        old_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        old_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))

        new_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        new_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=True))

        result = await App._app_specs_are_equal(old_spec, new_spec)
        assert result is False

    @pytest.mark.asyncio
    async def test_old_spec_without_ingress_raises_assertion_error(self):
        """Test that an AssertionError is raised when old_app_spec has no ingress.

        The old app spec is expected to always have an ingress field set since
        it comes from an existing app in the system.
        """
        old_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        new_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )

        with pytest.raises(AssertionError):
            await App._app_specs_are_equal(old_spec, new_spec)


class TestAppReplace:
    """Test suite for the App.replace class method."""

    @pytest.fixture
    def mock_creator(self):
        """Create a mock creator identity."""
        creator = identity_pb2.EnrichedIdentity()
        creator.user.id.subject = "test-user"
        return creator

    @pytest.fixture
    def mock_app_pb2(self, mock_creator):
        """Create a mock app protobuf."""
        app_pb2 = app_definition_pb2.App(
            metadata=app_definition_pb2.Meta(
                id=app_definition_pb2.Identifier(
                    org="test-org",
                    project="test-project",
                    domain="test-domain",
                    name="test-app",
                ),
                revision=1,
                labels={"env": "test"},
            ),
            spec=app_definition_pb2.Spec(
                desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
            ),
            status=app_definition_pb2.Status(),
        )
        app_pb2.spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))
        app_pb2.spec.creator.CopyFrom(mock_creator)
        return app_pb2

    @pytest.fixture
    def mock_app(self, mock_app_pb2):
        """Create a mock App instance."""
        return App(mock_app_pb2)

    @pytest.mark.asyncio
    async def test_replace_skips_update_when_specs_equal(self, mock_app, mock_app_pb2):
        """Test that replace skips update when specs are equal."""
        # Create an updated spec that's the same as the existing one
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
        )
        updated_spec.ingress.CopyFrom(app_definition_pb2.IngressConfig(private=False))

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
            patch("flyte.remote._app.logger") as mock_logger,
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            result = await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
            )

            # Verify get was called
            mock_get.aio.assert_called_once_with(name="test-app", project=None, domain=None)

            # Verify update was NOT called since specs are equal
            mock_update.aio.assert_not_called()

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "No changes in the App spec" in mock_logger.warning.call_args[0][0]

            # Verify the original app was returned
            assert result == mock_app

    @pytest.mark.asyncio
    async def test_replace_updates_when_specs_different(self, mock_app, mock_app_pb2):
        """Test that replace calls update when specs are different."""
        # Create an updated spec that's different from the existing one
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        updated_app = App(mock_app_pb2)

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock(return_value=updated_app)

            result = await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
            )

            # Verify get was called
            mock_get.aio.assert_called_once_with(name="test-app", project=None, domain=None)

            # Verify update was called since specs are different
            mock_update.aio.assert_called_once()

            # Verify the updated app was returned
            assert result == updated_app

    @pytest.mark.asyncio
    async def test_replace_copies_creator_from_existing_app(self, mock_app, mock_app_pb2, mock_creator):
        """Test that replace copies the creator from the existing app to the updated spec."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        # Verify creator is not set initially
        assert not updated_spec.creator.user.id.subject

        updated_app = App(mock_app_pb2)

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock(return_value=updated_app)

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
            )

            # Verify creator was copied from existing app
            assert updated_spec.creator.user.id.subject == "test-user"

    @pytest.mark.asyncio
    async def test_replace_preserves_labels_when_not_provided(self, mock_app, mock_app_pb2):
        """Test that replace preserves existing labels when new labels are not provided."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
                labels=None,  # No labels provided
            )

            # Verify update was called with the existing labels
            call_args = mock_update.aio.call_args
            new_app_proto = call_args[0][0]
            assert dict(new_app_proto.metadata.labels) == {"env": "test"}

    @pytest.mark.asyncio
    async def test_replace_uses_new_labels_when_provided(self, mock_app, mock_app_pb2):
        """Test that replace uses new labels when provided."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        new_labels = {"env": "production", "team": "platform"}

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
                labels=new_labels,
            )

            # Verify update was called with the new labels
            call_args = mock_update.aio.call_args
            new_app_proto = call_args[0][0]
            assert dict(new_app_proto.metadata.labels) == new_labels

    @pytest.mark.asyncio
    async def test_replace_with_project_and_domain(self, mock_app, mock_app_pb2):
        """Test that replace passes project and domain to get."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
                project="custom-project",
                domain="custom-domain",
            )

            # Verify get was called with custom project and domain
            mock_get.aio.assert_called_once_with(
                name="test-app",
                project="custom-project",
                domain="custom-domain",
            )

    @pytest.mark.asyncio
    async def test_replace_preserves_metadata_id_and_revision(self, mock_app, mock_app_pb2):
        """Test that replace preserves the metadata id and revision from the existing app."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
            )

            # Verify update was called with correct metadata
            call_args = mock_update.aio.call_args
            new_app_proto = call_args[0][0]
            assert new_app_proto.metadata.id.name == "test-app"
            assert new_app_proto.metadata.id.project == "test-project"
            assert new_app_proto.metadata.id.domain == "test-domain"
            assert new_app_proto.metadata.id.org == "test-org"
            assert new_app_proto.metadata.revision == 1

    @pytest.mark.asyncio
    async def test_replace_preserves_status(self, mock_app, mock_app_pb2):
        """Test that replace preserves the status from the existing app."""
        updated_spec = app_definition_pb2.Spec(
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
        )

        with (
            patch.object(App, "get") as mock_get,
            patch.object(App, "update") as mock_update,
            patch("flyte.remote._app.ensure_client"),
        ):
            mock_get.aio = AsyncMock(return_value=mock_app)
            mock_update.aio = AsyncMock()

            await App.replace.aio(
                name="test-app",
                updated_app_spec=updated_spec,
                reason="test reason",
            )

            # Verify update was called with the existing status
            call_args = mock_update.aio.call_args
            new_app_proto = call_args[0][0]
            assert new_app_proto.status == mock_app_pb2.status


class TestAppProperties:
    """Test suite for App properties."""

    @pytest.fixture
    def app_pb2(self):
        """Create an app protobuf for testing properties."""
        app_pb2 = app_definition_pb2.App(
            metadata=app_definition_pb2.Meta(
                id=app_definition_pb2.Identifier(
                    org="test-org",
                    project="test-project",
                    domain="test-domain",
                    name="test-app",
                ),
                revision=5,
            ),
            spec=app_definition_pb2.Spec(
                desired_state=app_definition_pb2.Spec.DESIRED_STATE_STARTED,
            ),
            status=app_definition_pb2.Status(
                ingress=app_definition_pb2.Ingress(
                    public_url="https://test-app.example.com",
                ),
            ),
        )
        return app_pb2

    def test_name_property(self, app_pb2):
        """Test the name property."""
        app = App(app_pb2)
        assert app.name == "test-app"

    def test_revision_property(self, app_pb2):
        """Test the revision property."""
        app = App(app_pb2)
        assert app.revision == 5

    def test_endpoint_property(self, app_pb2):
        """Test the endpoint property."""
        app = App(app_pb2)
        assert app.endpoint == "https://test-app.example.com"

    def test_desired_state_property(self, app_pb2):
        """Test the desired_state property."""
        app = App(app_pb2)
        assert app.desired_state == app_definition_pb2.Spec.DESIRED_STATE_STARTED


class TestAppDeploymentStatus:
    """Test suite for App deployment status methods."""

    def test_is_active_when_active(self):
        """Test is_active returns True when deployment status is ACTIVE."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(
                conditions=[
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE,
                    ),
                ],
            ),
        )
        app = App(app_pb2)
        assert app.is_active() is True
        assert app.is_deactivated() is False

    def test_is_active_when_started(self):
        """Test is_active returns True when deployment status is STARTED."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(
                conditions=[
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STARTED,
                    ),
                ],
            ),
        )
        app = App(app_pb2)
        assert app.is_active() is True
        assert app.is_deactivated() is False

    def test_is_deactivated_when_stopped(self):
        """Test is_deactivated returns True when deployment status is STOPPED."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(
                conditions=[
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STOPPED,
                    ),
                ],
            ),
        )
        app = App(app_pb2)
        assert app.is_active() is False
        assert app.is_deactivated() is True

    def test_is_deactivated_when_unassigned(self):
        """Test is_deactivated returns True when deployment status is UNASSIGNED."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(
                conditions=[
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_UNASSIGNED,
                    ),
                ],
            ),
        )
        app = App(app_pb2)
        assert app.is_active() is False
        assert app.is_deactivated() is True

    def test_deployment_status_with_no_conditions(self):
        """Test deployment_status returns UNSPECIFIED when there are no conditions."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(conditions=[]),
        )
        app = App(app_pb2)
        assert app.deployment_status == app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_UNSPECIFIED

    def test_deployment_status_returns_latest_condition(self):
        """Test deployment_status returns the status from the latest condition."""
        app_pb2 = app_definition_pb2.App(
            status=app_definition_pb2.Status(
                conditions=[
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STOPPED,
                    ),
                    app_definition_pb2.Condition(
                        deployment_status=app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE,
                    ),
                ],
            ),
        )
        app = App(app_pb2)
        assert app.deployment_status == app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE
