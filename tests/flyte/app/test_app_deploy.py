from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.app import app_definition_pb2
from flyteidl2.auth import identity_pb2

from flyte.app._deploy import _deploy_app
from flyte.models import SerializationContext


class TestDeployAppCreator:
    """Test that _deploy_app populates the creator field from the authenticated user."""

    @pytest.fixture
    def mock_serialization_context(self):
        ctx = MagicMock(spec=SerializationContext)
        ctx.org = "test-org"
        ctx.project = "test-project"
        ctx.domain = "test-domain"
        ctx.version = "abc123"
        ctx.code_bundle = None
        ctx.root_dir = None
        return ctx

    @pytest.fixture
    def mock_app_idl(self):
        return app_definition_pb2.App(
            metadata=app_definition_pb2.Meta(
                id=app_definition_pb2.Identifier(
                    org="test-org",
                    project="test-project",
                    domain="test-domain",
                    name="test-app",
                ),
            ),
            spec=app_definition_pb2.Spec(
                desired_state=app_definition_pb2.Spec.DESIRED_STATE_ACTIVE,
            ),
        )

    @pytest.fixture
    def mock_user_info_response(self):
        resp = identity_pb2.UserInfoResponse()
        resp.subject = "user-123"
        resp.name = "Test User"
        return resp

    @pytest.mark.asyncio
    async def test_deploy_app_sets_creator_from_user_info(
        self, mock_serialization_context, mock_app_idl, mock_user_info_response
    ):
        """Test that _deploy_app fetches user info and sets creator on the app spec."""
        mock_identity_service = AsyncMock()
        mock_identity_service.UserInfo = AsyncMock(return_value=mock_user_info_response)

        mock_client = MagicMock()
        mock_client.identity_service = mock_identity_service

        mock_created_app = MagicMock()

        mock_translate = MagicMock()
        mock_translate.aio = AsyncMock(return_value=mock_app_idl)

        mock_create = MagicMock()
        mock_create.aio = AsyncMock(return_value=mock_created_app)

        with (
            patch("flyte.app._deploy.ensure_client"),
            patch("flyte.app._deploy.get_client", return_value=mock_client),
            patch("flyte.app._runtime.translate_app_env_to_idl", mock_translate),
            patch("flyte.remote._app.App.create", mock_create),
            patch("flyte.app._deploy.status"),
        ):
            app_env = MagicMock()
            app_env.name = "test-app"
            app_env.image = "test-image:latest"
            app_env.include = None

            await _deploy_app(app_env, mock_serialization_context)

            # Verify creator was set from user info
            assert mock_app_idl.spec.creator.user.id.subject == "user-123"

            # Verify the app was created with the modified IDL
            mock_create.aio.assert_called_once_with(mock_app_idl)

    @pytest.mark.asyncio
    async def test_deploy_app_continues_if_user_info_fails(self, mock_serialization_context, mock_app_idl):
        """Test that _deploy_app continues even if fetching user info fails."""
        mock_identity_service = AsyncMock()
        mock_identity_service.UserInfo = AsyncMock(side_effect=Exception("auth error"))

        mock_client = MagicMock()
        mock_client.identity_service = mock_identity_service

        mock_created_app = MagicMock()

        mock_translate = MagicMock()
        mock_translate.aio = AsyncMock(return_value=mock_app_idl)

        mock_create = MagicMock()
        mock_create.aio = AsyncMock(return_value=mock_created_app)

        with (
            patch("flyte.app._deploy.ensure_client"),
            patch("flyte.app._deploy.get_client", return_value=mock_client),
            patch("flyte.app._runtime.translate_app_env_to_idl", mock_translate),
            patch("flyte.remote._app.App.create", mock_create),
            patch("flyte.app._deploy.status"),
            patch("flyte.app._deploy.logger") as mock_logger,
        ):
            app_env = MagicMock()
            app_env.name = "test-app"
            app_env.image = "test-image:latest"
            app_env.include = None

            await _deploy_app(app_env, mock_serialization_context)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "Could not fetch user identity" in mock_logger.warning.call_args[0][0]

            # Verify deploy still proceeded (creator field will be empty)
            mock_create.aio.assert_called_once_with(mock_app_idl)
