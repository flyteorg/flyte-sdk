from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.app import app_definition_pb2

from flyte.app import AppEnvironment
from flyte.app._deploy import DeployedAppEnvironment, _deploy_app
from flyte.models import SerializationContext
from flyte.remote import App


def test_deployed_app_environment_table_repr_uses_public_and_console_urls():
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
            desired_state=app_definition_pb2.Spec.DESIRED_STATE_ACTIVE,
        ),
        status=app_definition_pb2.Status(
            ingress=app_definition_pb2.Ingress(
                public_url="https://public.example.com/app",
            ),
        ),
    )
    deployed_app = App(app_pb2)
    app_env = AppEnvironment(name="test-app", image="auto")
    mock_client = MagicMock()
    mock_client.console.app_url.return_value = "https://console.example.com/apps/test-app"

    with patch("flyte.remote._app.get_client", return_value=mock_client):
        row = dict(DeployedAppEnvironment(env=app_env, deployed_app=deployed_app).table_repr()[0])

    assert row["public_url"] == "[link=https://public.example.com/app]https://public.example.com/app[/link]"
    assert row["console_url"] == (
        "[link=https://console.example.com/apps/test-app]https://console.example.com/apps/test-app[/link]"
    )


@pytest.mark.asyncio
async def test_deploy_app_dryrun_returns_app_wrapper():
    """Regression for FLYTE-SDK-5W: dryrun must return an App, not the raw IDL proto.

    Returning the bare proto made DeployedAppEnvironment.table_repr() raise
    `AttributeError: name` because the proto has no top-level `.name` attribute.
    """
    app_pb2 = app_definition_pb2.App(
        metadata=app_definition_pb2.Meta(
            id=app_definition_pb2.Identifier(
                org="o",
                project="p",
                domain="d",
                name="dryrun-app",
            ),
        ),
        spec=app_definition_pb2.Spec(desired_state=app_definition_pb2.Spec.DESIRED_STATE_ACTIVE),
    )
    app_env = AppEnvironment(name="dryrun-app", image="auto")
    sc = MagicMock(spec=SerializationContext)
    sc.code_bundle = None
    sc.image_cache = None

    translate = MagicMock()
    translate.aio = AsyncMock(return_value=app_pb2)
    with patch("flyte.app._runtime.translate_app_env_to_idl", translate):
        result = await _deploy_app(app_env, sc, dryrun=True)

    assert isinstance(result, App)
    assert result.name == "dryrun-app"

    # table_repr must not raise AttributeError on the dryrun result.
    mock_client = MagicMock()
    mock_client.console.app_url.return_value = "https://console.example.com/apps/dryrun-app"
    with patch("flyte.remote._app.get_client", return_value=mock_client):
        row = dict(DeployedAppEnvironment(env=app_env, deployed_app=result).table_repr()[0])

    assert row["name"] == "dryrun-app"
