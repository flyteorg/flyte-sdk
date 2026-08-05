from unittest.mock import MagicMock, patch

from flyteidl2.app import app_definition_pb2

from flyte.app import AppEnvironment
from flyte.app._deploy import DeployedAppEnvironment
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
