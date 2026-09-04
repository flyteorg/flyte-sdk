"""Tests for env-var-driven auth in `init_in_cluster`.

A deployment that does not issue eager API keys to task pods can instead set the
standard credentials config env vars as default env vars on the pod:

    FLYTE_ADMIN_AUTHTYPE=ExternalCommand
    FLYTE_ADMIN_COMMAND="/usr/local/bin/mint-token --audience flyte"

The explicit auth type must win over any injected `EAGER_API_KEY` /
`_UNION_EAGER_API_KEY`, and the resulting kwargs must reach BOTH the SDK client
(via `init`) and the controller (via `create_remote_controller`) — the controller
is what enqueues and watches child actions.
"""

from unittest.mock import AsyncMock, patch

import pytest

from flyte._initialize import _auth_overrides_from_env, init_in_cluster
from flyte.errors import InitializationError

AUTH_ENV_VARS = (
    "FLYTE_ADMIN_AUTHTYPE",
    "FLYTE_ADMIN_COMMAND",
    "FLYTE_ADMIN_PROXYCOMMAND",
    "FLYTE_ADMIN_ENDPOINT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Start from a pod with no config file, no api key and no auth env vars."""
    for var in (
        "UCTL_CONFIG",
        "FLYTECTL_CONFIG",
        "_UNION_EAGER_API_KEY",
        "EAGER_API_KEY",
        "_U_EP_OVERRIDE",
        *AUTH_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestAuthOverridesFromEnv:
    def test_empty_when_nothing_set(self, clean_env):
        assert _auth_overrides_from_env() == {}

    def test_command_is_shell_split(self, clean_env):
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "/usr/local/bin/mint-token --audience flyte")

        assert _auth_overrides_from_env() == {
            "auth_type": "ExternalCommand",
            "command": ["/usr/local/bin/mint-token", "--audience", "flyte"],
        }

    def test_command_accepts_json_array(self, clean_env):
        """A JSON array is the unambiguous form for arguments containing spaces."""
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", '["mint-token", "--claim", "team = data"]')

        assert _auth_overrides_from_env()["command"] == ["mint-token", "--claim", "team = data"]

    def test_quoted_argument_survives_shell_split(self, clean_env):
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "mint-token --claim 'team = data'")

        assert _auth_overrides_from_env()["command"] == ["mint-token", "--claim", "team = data"]

    def test_proxy_command_is_picked_up(self, clean_env):
        clean_env.setenv("FLYTE_ADMIN_PROXYCOMMAND", "get-proxy-token --quiet")

        assert _auth_overrides_from_env() == {"proxy_command": ["get-proxy-token", "--quiet"]}

    def test_external_command_without_command_is_loud(self, clean_env):
        """Otherwise this only surfaces as an AuthenticationError from the auth
        interceptor on the first RPC, naming neither env var."""
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")

        with pytest.raises(InitializationError, match="FLYTE_ADMIN_COMMAND"):
            _auth_overrides_from_env()


class TestInitInClusterEnvAuth:
    @pytest.mark.asyncio
    async def test_external_command_env_vars_reach_client_and_controller(self, clean_env):
        clean_env.setenv("_U_EP_OVERRIDE", "dns:///example.com:443")
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "mint-token --audience flyte")

        with patch("flyte._initialize.init", new_callable=AsyncMock) as mock_init:
            controller_kwargs = await init_in_cluster.aio()

        # Returned kwargs are spread into create_remote_controller by the runtime.
        assert controller_kwargs["auth_type"] == "ExternalCommand"
        assert controller_kwargs["command"] == ["mint-token", "--audience", "flyte"]
        assert controller_kwargs["endpoint"] == "dns:///example.com:443"
        assert "api_key" not in controller_kwargs

        # ...and the same settings configure the SDK client.
        init_kwargs = mock_init.aio.await_args.kwargs
        assert init_kwargs["auth_type"] == "ExternalCommand"
        assert init_kwargs["command"] == ["mint-token", "--audience", "flyte"]

    @pytest.mark.asyncio
    async def test_explicit_auth_type_beats_injected_api_key(self, clean_env):
        """The whole point of the knob: a cluster that still injects an eager API
        key must not silently keep using it once ExternalCommand is configured."""
        clean_env.setenv("_U_EP_OVERRIDE", "dns:///example.com:443")
        clean_env.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")
        clean_env.setenv("EAGER_API_KEY", "another-token")
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "mint-token")

        with patch("flyte._initialize.init", new_callable=AsyncMock) as mock_init:
            controller_kwargs = await init_in_cluster.aio()

        assert "api_key" not in controller_kwargs
        assert mock_init.aio.await_args.kwargs.get("api_key") is None

    @pytest.mark.asyncio
    async def test_endpoint_falls_back_to_config_env_var(self, clean_env):
        """Deployments that drop the api key also lose the endpoint it decodes to.
        `_U_EP_OVERRIDE` is what the backend injects; `FLYTE_ADMIN_ENDPOINT` is the
        user-settable equivalent for the same reason the auth vars are."""
        clean_env.setenv("FLYTE_ADMIN_ENDPOINT", "dns:///byo.example.com:443")
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "mint-token")

        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()

        assert controller_kwargs["endpoint"] == "dns:///byo.example.com:443"

    @pytest.mark.asyncio
    async def test_injected_api_key_still_wins_when_no_auth_type_set(self, clean_env):
        """No auth env vars -> unchanged legacy behavior."""
        clean_env.setenv("_UNION_EAGER_API_KEY", "legacy-composite-token")

        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio()

        assert controller_kwargs["api_key"] == "legacy-composite-token"
        assert "auth_type" not in controller_kwargs

    @pytest.mark.asyncio
    async def test_explicit_api_key_arg_wins_over_env_auth(self, clean_env):
        clean_env.setenv("FLYTE_ADMIN_AUTHTYPE", "ExternalCommand")
        clean_env.setenv("FLYTE_ADMIN_COMMAND", "mint-token")

        with patch("flyte._initialize.init", new_callable=AsyncMock):
            controller_kwargs = await init_in_cluster.aio(api_key="explicit-key")

        assert controller_kwargs["api_key"] == "explicit-key"
        assert "auth_type" not in controller_kwargs


class TestControllerForwardsAuth:
    @pytest.mark.asyncio
    async def test_create_remote_controller_forwards_external_command(self):
        """The controller is the half of in-cluster init that enqueues and watches
        child actions. Dropping `command` here left `auth_type=ExternalCommand`
        with nothing to run, which only surfaced as an AuthenticationError on the
        controller's first RPC."""
        from flyte._internal.controllers.remote import create_remote_controller

        captured: dict = {}

        async def fake_for_endpoint(endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured.update(kwargs)
            return object()

        with (
            patch(
                "flyte._internal.controllers.remote._client.ControllerClient.for_endpoint",
                staticmethod(fake_for_endpoint),
            ),
            patch("flyte._internal.controllers.remote._controller.RemoteController") as mock_controller,
        ):
            create_remote_controller(
                endpoint="dns:///example.com:443",
                auth_type="ExternalCommand",
                command=["mint-token", "--audience", "flyte"],
            )
            # Awaiting the captured coroutine is what actually builds the client.
            await mock_controller.call_args.kwargs["client_coro"]

        assert captured["endpoint"] == "dns:///example.com:443"
        assert captured["auth_type"] == "ExternalCommand"
        assert captured["command"] == ["mint-token", "--audience", "flyte"]
