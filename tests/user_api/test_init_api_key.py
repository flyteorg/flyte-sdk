from unittest.mock import AsyncMock, patch

import pytest

from flyte import _initialize as init_module
from flyte._initialize import init_from_api_key
from flyte.errors import InitializationError


class TestInitFromApiKey:
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        init_module._init_config = None
        yield
        init_module._init_config = None

    @pytest.mark.asyncio
    async def test_init_from_api_key_no_key_no_env(self, monkeypatch):
        monkeypatch.delenv("FLYTE_API_KEY", raising=False)
        with pytest.raises(InitializationError, match="API key must be provided"):
            await init_from_api_key.aio()

    @patch("flyte._initialize.init")
    @patch("flyte.remote._client.auth._auth_utils.decode_api_key")
    @patch("flyte._utils.sanitize_endpoint")
    @pytest.mark.asyncio
    async def test_init_from_api_key_with_key(self, mock_sanitize, mock_decode, mock_init):
        mock_decode.return_value = ("test.endpoint.com", "client-id", "client-secret", "my-org")
        mock_sanitize.return_value = "https://test.endpoint.com"
        mock_init.aio = AsyncMock()

        await init_from_api_key.aio(api_key="encoded-key", project="proj", domain="dev")

        mock_decode.assert_called_once_with("encoded-key")
        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["project"] == "proj"
        assert call_kwargs["domain"] == "dev"
        assert call_kwargs["auth_type"] == "ClientSecret"

    @patch("flyte._initialize.init")
    @patch("flyte.remote._client.auth._auth_utils.decode_api_key")
    @patch("flyte._utils.sanitize_endpoint")
    @pytest.mark.asyncio
    async def test_init_from_api_key_reads_env(self, mock_sanitize, mock_decode, mock_init, monkeypatch):
        monkeypatch.setenv("FLYTE_API_KEY", "env-key")
        mock_decode.return_value = ("endpoint.com", "cid", "csecret", "org")
        mock_sanitize.return_value = "https://endpoint.com"
        mock_init.aio = AsyncMock()

        await init_from_api_key.aio()

        mock_decode.assert_called_once_with("env-key")


class TestInitPassthrough:
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        init_module._init_config = None
        yield
        init_module._init_config = None

    @patch("flyte._initialize.init")
    @pytest.mark.asyncio
    async def test_init_passthrough_basic(self, mock_init):
        mock_init.aio = AsyncMock()
        from flyte._initialize import init_passthrough

        result = await init_passthrough.aio(endpoint="my.endpoint.com", project="proj", domain="dev")

        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["auth_type"] == "Passthrough"
        assert call_kwargs["endpoint"] == "my.endpoint.com"
        assert result["endpoint"] == "my.endpoint.com"

    @patch("flyte._initialize.init")
    @pytest.mark.asyncio
    async def test_init_passthrough_reads_env_endpoint(self, mock_init, monkeypatch):
        mock_init.aio = AsyncMock()
        monkeypatch.setenv("_U_EP_OVERRIDE", "env-endpoint:8080")
        from flyte._initialize import init_passthrough

        await init_passthrough.aio()

        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["endpoint"] == "env-endpoint:8080"
