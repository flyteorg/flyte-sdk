import unittest
from unittest import mock

import grpc
import grpc.aio

from flyte.remote._client.auth._channel import create_channel


class TestCreateChannel(unittest.TestCase):
    """Test cases for the create_channel function in the auth module."""

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("grpc.aio.insecure_channel")
    async def test_create_insecure_channel(self, mock_insecure_channel, mock_proxy_auth, mock_auth):
        """Test creating an insecure channel."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_insecure_channel.return_value = mock_channel
        mock_proxy_auth.return_value = None
        mock_auth.return_value = None

        # Call the function
        endpoint = "localhost:8080"
        result = create_channel(endpoint=endpoint, insecure=True)

        # Verify results
        mock_insecure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("grpc.aio.secure_channel")
    @mock.patch("grpc.ssl_channel_credentials")
    async def test_create_secure_channel(self, mock_ssl_creds, mock_secure_channel, mock_proxy_auth, mock_auth):
        """Test creating a secure channel."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_secure_channel.return_value = mock_channel
        mock_creds = mock.Mock(spec=grpc.ChannelCredentials)
        mock_ssl_creds.return_value = mock_creds
        mock_proxy_auth.return_value = None
        mock_auth.return_value = None

        # Call the function
        endpoint = "localhost:8080"
        result = create_channel(endpoint=endpoint, insecure=False)

        # Verify results
        mock_ssl_creds.assert_called_once()
        mock_secure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("grpc.aio.secure_channel")
    async def test_create_channel_with_custom_ssl_credentials(self, mock_secure_channel, mock_proxy_auth, mock_auth):
        """Test creating a channel with custom SSL credentials."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_secure_channel.return_value = mock_channel
        mock_proxy_auth.return_value = None
        mock_auth.return_value = None
        mock_ssl_creds = mock.Mock(spec=grpc.ChannelCredentials)

        # Call the function
        endpoint = "localhost:8080"
        result = create_channel(endpoint=endpoint, insecure=False, ssl_credentials=mock_ssl_creds)

        # Verify results
        mock_secure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.bootstrap_creds_from_server")
    @mock.patch("grpc.aio.secure_channel")
    async def test_create_channel_with_insecure_skip_verify(
        self, mock_secure_channel, mock_bootstrap, mock_proxy_auth, mock_auth
    ):
        """Test creating a channel with insecure_skip_verify=True."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_secure_channel.return_value = mock_channel
        mock_creds = mock.Mock(spec=grpc.ChannelCredentials)
        mock_bootstrap.return_value = mock_creds
        mock_proxy_auth.return_value = None
        mock_auth.return_value = None

        # Call the function
        endpoint = "localhost:8080"
        result = create_channel(endpoint=endpoint, insecure=False, insecure_skip_verify=True)

        # Verify results
        mock_bootstrap.assert_called_once_with(endpoint)
        mock_secure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"test-cert")
    @mock.patch("grpc.ssl_channel_credentials")
    @mock.patch("grpc.aio.secure_channel")
    async def test_create_channel_with_ca_cert_file(
        self, mock_secure_channel, mock_ssl_creds, mock_open, mock_proxy_auth, mock_auth
    ):
        """Test creating a channel with a CA certificate file."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_secure_channel.return_value = mock_channel
        mock_creds = mock.Mock(spec=grpc.ChannelCredentials)
        mock_ssl_creds.return_value = mock_creds
        mock_proxy_auth.return_value = None
        mock_auth.return_value = None

        # Call the function
        endpoint = "localhost:8080"
        ca_cert_path = "/path/to/ca.crt"
        result = create_channel(endpoint=endpoint, insecure=False, ca_cert_file_path=ca_cert_path)

        # Verify results
        mock_open.assert_called_once_with(ca_cert_path, "rb")
        mock_ssl_creds.assert_called_once_with(b"test-cert")
        mock_secure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

    @mock.patch("flyte.remote.client._auth._async_utils.create_auth_interceptor")
    @mock.patch("flyte.remote.client._auth._async_utils.create_proxy_auth_interceptor")
    @mock.patch("grpc.aio.secure_channel")
    @mock.patch("grpc.aio.insecure_channel")
    async def test_create_channel_with_interceptors(
        self, mock_insecure_channel, mock_secure_channel, mock_proxy_auth, mock_auth
    ):
        """Test creating a channel with auth interceptors."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_insecure_channel.return_value = mock_channel
        mock_secure_channel.return_value = mock_channel

        mock_proxy_interceptor = mock.Mock(spec=grpc.aio.ClientInterceptor)
        mock_proxy_auth.return_value = mock_proxy_interceptor

        mock_auth_interceptor = mock.Mock(spec=grpc.aio.ClientInterceptor)
        mock_auth.return_value = mock_auth_interceptor

        # Call the function
        endpoint = "localhost:8080"
        result = create_channel(endpoint=endpoint, insecure=True)

        # Verify results
        mock_insecure_channel.assert_called_once()
        self.assertEqual(result, mock_channel)

        # Verify interceptors were added
        args, kwargs = mock_insecure_channel.call_args
        self.assertIn("interceptors", kwargs)
        self.assertIn(mock_proxy_interceptor, kwargs["interceptors"])
        self.assertIn(mock_auth_interceptor, kwargs["interceptors"])
