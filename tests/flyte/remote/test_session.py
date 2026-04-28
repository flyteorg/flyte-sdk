import datetime
from unittest.mock import MagicMock, patch

import pyqwest
import pytest

from flyte.remote._client.auth._session import (
    SessionConfig,
    _bootstrap_ssl_from_server,
    _build_pyqwest_client,
    _resolve_tls_ca_cert,
    create_session_config,
)


def _make_valid_cert_pem() -> bytes:
    """Generate a valid self-signed PEM certificate for testing."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_insecure_skips_auth_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("localhost:8080", insecure=True)
    assert isinstance(result, SessionConfig)
    mock_auth.assert_not_called()
    assert result.endpoint == "http://localhost:8080"


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=["auth1"])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_secure_creates_auth_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("example.com:443", insecure=False)
    mock_auth.assert_called_once()
    assert "auth1" in result.interceptors
    assert result.endpoint == "https://example.com:443"


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_rpc_retries_creates_retry_interceptors(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("example.com:443", insecure=True, rpc_retries=3)
    from flyte.remote._client.auth._interceptors.retry import RetryServerStreamInterceptor, RetryUnaryInterceptor

    retry_unary = [i for i in result.interceptors if isinstance(i, RetryUnaryInterceptor)]
    retry_stream = [i for i in result.interceptors if isinstance(i, RetryServerStreamInterceptor)]
    assert len(retry_unary) == 1
    assert len(retry_stream) == 1
    assert retry_unary[0]._max_attempts == 4  # rpc_retries + 1


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_returns_session_config(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("dns:///example.com:8089", insecure=False)
    assert isinstance(result, SessionConfig)
    assert result.endpoint == "https://example.com:8089"
    assert isinstance(result.http_client, pyqwest.Client)
    assert len(result.interceptors) >= 1  # At least DefaultMetadataInterceptor


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
@patch("flyte.remote._client.auth._session._resolve_tls_ca_cert")
@patch("flyte.remote._client.auth._session._build_pyqwest_client")
async def test_custom_tls_creates_pyqwest_client(mock_build, mock_tls, mock_get_session, mock_proxy, mock_auth):
    mock_tls.return_value = b"cert-bytes"
    mock_client = MagicMock(spec=pyqwest.Client)
    mock_build.return_value = mock_client
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("example.com:443", insecure=False, ca_cert_file_path="/tmp/ca.pem")
    mock_build.assert_called_once_with(b"cert-bytes")
    assert result.http_client is mock_client


@pytest.mark.asyncio
@patch("flyte.remote._client.auth._session.create_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.create_proxy_auth_interceptors", return_value=[])
@patch("flyte.remote._client.auth._session.get_async_session")
async def test_insecure_no_tls_resolution(mock_get_session, mock_proxy, mock_auth):
    mock_get_session.return_value = MagicMock()
    result = await create_session_config("localhost:8080", insecure=True)
    assert isinstance(result.http_client, pyqwest.Client)


class TestResolveTlsCaCert:
    @pytest.mark.asyncio
    async def test_insecure_returns_none(self):
        result = await _resolve_tls_ca_cert(
            "http://localhost:8080", insecure=True, insecure_skip_verify=False, ca_cert_file_path=None
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("flyte.remote._client.auth._session._bootstrap_ssl_from_server", return_value=b"PEM-CERT")
    async def test_insecure_skip_verify_fetches_from_server(self, mock_bootstrap):
        result = await _resolve_tls_ca_cert(
            "https://example.com:443", insecure=False, insecure_skip_verify=True, ca_cert_file_path=None
        )
        assert result == b"PEM-CERT"
        mock_bootstrap.assert_called_once_with("https://example.com:443")

    @pytest.mark.asyncio
    async def test_ca_cert_file_reads_bytes(self, tmp_path):
        cert_content = b"-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n"
        cert_file = tmp_path / "ca.pem"
        cert_file.write_bytes(cert_content)
        result = await _resolve_tls_ca_cert(
            "https://example.com:443", insecure=False, insecure_skip_verify=False, ca_cert_file_path=str(cert_file)
        )
        assert result == cert_content

    @pytest.mark.asyncio
    async def test_no_tls_config_returns_none(self):
        result = await _resolve_tls_ca_cert(
            "https://example.com:443", insecure=False, insecure_skip_verify=False, ca_cert_file_path=None
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("flyte.remote._client.auth._session._bootstrap_ssl_from_server", return_value=b"PEM-CERT")
    async def test_insecure_skip_verify_takes_precedence_over_ca_cert(self, mock_bootstrap):
        """When both insecure_skip_verify and ca_cert_file_path are set, skip_verify wins."""
        result = await _resolve_tls_ca_cert(
            "https://example.com:443",
            insecure=False,
            insecure_skip_verify=True,
            ca_cert_file_path="/tmp/ca.pem",
        )
        assert result == b"PEM-CERT"
        mock_bootstrap.assert_called_once()


class TestBuildPyqwestClient:
    def test_no_cert_returns_client_with_defaults(self):
        client = _build_pyqwest_client(None)
        assert isinstance(client, pyqwest.Client)

    def test_with_valid_cert_returns_pyqwest_client(self):
        cert_pem = _make_valid_cert_pem()
        client = _build_pyqwest_client(cert_pem)
        assert isinstance(client, pyqwest.Client)


_SESSION_MOD = "flyte.remote._client.auth._session"


class TestBootstrapSslFromServer:
    def test_fetches_full_chain_and_encodes(self):
        """Should retrieve the full certificate chain via pyOpenSSL."""
        fake_cert_1 = MagicMock()
        fake_cert_2 = MagicMock()

        mock_conn = MagicMock()
        mock_conn.get_peer_cert_chain.return_value = [fake_cert_1, fake_cert_2]

        with (
            patch(f"{_SESSION_MOD}.SSL") as mock_ssl,
            patch(f"{_SESSION_MOD}.crypto") as mock_crypto,
            patch(f"{_SESSION_MOD}.socket") as mock_socket,
        ):
            mock_ssl.Connection.return_value = mock_conn
            mock_crypto.dump_certificate.side_effect = lambda fmt, cert: (
                b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n"
            )
            result = _bootstrap_ssl_from_server("https://example.com:8089")

        mock_socket.create_connection.assert_called_once_with(("example.com", 8089), timeout=10)
        mock_conn.do_handshake.assert_called_once()
        assert b"-----BEGIN CERTIFICATE-----" in result
        # Should contain both certs
        assert result.count(b"-----BEGIN CERTIFICATE-----") == 2

    def test_defaults_to_port_443(self):
        mock_conn = MagicMock()
        mock_conn.get_peer_cert_chain.return_value = [MagicMock()]

        with (
            patch(f"{_SESSION_MOD}.SSL") as mock_ssl,
            patch(f"{_SESSION_MOD}.crypto") as mock_crypto,
            patch(f"{_SESSION_MOD}.socket") as mock_socket,
        ):
            mock_ssl.Connection.return_value = mock_conn
            mock_crypto.dump_certificate.return_value = (
                b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n"
            )
            _bootstrap_ssl_from_server("https://example.com")

        mock_socket.create_connection.assert_called_once_with(("example.com", 443), timeout=10)

    def test_raises_on_empty_cert_chain(self):
        """RuntimeError when server returns no certificates."""
        mock_conn = MagicMock()
        mock_conn.get_peer_cert_chain.return_value = []

        with (
            patch(f"{_SESSION_MOD}.SSL") as mock_ssl,
            patch(f"{_SESSION_MOD}.crypto"),
            patch(f"{_SESSION_MOD}.socket"),
        ):
            mock_ssl.Connection.return_value = mock_conn
            with pytest.raises(RuntimeError, match="returned no certificates"):
                _bootstrap_ssl_from_server("https://example.com:443")

        mock_conn.close.assert_called_once()

    def test_closes_conn_on_success(self):
        """conn.close() is called after successful chain retrieval."""
        mock_conn = MagicMock()
        mock_conn.get_peer_cert_chain.return_value = [MagicMock()]
        mock_sock = MagicMock()

        with (
            patch(f"{_SESSION_MOD}.SSL") as mock_ssl,
            patch(f"{_SESSION_MOD}.crypto") as mock_crypto,
            patch(f"{_SESSION_MOD}.socket") as mock_socket,
        ):
            mock_socket.create_connection.return_value = mock_sock
            mock_ssl.Connection.return_value = mock_conn
            mock_crypto.dump_certificate.return_value = (
                b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n"
            )
            _bootstrap_ssl_from_server("https://example.com:443")

        mock_sock.settimeout.assert_called_once_with(None)
        mock_conn.close.assert_called_once()

    def test_closes_socket_if_connection_setup_fails(self):
        """Socket is closed if SSL.Connection() raises before conn is set."""
        mock_sock = MagicMock()

        with (
            patch(f"{_SESSION_MOD}.SSL") as mock_ssl,
            patch(f"{_SESSION_MOD}.crypto"),
            patch(f"{_SESSION_MOD}.socket") as mock_socket,
        ):
            mock_socket.create_connection.return_value = mock_sock
            mock_ssl.Connection.side_effect = RuntimeError("SSL init failed")
            with pytest.raises(RuntimeError, match="SSL init failed"):
                _bootstrap_ssl_from_server("https://example.com:443")

        mock_sock.close.assert_called_once()


class TestClientSetSessionConfig:
    def test_exposes_session_config(self):
        from flyte.remote._client.controlplane import ClientSet

        cs = ClientSet(
            SessionConfig(
                endpoint="https://example.com",
                insecure=False,
                insecure_skip_verify=False,
                interceptors=("a",),
                http_client="fake",
            )
        )
        cfg = cs.session_config
        assert isinstance(cfg, SessionConfig)
        assert cfg.endpoint == "https://example.com"
        assert cfg.interceptors == ("a",)
        assert cfg.http_client == "fake"
