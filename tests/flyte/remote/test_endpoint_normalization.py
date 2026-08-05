import pytest

from flyte.remote._client.auth._session import normalize_rpc_endpoint


class TestNormalizeRpcEndpoint:
    def test_dns_scheme_secure(self):
        assert normalize_rpc_endpoint("dns:///example.com:8089", insecure=False) == "https://example.com:8089"

    def test_dns_scheme_insecure(self):
        assert normalize_rpc_endpoint("dns:///example.com:8089", insecure=True) == "http://example.com:8089"

    def test_https_preserved(self):
        assert normalize_rpc_endpoint("https://example.com:443", insecure=False) == "https://example.com:443"

    def test_http_preserved(self):
        assert normalize_rpc_endpoint("http://localhost:8080", insecure=True) == "http://localhost:8080"

    def test_bare_host_port_secure(self):
        assert normalize_rpc_endpoint("example.com:8089", insecure=False) == "https://example.com:8089"

    def test_bare_host_port_insecure(self):
        assert normalize_rpc_endpoint("example.com:8089", insecure=True) == "http://example.com:8089"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown scheme"):
            normalize_rpc_endpoint("ftp://example.com", insecure=False)

    def test_preserves_localhost_port(self):
        """Must NOT rewrite localhost to port 8080 like Console does."""
        assert normalize_rpc_endpoint("dns:///localhost:8089", insecure=True) == "http://localhost:8089"
