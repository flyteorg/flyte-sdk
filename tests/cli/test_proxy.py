"""Unit tests for flyte.cli._proxy (the `flyte proxy app` command helpers)."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

from flyte.cli._proxy import (
    _emit_mcp_config,
    _filter_request_headers,
    _filter_response_headers,
    _identity,
)


def _jwt(claims: dict) -> str:
    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'none'})}.{b64(claims)}.signature"


class TestHeaderFiltering:
    def test_request_headers_drop_authorization_and_hop_by_hop(self):
        out = _filter_request_headers(
            {
                "Authorization": "Bearer inbound",
                "Host": "localhost:8600",
                "Connection": "keep-alive",
                "Content-Length": "10",
                "Accept": "*/*",
                "X-Custom": "keep",
            }
        )
        lowered = {k.lower() for k in out}
        assert "authorization" not in lowered  # inbound auth must never be forwarded
        assert "host" not in lowered
        assert "connection" not in lowered
        assert "content-length" not in lowered
        assert out["Accept"] == "*/*"
        assert out["X-Custom"] == "keep"

    def test_response_headers_drop_hop_by_hop_keep_others(self):
        out = _filter_response_headers(
            {"Transfer-Encoding": "chunked", "Content-Type": "application/json", "Content-Encoding": "gzip"}
        )
        lowered = {k.lower() for k in out}
        assert "transfer-encoding" not in lowered
        assert out["Content-Type"] == "application/json"
        assert out["Content-Encoding"] == "gzip"  # preserved for byte-verbatim streaming


class TestIdentity:
    def test_prefers_email_claim(self):
        auth = MagicMock()
        auth.get_credentials.return_value = MagicMock(access_token=_jwt({"email": "me@union.ai", "sub": "abc"}))
        assert _identity(auth) == "me@union.ai"

    def test_falls_back_to_sub(self):
        auth = MagicMock()
        auth.get_credentials.return_value = MagicMock(access_token=_jwt({"sub": "subject-123"}))
        assert _identity(auth) == "subject-123"

    def test_unknown_when_no_credentials(self):
        auth = MagicMock()
        auth.get_credentials.return_value = None
        assert _identity(auth) == "<unknown>"


class TestEmitMcpConfig:
    def test_emits_generic_http_block(self, capsys):
        _emit_mcp_config("grafana", "http://127.0.0.1:8600")
        block = json.loads(capsys.readouterr().out)
        assert block == {"mcpServers": {"grafana": {"type": "http", "url": "http://127.0.0.1:8600"}}}
