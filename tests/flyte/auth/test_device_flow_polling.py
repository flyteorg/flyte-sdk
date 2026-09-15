"""Device authorization grant polling behaviour (RFC 8628)."""

import time

import httpx
import pytest

from flyte.remote._client.auth import _token_client as token_client
from flyte.remote._client.auth.errors import AuthenticationError


def _device_code(**overrides) -> token_client.DeviceCodeResponse:
    payload = {
        "device_code": "device-code",
        "user_code": "BCDF-GHJK",
        "verification_uri": "https://login.example.com/device",
        "expires_in": 600,
        "interval": 1,
    }
    payload.update(overrides)
    return token_client.DeviceCodeResponse(**payload)


def _session(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _poll(session, resp=None):
    return await token_client.poll_token_endpoint(
        resp or _device_code(),
        token_endpoint="https://login.example.com/token",
        client_id="client",
        http_session=session,
    )


class TestDeviceCodeResponse:
    def test_interval_is_optional_and_defaults_to_five(self):
        """RFC 8628 3.2: interval is OPTIONAL; clients MUST default to 5."""
        resp = token_client.DeviceCodeResponse.from_json_response(
            {
                "device_code": "d",
                "user_code": "U",
                "verification_uri": "https://login.example.com/device",
                "expires_in": 600,
            }
        )
        assert resp.interval == 5

    def test_verification_uri_complete_is_captured(self):
        resp = token_client.DeviceCodeResponse.from_json_response(
            {
                "device_code": "d",
                "user_code": "BCDF-GHJK",
                "verification_uri": "https://login.example.com/device",
                "verification_uri_complete": "https://login.example.com/device?user_code=BCDF-GHJK",
                "expires_in": 600,
                "interval": 5,
            }
        )
        assert resp.verification_uri_complete.endswith("user_code=BCDF-GHJK")


class TestPolling:
    @pytest.mark.asyncio
    async def test_slow_down_widens_the_interval(self):
        """RFC 8628 3.5: each slow_down MUST add 5s to the polling interval."""
        seen: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(time.monotonic())
            if len(seen) == 1:
                return httpx.Response(400, json={"error": "slow_down"})
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})

        await _poll(_session(handler))

        gap = seen[1] - seen[0]
        # started at 1s; one slow_down must take it to 6s
        assert gap >= 5.5, f"interval did not widen after slow_down (gap={gap:.2f}s)"

    @pytest.mark.asyncio
    async def test_authorization_pending_keeps_the_interval(self):
        seen: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(time.monotonic())
            if len(seen) == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})

        await _poll(_session(handler))

        gap = seen[1] - seen[0]
        assert gap < 3, f"pending should not widen the interval (gap={gap:.2f}s)"

    @pytest.mark.asyncio
    async def test_non_json_error_body_does_not_kill_the_poll(self):
        """A gateway answering HTML mid-poll must not abort the whole login."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(502, text="<html>502 Bad Gateway</html>", headers={"content-type": "text/html"})
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})

        with pytest.raises(AuthenticationError):
            # A 502 with no error code is still a hard failure, but it must be a
            # typed AuthenticationError rather than a JSONDecodeError escaping.
            await _poll(_session(handler))

    @pytest.mark.asyncio
    async def test_access_denied_reports_the_denial(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "access_denied"})

        # Match the explanation, not the raw error code echoed back in a
        # status dump, so this fails if the message regresses to that.
        with pytest.raises(AuthenticationError, match="denied on the other device"):
            await _poll(_session(handler))

    @pytest.mark.asyncio
    async def test_expired_token_reports_the_expiry(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "expired_token"})

        with pytest.raises(AuthenticationError, match="expired before the request was approved"):
            await _poll(_session(handler))

    @pytest.mark.asyncio
    async def test_polling_stops_once_the_code_expires(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "authorization_pending"})

        with pytest.raises(AuthenticationError, match="Timed out"):
            await _poll(_session(handler), _device_code(expires_in=1, interval=1))

    @pytest.mark.asyncio
    async def test_successful_poll_returns_the_tokens(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "a", "refresh_token": "r", "expires_in": 42})

        assert await _poll(_session(handler)) == ("a", "r", 42)
