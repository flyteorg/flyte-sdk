"""Tests for how the OAuth token/device-code endpoints handle a body that is not JSON.

Both endpoints are specified to answer in JSON, but the SDK talks to them through whatever a
deployment puts in the way: a load balancer's HTML 502 page, a proxy's plain-text "Internal
Server Error", an SSO interstitial. Reading such a body used to raise `json.JSONDecodeError`
straight out of the error branch that was about to raise a perfectly good `AuthenticationError`
(FLYTE-SDK-60).
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from flyte.remote._client.auth._token_client import (
    GrantType,
    _body_snippet,
    _json_object_or_none,
    get_device_code,
    get_token,
)
from flyte.remote._client.auth.errors import AuthenticationError, AuthenticationPending

# What a proxy in front of a broken IDP actually returns -- the FLYTE-SDK-60 event.
NGINX_502 = "<html><head><title>502 Bad Gateway</title></head><body>502 Bad Gateway</body></html>"


def _session(response: httpx.Response) -> MagicMock:
    session = MagicMock()
    session.post = AsyncMock(return_value=response)
    return session


class TestJsonObjectOrNone:
    @pytest.mark.parametrize(
        "response, expected",
        [
            (httpx.Response(200, json={"access_token": "t"}), {"access_token": "t"}),
            (httpx.Response(500, text=NGINX_502), None),
            (httpx.Response(500, text="Internal Server Error"), None),
            (httpx.Response(200, text=""), None),
            # Valid JSON that is not an object: `"error" in j` would answer by substring on a
            # bare string, which is not the membership test the caller means.
            (httpx.Response(400, json="error"), None),
            (httpx.Response(400, json=["error"]), None),
        ],
    )
    def test_only_json_objects_survive(self, response, expected):
        assert _json_object_or_none(response) == expected


class TestBodySnippet:
    def test_reports_content_type_and_body(self):
        snippet = _body_snippet(httpx.Response(500, text="boom", headers={"content-type": "text/plain"}))
        assert "text/plain" in snippet
        assert "boom" in snippet

    def test_empty_body_is_said_to_be_empty(self):
        assert "empty body" in _body_snippet(httpx.Response(500, text=""))

    def test_long_body_is_truncated(self):
        snippet = _body_snippet(httpx.Response(500, text="x" * 5000), limit=50)
        assert len(snippet) < 200
        assert snippet.endswith("...'")


class TestGetTokenNonJsonBody:
    @pytest.mark.asyncio
    async def test_error_status_with_html_body_reports_the_status(self):
        """The failure branch already had the right error to raise; it just had to get there."""
        with pytest.raises(AuthenticationError) as excinfo:
            await get_token("https://idp.example.com/token", _session(httpx.Response(500, text=NGINX_502)))

        assert "Status Code (500)" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_error_status_with_empty_body_reports_the_status(self):
        with pytest.raises(AuthenticationError) as excinfo:
            await get_token("https://idp.example.com/token", _session(httpx.Response(503, text="")))

        assert "Status Code (503)" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_success_status_with_non_json_body_names_the_endpoint(self):
        """A 200 carrying a login page is not an SDK bug, and must not read as one."""
        with pytest.raises(AuthenticationError) as excinfo:
            await get_token("https://idp.example.com/token", _session(httpx.Response(200, text=NGINX_502)))

        message = str(excinfo.value)
        assert "https://idp.example.com/token" in message
        assert "not an access token" in message

    @pytest.mark.asyncio
    async def test_success_status_without_access_token_is_reported(self):
        with pytest.raises(AuthenticationError) as excinfo:
            await get_token("https://idp.example.com/token", _session(httpx.Response(200, json={"scope": "all"})))

        assert "not an access token" in str(excinfo.value)


class TestGetTokenStillWorks:
    @pytest.mark.asyncio
    async def test_access_token_is_returned(self):
        response = httpx.Response(200, json={"access_token": "abc", "refresh_token": "r", "expires_in": 3600})

        access, refresh, expires = await get_token("https://idp.example.com/token", _session(response))

        assert (access, refresh, expires) == ("abc", "r", 3600)

    @pytest.mark.asyncio
    async def test_missing_refresh_token_is_fine(self):
        response = httpx.Response(200, json={"access_token": "abc", "expires_in": 3600})

        access, refresh, expires = await get_token("https://idp.example.com/token", _session(response))

        assert (access, refresh, expires) == ("abc", None, 3600)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("err", ["authorization_pending", "slow_down"])
    async def test_device_flow_pending_still_raises_authentication_pending(self, err):
        """The JSON error branch is the one that keeps the device-code poll loop alive."""
        response = httpx.Response(400, json={"error": err})

        with pytest.raises(AuthenticationPending):
            await get_token(
                "https://idp.example.com/token",
                _session(response),
                grant_type=GrantType.DEVICE_CODE,
                device_code="dc",
            )


class TestGetDeviceCodeNonJsonBody:
    @pytest.mark.asyncio
    async def test_error_status_does_not_crash_building_its_own_message(self):
        """`Reason {resp.json()}` was interpolated into the error it was raising."""
        with pytest.raises(AuthenticationError) as excinfo:
            await get_device_code(
                "https://idp.example.com/device", "client", _session(httpx.Response(502, text=NGINX_502))
            )

        assert "Status Code 502" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_success_status_with_non_json_body_names_the_endpoint(self):
        with pytest.raises(AuthenticationError) as excinfo:
            await get_device_code(
                "https://idp.example.com/device", "client", _session(httpx.Response(200, text=NGINX_502))
            )

        assert "https://idp.example.com/device" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_valid_device_code_response_is_parsed(self):
        response = httpx.Response(
            200,
            json={
                "device_code": "dc",
                "user_code": "UC",
                "verification_uri": "https://idp.example.com/activate",
                "expires_in": 600,
                "interval": 5,
            },
        )

        result = await get_device_code("https://idp.example.com/device", "client", _session(response))

        assert result.device_code == "dc"
        assert result.interval == 5
