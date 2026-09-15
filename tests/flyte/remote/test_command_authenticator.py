"""Tests for external-command auth token validation (FLYTE-SDK-6Y)."""

from unittest.mock import AsyncMock, patch

import pytest

from flyte.remote._client.auth._authenticators.base import Authenticator, is_usable_access_token
from flyte.remote._client.auth._authenticators.external_command import AsyncCommandAuthenticator
from flyte.remote._client.auth._keyring import Credentials
from flyte.remote._client.auth.errors import AuthenticationError

# The stdout that triggered FLYTE-SDK-6Y: an interactive helper printed its prompt on
# stdout and exited 0, so the prompt was used verbatim as the bearer token.
OAUTH_PROMPT = (
    "Please visit this URL to authorize this application: "
    "https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=1166.apps.googleusercontent.com"
)


@pytest.mark.parametrize(
    "token",
    [
        "abc123",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.sig",
        "ya29.a0-_dashes.and.dots~tilde",
    ],
)
def test_usable_tokens(token):
    assert is_usable_access_token(token)


@pytest.mark.parametrize(
    "token",
    [
        None,
        "",
        OAUTH_PROMPT,
        "token with spaces",
        "token\nwith-newline",
        "token\twith-tab",
        "token\rwith-cr",
        "token\x00with-nul",
        "token\x7fwith-del",
        "tökén-with-non-latin1-中",
    ],
)
def test_unusable_tokens(token):
    assert not is_usable_access_token(token)


def _authenticator(command):
    with patch("flyte.remote._client.auth._authenticators.base.KeyringStore.retrieve", return_value=None):
        return AsyncCommandAuthenticator(command=command, endpoint="dns:///fake.union.ai")


async def _run_command_auth(stdout: bytes, returncode: int = 0):
    auth = _authenticator(["fake-token-helper"])
    process = AsyncMock()
    process.communicate = AsyncMock(return_value=(stdout, b""))
    process.returncode = returncode
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        return await auth._do_refresh_credentials()


@pytest.mark.asyncio
async def test_command_returning_prompt_raises_authentication_error():
    """An interactive prompt on stdout must not become a bearer token."""
    with pytest.raises(AuthenticationError) as exc_info:
        await _run_command_auth(OAUTH_PROMPT.encode())

    message = str(exc_info.value)
    assert "did not return a usable access token" in message
    # The output may contain a real secret, so it must not be echoed back.
    assert "accounts.google.com" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", [b"", b"   \n", b"usage: helper [-h]\n  --flag\n"])
async def test_command_returning_junk_raises_authentication_error(stdout):
    with pytest.raises(AuthenticationError):
        await _run_command_auth(stdout)


@pytest.mark.asyncio
async def test_command_returning_token_succeeds():
    creds = await _run_command_auth(b"  a-real-token\n")
    assert creds.access_token == "a-real-token"


@pytest.mark.asyncio
async def test_nonzero_exit_keeps_its_specific_message():
    """The command-failed error must not be re-wrapped by the generic handler."""
    with pytest.raises(AuthenticationError) as exc_info:
        await _run_command_auth(b"", returncode=1)

    assert "Failed to refresh token" in str(exc_info.value)


class _StubAuthenticator(Authenticator):
    async def _do_refresh_credentials(self) -> Credentials:  # pragma: no cover - not exercised
        raise NotImplementedError


def _stub_with_token(token: str) -> _StubAuthenticator:
    creds = Credentials(for_endpoint="dns:///fake.union.ai", access_token=token)
    with patch("flyte.remote._client.auth._authenticators.base.KeyringStore.retrieve", return_value=creds):
        return _StubAuthenticator(endpoint="dns:///fake.union.ai")


@pytest.mark.asyncio
async def test_cached_malformed_token_is_ignored():
    """A poisoned keyring entry must not be injected as a header."""
    auth = _stub_with_token(OAUTH_PROMPT)
    assert await auth.get_auth_headers() is None


@pytest.mark.asyncio
async def test_cached_valid_token_is_used():
    auth = _stub_with_token("a-real-token")
    headers = await auth.get_auth_headers()
    assert headers is not None
    assert headers.headers["authorization"] == "Bearer a-real-token"
