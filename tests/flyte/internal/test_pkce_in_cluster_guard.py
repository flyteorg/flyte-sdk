import pytest

from flyte.remote._client.auth._authenticators.pkce import AuthorizationClient
from flyte.remote._client.auth.errors import AuthenticationError


@pytest.mark.asyncio
async def test_pkce_refuses_to_run_in_cluster(monkeypatch):
    """In-cluster task pods must fail fast instead of binding the OAuth callback port."""
    monkeypatch.setenv("ACTION_NAME", "a0")
    client = AuthorizationClient.__new__(AuthorizationClient)  # skip __init__: the guard fires first
    with pytest.raises(AuthenticationError, match="cannot run in a cluster pod"):
        await client.get_creds_from_remote()
