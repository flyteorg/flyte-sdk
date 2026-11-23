"""A client fetches its first token once, however many calls start before it arrives."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from flyte.remote._client.auth._authenticators import factory
from flyte.remote._client.auth._authenticators.base import Authenticator
from flyte.remote._client.auth._keyring import Credentials


class SlowFetch(Authenticator):
    """A token fetch that takes a while, as an external command does on a small pod."""

    def __init__(self):
        super().__init__("dns:///localhost", disable_keyring=True, http_session=Mock())
        self.fetches = 0

    async def _do_refresh_credentials(self) -> Credentials:
        self.fetches += 1
        await asyncio.sleep(0.05)
        return Credentials(for_endpoint=self._endpoint, access_token=f"token-{self.fetches}")


@pytest.mark.asyncio
async def test_calls_that_start_without_a_token_share_one_fetch():
    """Before: each call that found no token fetched its own, one after the other."""
    authenticator = SlowFetch()

    await asyncio.gather(authenticator.refresh_credentials(), authenticator.refresh_credentials())

    assert authenticator.fetches == 1
    assert authenticator.get_credentials().access_token == "token-1"


@pytest.mark.asyncio
async def test_a_forced_refresh_on_its_own_still_fetches():
    authenticator = SlowFetch()

    await authenticator.refresh_credentials()
    await authenticator.refresh_credentials()

    assert authenticator.fetches == 2


@pytest.mark.parametrize(
    "create",
    [
        lambda: factory.create_auth_interceptors("dns:///localhost"),
        lambda: factory.create_proxy_auth_interceptors("dns:///localhost", proxy_command=["token"]),
    ],
)
def test_unary_and_streaming_interceptors_share_one_authenticator(create):
    """Before: each interceptor built its own authenticator, so a watch and a unary call each fetched a token."""
    with (
        patch.object(factory, "get_async_authenticator", side_effect=lambda **_: Mock()) as build,
        patch.object(factory, "get_async_proxy_authenticator", side_effect=lambda **_: Mock()) as build_proxy,
    ):
        authenticators = {id(interceptor.authenticator) for interceptor in create()}

    assert len(authenticators) == 1
    assert build.call_count + build_proxy.call_count == 1
