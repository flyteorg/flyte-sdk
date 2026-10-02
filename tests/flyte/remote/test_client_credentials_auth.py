import pytest

from flyte.errors import InitializationError
from flyte.remote._client.auth._authenticators.client_credentials import ClientCredentialsAuthenticator


@pytest.mark.parametrize(
    "client_id,secret",
    [
        (None, None),
        ("", ""),
        ("client", None),
        ("client", ""),
        (None, "secret"),
        ("", "secret"),
    ],
)
def test_client_credentials_missing_creds_raises_initialization_error(client_id, secret):
    """FLYTE-SDK-55: missing client_id / secret is a creds-config mistake. The
    authenticator must raise a typed InitializationError (filtered from Sentry)
    rather than a bare ValueError that leaks as RuntimeSystemError('Failed to get
    signed url...')."""
    with pytest.raises(InitializationError) as exc_info:
        ClientCredentialsAuthenticator(
            client_id=client_id,
            client_credentials_secret=secret,
            endpoint="dns:///example.com",
        )

    assert "client_id and client_credentials_secret are required" in str(exc_info.value)
