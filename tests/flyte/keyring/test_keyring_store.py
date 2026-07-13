"""Tests for `flyte.remote._client.auth._keyring.KeyringStore`.

These tests also guard the lazy-import invariant: simply importing `_keyring`
or calling it with ``disable=True`` must not pull the heavy ``keyring`` package
into ``sys.modules``. That invariant matters for cluster-runtime cold start
(see `keyring.backends.macOS.api` probing on Linux, ~130ms).
"""

import importlib
import sys
from unittest.mock import patch

import pytest


def _fresh_keyring_module():
    """Reload `_keyring` after scrubbing `keyring` from sys.modules.

    Returns the freshly imported `_keyring` module. The caller can then assert
    on whether `keyring` is present in ``sys.modules`` after various calls.
    """
    for name in list(sys.modules):
        if name == "keyring" or name.startswith("keyring."):
            del sys.modules[name]
    sys.modules.pop("flyte.remote._client.auth._keyring", None)
    return importlib.import_module("flyte.remote._client.auth._keyring")


def test_import_does_not_load_keyring():
    mod = _fresh_keyring_module()
    assert hasattr(mod, "Credentials")
    assert hasattr(mod, "KeyringStore")
    assert "keyring" not in sys.modules, (
        "Importing flyte.remote._client.auth._keyring must not pull `keyring`. "
        "That would add ~130ms to cluster cold start via keyring.backends.macOS.api."
    )


def test_disable_true_never_loads_keyring():
    mod = _fresh_keyring_module()
    creds = mod.Credentials(access_token="tok", for_endpoint="foo")

    mod.KeyringStore.store(creds, disable=True)
    mod.KeyringStore.retrieve("foo", disable=True)
    mod.KeyringStore.delete("foo", disable=True)

    assert "keyring" not in sys.modules, "disable=True must short-circuit before importing keyring."


def test_credentials_model_works_without_keyring():
    """The Credentials pydantic model is used everywhere; it must not depend on keyring."""
    mod = _fresh_keyring_module()
    creds = mod.Credentials(access_token="abc", for_endpoint="https://flyte.example.com")
    assert creds.for_endpoint == "flyte.example.com"  # scheme stripped
    assert creds.id  # md5 of access_token populated
    assert "keyring" not in sys.modules


def test_store_skips_when_disabled():
    from flyte.remote._client.auth._keyring import Credentials, KeyringStore

    creds = Credentials(access_token="tok", for_endpoint="foo")
    with patch("keyring.set_password") as mock_set:
        result = KeyringStore.store(creds, disable=True)
    mock_set.assert_not_called()
    assert result is creds


def test_store_writes_single_keychain_item():
    """One keychain item for both tokens = one macOS keychain prompt on retrieve."""
    import json

    from flyte.remote._client.auth._keyring import Credentials, KeyringStore

    creds = Credentials(access_token="tok", for_endpoint="foo", refresh_token="rtok")
    with patch("keyring.set_password") as mock_set:
        KeyringStore.store(creds, disable=False)

    assert mock_set.call_count == 1
    assert mock_set.call_args.args[1] == "tokens"
    assert json.loads(mock_set.call_args.args[2]) == {"access_token": "tok", "refresh_token": "rtok"}


def test_store_swallows_no_keyring_error():
    from keyring.errors import NoKeyringError

    from flyte.remote._client.auth._keyring import Credentials, KeyringStore

    creds = Credentials(access_token="tok", for_endpoint="foo")
    with patch("keyring.set_password", side_effect=NoKeyringError("no backend")):
        # Should not raise; keyring unavailability is non-fatal.
        result = KeyringStore.store(creds, disable=False)
    assert result is creds


def test_retrieve_skips_when_disabled():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.get_password") as mock_get:
        result = KeyringStore.retrieve("foo", disable=True)
    mock_get.assert_not_called()
    assert result is None


def test_retrieve_returns_none_when_no_tokens_stored():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.get_password", return_value=None) as mock_get:
        result = KeyringStore.retrieve("foo", disable=False)
    assert mock_get.called
    assert result is None


def test_retrieve_returns_credentials_when_access_token_present():
    import json

    from flyte.remote._client.auth._keyring import KeyringStore

    stored = json.dumps({"access_token": "a", "refresh_token": "r"})
    with patch("keyring.get_password", return_value=stored) as mock_get:
        creds = KeyringStore.retrieve("https://flyte.example.com", disable=False)

    # Exactly one keychain read = exactly one macOS keychain prompt.
    assert mock_get.call_count == 1
    assert creds is not None
    assert creds.access_token == "a"
    assert creds.refresh_token == "r"
    assert creds.for_endpoint == "flyte.example.com"  # scheme stripped


def test_retrieve_returns_none_on_unparseable_tokens():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.get_password", return_value="not-json"):
        assert KeyringStore.retrieve("foo", disable=False) is None


def test_retrieve_strips_scheme_before_lookup():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.get_password", return_value=None) as mock_get:
        KeyringStore.retrieve("https://flyte.example.com/path", disable=False)

    # Ensure the endpoint passed to keyring has no scheme.
    for call in mock_get.call_args_list:
        assert call.args[0] == "flyte.example.com/path"


def test_retrieve_handles_no_keyring_error():
    from keyring.errors import NoKeyringError

    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.get_password", side_effect=NoKeyringError("no backend")):
        result = KeyringStore.retrieve("foo", disable=False)
    assert result is None


def test_delete_skips_when_disabled():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.delete_password") as mock_del:
        KeyringStore.delete("foo", disable=True)
    mock_del.assert_not_called()


def test_delete_removes_tokens_and_legacy_keys_when_enabled():
    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.delete_password") as mock_del:
        KeyringStore.delete("https://flyte.example.com", disable=False)

    keys_deleted = {call.args[1] for call in mock_del.call_args_list}
    assert keys_deleted == {"tokens", "access_token", "refresh_token"}


def test_delete_swallows_password_delete_error():
    from keyring.errors import PasswordDeleteError

    from flyte.remote._client.auth._keyring import KeyringStore

    with patch("keyring.delete_password", side_effect=PasswordDeleteError("missing")):
        # Should not raise even if both deletes fail.
        KeyringStore.delete("foo", disable=False)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://flyte.example.com", "flyte.example.com"),
        ("https://flyte.example.com/path", "flyte.example.com/path"),
        ("dns:///flyte.example.com", "flyte.example.com"),
        ("flyte.example.com", "flyte.example.com"),  # no scheme, untouched
    ],
)
def test_strip_scheme(raw, expected):
    from flyte.remote._client.auth._keyring import strip_scheme

    assert strip_scheme(raw) == expected
