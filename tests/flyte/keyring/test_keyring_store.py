from unittest.mock import patch

from flyte.remote._client.auth._keyring import Credentials, KeyringStore


def test_store_skips_when_disabled():
    creds = Credentials(access_token="tok", for_endpoint="foo")
    with patch("flyte.remote._client.auth._keyring.keyring.set_password") as mock_set:
        result = KeyringStore.store(creds, disable=True)
        mock_set.assert_not_called()
        assert result is creds


def test_store_writes_when_not_disabled():
    creds = Credentials(access_token="tok", for_endpoint="foo", refresh_token="rtok")
    with patch("flyte.remote._client.auth._keyring.keyring.set_password") as mock_set:
        KeyringStore.store(creds, disable=False)
        assert mock_set.call_count == 2


def test_retrieve_skips_when_disabled():
    with patch("flyte.remote._client.auth._keyring.keyring.get_password") as mock_get:
        result = KeyringStore.retrieve("foo", disable=True)
        mock_get.assert_not_called()
        assert result is None


def test_retrieve_reads_when_not_disabled():
    with patch("flyte.remote._client.auth._keyring.keyring.get_password", return_value=None) as mock_get:
        KeyringStore.retrieve("foo", disable=False)
        assert mock_get.called


def test_delete_skips_when_disabled():
    with patch("flyte.remote._client.auth._keyring.keyring.delete_password") as mock_del:
        KeyringStore.delete("foo", disable=True)
        mock_del.assert_not_called()


def test_delete_calls_when_not_disabled():
    with patch("flyte.remote._client.auth._keyring.keyring.delete_password") as mock_del:
        KeyringStore.delete("foo", disable=False)
        assert mock_del.called
