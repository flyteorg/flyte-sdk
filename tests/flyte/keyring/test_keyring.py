from pathlib import Path

import pytest
from keyring.errors import PasswordDeleteError

from flyte._keyring.file import SimplePlainTextKeyring


class _TestSimplePlainTextKeyring(SimplePlainTextKeyring):
    def __init__(self, home_root: Path):
        self.home_root = home_root

    @property
    def file_path(self) -> Path:
        return self.home_root / ".flyte" / "keyring.cfg"


def test_password_create_cycle(tmp_path):
    keyring = _TestSimplePlainTextKeyring(home_root=tmp_path)

    items = [
        ("acme.flyte.org", "access_token", "fun-token"),
        ("foo.flyte.org", "access_token", "another-token"),
        ("bar.flyte.org", "access_token", "another-token"),
    ]

    for service, username, password in items:
        keyring.set_password(service, username, password)

    for service, username, password in items:
        keyring_password = keyring.get_password(service, username)
        assert keyring_password == password


def test_password_delete_error(tmp_path):
    keyring = _TestSimplePlainTextKeyring(home_root=tmp_path)

    msg = "Config file does not exist"
    with pytest.raises(PasswordDeleteError, match=msg):
        keyring.delete_password("my_service", "wow")

    keyring.set_password("my_service", "wow", "this_password")

    msg = "Password not found"
    with pytest.raises(PasswordDeleteError, match=msg):
        keyring.delete_password("my_service", "wow2")

    msg = "Password not found"
    with pytest.raises(PasswordDeleteError, match=msg):
        keyring.delete_password("my_servic", "wow")


def test_priority_forced_by_env_var(tmp_path, monkeypatch):
    keyring = _TestSimplePlainTextKeyring(home_root=tmp_path)

    monkeypatch.delenv("FLYTE_USE_FILE_KEYRING", raising=False)
    assert keyring.priority < 5  # system keyring (macOS = 5) wins by default

    for value in ("1", "true", "True"):
        monkeypatch.setenv("FLYTE_USE_FILE_KEYRING", value)
        assert keyring.priority > 5  # forced: outranks the macOS keychain

    monkeypatch.setenv("FLYTE_USE_FILE_KEYRING", "0")
    assert keyring.priority < 5
