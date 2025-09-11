from pathlib import Path

import pytest
from keyring.errors import PasswordDeleteError

from flyte._keyring.file import SimplePlainTextKeyring


class _TestSimplePlainTextKeyring(SimplePlainTextKeyring):
    def __init__(self, home_root: Path):
        self.home_root = home_root

    @property
    def file_path(self) -> Path:
        return self.home_root / ".union" / "keyring.cfg"


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
