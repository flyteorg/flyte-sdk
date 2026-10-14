from subprocess import CompletedProcess
from unittest.mock import patch

import pytest
from keyring.errors import KeyringError, PasswordDeleteError, PasswordSetError

from flyte._keyring.macos import SecurityCliKeyring


@pytest.fixture
def kr():
    return SecurityCliKeyring()


def _proc(rc=0, stdout="", stderr=""):
    return CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def test_get_password_strips_single_trailing_newline(kr):
    with patch("subprocess.run", return_value=_proc(stdout="tok\n\n")):
        assert kr.get_password("svc", "tokens") == "tok\n"


def test_get_password_returns_none_only_when_not_found(kr):
    with patch("subprocess.run", return_value=_proc(rc=44, stderr="not found")):
        assert kr.get_password("svc", "tokens") is None


def test_get_password_raises_on_other_failures(kr):
    # A locked keychain / denied access must not masquerade as "no password".
    with patch("subprocess.run", return_value=_proc(rc=36, stderr="User interaction is not allowed.")):
        with pytest.raises(KeyringError, match="interaction"):
            kr.get_password("svc", "tokens")


def test_set_password_passes_secret_as_single_argv_element(kr):
    # argv is execve'd verbatim — no interpreter parses the secret, so
    # newlines, quotes, and >4KiB payloads are stored exactly as given
    # (all of which are unsafe through `security -i`).
    secret = '{"access_token": "a\nb \\" c"}' + "x" * 8192
    with patch("subprocess.run", return_value=_proc()) as mock_run:
        kr.set_password("svc", "tokens", secret)

    add_argv = mock_run.call_args.args[0]
    assert add_argv[0] == "/usr/bin/security"
    assert add_argv[1] == "add-generic-password"
    assert add_argv[-2:] == ["-w", secret]


def test_set_password_recreates_item(kr):
    # delete-then-add: an in-place update (-U) would preserve a legacy
    # interpreter-owned item's ACL and keep prompting.
    with patch("subprocess.run", return_value=_proc()) as mock_run:
        kr.set_password("svc", "tokens", "v")

    subcommands = [call.args[0][1] for call in mock_run.call_args_list]
    assert subcommands == ["delete-generic-password", "add-generic-password"]


def test_set_password_raises_typed_error(kr):
    with patch("subprocess.run", side_effect=[_proc(rc=44), _proc(rc=1, stderr="boom")]):
        with pytest.raises(PasswordSetError, match="boom"):
            kr.set_password("svc", "tokens", "v")


def test_delete_password_raises_when_not_found(kr):
    with patch("subprocess.run", return_value=_proc(rc=44)):
        with pytest.raises(PasswordDeleteError):
            kr.delete_password("svc", "tokens")


def test_all_calls_have_timeouts(kr):
    # A wedged SecurityAgent must not hang the auth path forever.
    with patch("subprocess.run", return_value=_proc()) as mock_run:
        kr.set_password("svc", "tokens", "v")
        kr.get_password("svc", "tokens")
    assert all(call.kwargs.get("timeout") for call in mock_run.call_args_list)
