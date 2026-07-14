from subprocess import CompletedProcess
from unittest.mock import patch

import pytest
from keyring.errors import PasswordDeleteError

from flyte._keyring.macos import SecurityCliKeyring, _quote


@pytest.fixture
def kr():
    return SecurityCliKeyring()


def test_quote_escapes_backslashes_and_quotes():
    assert _quote('a "b" c\\d') == '"a \\"b\\" c\\\\d"'


def test_priority_darwin_only(kr):
    with patch("platform.system", return_value="Darwin"):
        assert kr.priority == 6  # outranks native macOS backend (5)
    with patch("platform.system", return_value="Linux"):
        assert kr.priority == -1


def test_get_password_strips_single_trailing_newline(kr):
    proc = CompletedProcess(args=[], returncode=0, stdout="tok\n\n", stderr="")
    with patch("subprocess.run", return_value=proc):
        assert kr.get_password("svc", "tokens") == "tok\n"


def test_get_password_returns_none_when_not_found(kr):
    proc = CompletedProcess(args=[], returncode=44, stdout="", stderr="not found")
    with patch("subprocess.run", return_value=proc):
        assert kr.get_password("svc", "tokens") is None


def test_set_password_keeps_secret_out_of_argv(kr):
    proc = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=proc) as mock_run:
        kr.set_password("svc", "tokens", '{"access_token": "se cr\\"et"}')

    argv = mock_run.call_args.args[0]
    assert argv == ["/usr/bin/security", "-i"]
    command = mock_run.call_args.kwargs["input"]
    assert command.startswith("add-generic-password -U ")
    assert '-w "{\\"access_token\\": \\"se cr\\\\\\"et\\"}"' in command


def test_set_password_raises_on_failure(kr):
    proc = CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="boom"):
            kr.set_password("svc", "tokens", "v")


def test_delete_password_raises_when_not_found(kr):
    proc = CompletedProcess(args=[], returncode=44, stdout="", stderr="")
    with patch("subprocess.run", return_value=proc):
        with pytest.raises(PasswordDeleteError):
            kr.delete_password("svc", "tokens")
