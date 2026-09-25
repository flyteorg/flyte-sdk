"""Tests for `flyte auth token` — command wiring and the JWT expiry guard."""

import base64
import json
import time

from click.testing import CliRunner

from flyte.cli._auth import _expired
from flyte.cli.main import main


def _jwt(exp) -> str:
    payload = {} if exp is None else {"exp": exp}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.sig"


def test_expired_true_for_past_and_near_expiry():
    assert _expired(_jwt(time.time() - 10)) is True
    assert _expired(_jwt(time.time() + 30)) is True  # inside the 120s skew


def test_expired_false_for_comfortably_valid_token():
    assert _expired(_jwt(time.time() + 3600)) is False


def test_expired_false_when_no_exp_claim():
    # A token without exp can't be judged expired — don't force a refresh on it.
    assert _expired(_jwt(None)) is False


def test_expired_true_for_unparseable_token():
    # Fail safe: refresh rather than emit a token we can't validate.
    assert _expired("not-a-jwt") is True


def test_auth_token_command_is_registered():
    result = CliRunner().invoke(main, ["auth", "token", "--help"])
    assert result.exit_code == 0
    assert "access token" in result.output.lower()
