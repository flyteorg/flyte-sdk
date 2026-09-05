"""GitHub App auth: minting, fallbacks, and clone-URL construction.

Modeled on the agents this replaces (Cally's `github_app.py`, Nodey's
`get_github_app_token`): the JWT must verify against the app's key, missing
configuration and GitHub outages degrade to None rather than crashing, and the
clone URL carries the `x-access-token` username GitHub expects.
"""

from __future__ import annotations

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from flyteplugins.github import _app_auth, clone_url, mint_installation_token


@pytest.fixture(scope="module")
def rsa_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return key, pem


@pytest.fixture
def app_env(monkeypatch, rsa_key):
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "67890")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", rsa_key[1])
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)


@pytest.fixture
def no_env(monkeypatch):
    for env in ("GITHUB_APP_ID", "GITHUB_APP_INSTALLATION_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(env, raising=False)


def test_the_app_jwt_verifies_and_hits_the_installation(monkeypatch, app_env, rsa_key):
    """The JWT sent to GitHub must verify against the app's key and carry the
    app id as issuer — a malformed JWT fails silently at GitHub."""
    seen = {}

    def fake_post(url, *, bearer):
        seen["url"], seen["bearer"] = url, bearer
        return {"token": "ghs_short_lived"}

    monkeypatch.setattr(_app_auth, "_post_json", fake_post)
    assert mint_installation_token() == "ghs_short_lived"
    assert seen["url"].endswith("/app/installations/67890/access_tokens")
    claims = jwt.decode(seen["bearer"], rsa_key[0].public_key(), algorithms=["RS256"])
    assert claims["iss"] == "12345"


def test_missing_app_secrets_fall_back_to_a_plain_token(monkeypatch, no_env):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_legacy")
    assert mint_installation_token() == "ghp_legacy"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.setenv("GH_TOKEN", "ghp_older")
    assert mint_installation_token() == "ghp_older"


def test_nothing_configured_returns_none(no_env):
    assert mint_installation_token() is None


def test_a_github_outage_degrades_to_none_not_a_crash(monkeypatch, app_env):
    def boom(url, *, bearer):
        raise RuntimeError("github down")

    monkeypatch.setattr(_app_auth, "_post_json", boom)
    assert mint_installation_token() is None


def test_explicit_arguments_beat_the_environment(monkeypatch, no_env, rsa_key):
    monkeypatch.setattr(_app_auth, "_post_json", lambda url, *, bearer: {"token": "ghs_explicit"})
    token = mint_installation_token(app_id="1", installation_id="2", private_key=rsa_key[1])
    assert token == "ghs_explicit"


def test_clone_url_authenticates_as_x_access_token():
    assert clone_url("unionai/cloud", "ghs_tok") == "https://x-access-token:ghs_tok@github.com/unionai/cloud.git"


def test_clone_url_without_a_token_is_plain():
    assert clone_url("flyteorg/flyte-sdk") == "https://github.com/flyteorg/flyte-sdk.git"
