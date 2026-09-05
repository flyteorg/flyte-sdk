"""GitHub App installation tokens, for agents that clone, push, or open PRs.

The pattern: hold no personal access token. Authenticate as a GitHub App and
mint a short-lived installation token whenever one is needed. Tokens live one
hour — plenty for a clone or a `gh pr create`, useless to an attacker who
exfiltrates one from a log.

This belongs in the plugin because every agent otherwise carries its own copy
of the same fifty lines: sign an RS256 JWT as the app, trade it for an
installation token, splice it into a clone URL. The webhook side of a GitHub
agent already imports this package, so the auth side comes from it too.

Inputs, all injected as Flyte secrets (or passed explicitly):

  GITHUB_APP_ID               the app's numeric id
  GITHUB_APP_INSTALLATION_ID  the installation's numeric id
  GITHUB_APP_PRIVATE_KEY      the app's PEM private key

`GITHUB_TOKEN` and `GH_TOKEN` are honored as fallbacks so a deployment can
migrate one secret at a time; once the app secrets exist the fallback never
fires.

`PyJWT[crypto]` signs the app JWT, so a webhook-only install stays lean:

```bash
pip install "flyteplugins-github[auth]"
```
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
#: App tokens authenticate with this literal username in clone URLs.
GIT_USERNAME = "x-access-token"

#: Environment variables the app credentials default to.
DEFAULT_APP_ID_ENV = "GITHUB_APP_ID"
DEFAULT_INSTALLATION_ID_ENV = "GITHUB_APP_INSTALLATION_ID"
DEFAULT_PRIVATE_KEY_ENV = "GITHUB_APP_PRIVATE_KEY"

#: Plain-token fallbacks, checked in order when the app credentials are absent.
FALLBACK_TOKEN_ENVS = ("GITHUB_TOKEN", "GH_TOKEN")


def _post_json(url: str, *, bearer: str) -> dict[str, Any]:
    """POST to the GitHub API with a bearer credential and return the JSON."""
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {bearer}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def mint_installation_token(
    *,
    app_id: str | None = None,
    installation_id: str | None = None,
    private_key: str | None = None,
) -> str | None:
    """A fresh installation token, or None with a logged reason.

    None means "proceed unauthenticated or not at all" — treat it the way a
    missing token is treated today, so a half-configured deployment degrades
    instead of crashing. Synchronous, one HTTPS round trip: call through
    `asyncio.to_thread` from handlers and other async code.

    Args:
        app_id: The app's numeric id; otherwise read from `GITHUB_APP_ID`.
        installation_id: The installation's numeric id; otherwise read from
            `GITHUB_APP_INSTALLATION_ID`.
        private_key: The app's PEM private key; otherwise read from
            `GITHUB_APP_PRIVATE_KEY`.
    """
    app_id = app_id or os.environ.get(DEFAULT_APP_ID_ENV)
    installation_id = installation_id or os.environ.get(DEFAULT_INSTALLATION_ID_ENV)
    private_key = private_key or os.environ.get(DEFAULT_PRIVATE_KEY_ENV)

    if not (app_id and installation_id and private_key):
        for env in FALLBACK_TOKEN_ENVS:
            fallback = os.environ.get(env)
            if fallback:
                logger.info("GitHub App credentials not set; using %s fallback", env)
                return fallback
        logger.warning(
            "Neither the %s/%s/%s secrets nor a fallback token (%s) are set; "
            "authenticated GitHub operations will be skipped",
            DEFAULT_APP_ID_ENV,
            DEFAULT_INSTALLATION_ID_ENV,
            DEFAULT_PRIVATE_KEY_ENV,
            "/".join(FALLBACK_TOKEN_ENVS),
        )
        return None

    try:
        import jwt  # PyJWT[crypto]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(
            "PyJWT is not installed. Install 'flyteplugins-github[auth]' to mint GitHub App tokens."
        ) from exc

    now = int(time.time())
    # iat is backdated 60s because GitHub rejects JWTs it considers issued in
    # the future, and clocks drift.
    app_jwt = jwt.encode({"iat": now - 60, "exp": now + 600, "iss": app_id}, private_key, algorithm="RS256")
    try:
        data = _post_json(f"{GITHUB_API}/app/installations/{installation_id}/access_tokens", bearer=app_jwt)
        return data["token"]
    except Exception as exc:  # callers degrade, they don't crash
        logger.warning("Could not mint a GitHub App installation token: %s", exc)
        return None


def clone_url(repo: str, token: str | None = None) -> str:
    """An https clone URL for `repo` ("owner/name"), authenticated when a token is given."""
    if token:
        return f"https://{GIT_USERNAME}:{token}@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"
