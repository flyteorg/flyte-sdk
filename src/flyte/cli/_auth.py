import asyncio
import base64
import json
import time

import rich_click as click

from . import _common as common


@click.group(name="auth")
def auth():
    """Inspect the CLI's Union authentication."""


@auth.command(name="token")
@click.pass_obj
def token(cfg: common.CLIConfig):
    """Print the current Union access token to stdout.

    Reuses the CLI's own authentication (with auto-refresh), so a shell can inject
    a fresh bearer without any token handling of its own — e.g. talking to the
    read-only in-cluster kubectl proxy:

        kubectl --server="https://union-k8s-ro.apps.<dp>.<domain>" \\
                --token="$(flyte auth token)" get pods -A

    Prints only the raw token, so it composes cleanly in $(...). The token is
    written to stdout only; it is never logged.
    """
    cfg.init()
    click.echo(asyncio.run(_current_access_token(cfg)))


def _expired(access_token: str, skew_seconds: int = 120) -> bool:
    """Best-effort: True if the JWT is expired or expires within `skew_seconds`.
    Unparseable tokens are treated as expired so we refresh rather than emit a
    stale bearer."""
    try:
        payload = access_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        if exp is None:
            return False
        return time.time() >= (float(exp) - skew_seconds)
    except Exception:  # noqa: BLE001
        return True


async def _current_access_token(cfg: common.CLIConfig) -> str:
    # Reuse the proxy command's authenticator builder so this stays in lockstep
    # with how the rest of the CLI authenticates (endpoint, auth mode, TLS posture).
    from ._proxy import _build_authenticator

    authenticator = _build_authenticator(cfg)
    creds = authenticator.get_credentials()
    if creds is None or not creds.access_token or _expired(creds.access_token):
        await authenticator.refresh_credentials()
        creds = authenticator.get_credentials()
    if not creds or not creds.access_token:
        raise click.ClickException("No Union access token available; run a `flyte` command that logs in first.")
    return creds.access_token
