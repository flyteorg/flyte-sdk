"""Error types raised by the Slack plugin."""

from __future__ import annotations


class SlackPluginError(Exception):
    """Base class for all errors raised by the Slack plugin."""


class MissingCredentialsError(SlackPluginError):
    """Raised when an operation requires a Slack token but none is mounted.

    The message names the environment variable the plugin looked at, which in
    a Flyte deployment corresponds to a `flyte.Secret` that needs to be
    created and requested by the task or app environment.
    """

    def __init__(self, env_var: str):
        self.env_var = env_var
        super().__init__(
            f"Slack bot token not found: set the {env_var} environment variable. "
            f"On Flyte, create a secret (flyte create secret {env_var} ...) and "
            f"add it to your task or app environment's `secrets=[...]`."
        )


class EventSignatureError(SlackPluginError):
    """Raised when an incoming Events API request fails signature verification."""


class SlackAPIError(SlackPluginError):
    """Raised when the Slack Web API returns `ok: false` or an HTTP error.

    Args:
        error: Slack error code (e.g. `channel_not_found`) or an HTTP message.
        status_code: HTTP status code, if one was returned.
        url: Request URL.
    """

    def __init__(self, error: str, *, status_code: int = 200, url: str = ""):
        self.error = error
        self.status_code = status_code
        self.url = url
        super().__init__(f"Slack API error for {url or '<unknown>'}: {error} (HTTP {status_code})")

    @property
    def is_rate_limited(self) -> bool:
        """Whether this error is a rate-limit response (HTTP 429)."""
        return self.status_code == 429
