"""Error types raised by the Jira plugin."""

from __future__ import annotations

from typing import Any


class JiraPluginError(Exception):
    """Base class for all errors raised by the Jira plugin."""


class MissingCredentialsError(JiraPluginError):
    """Raised when an operation requires a Jira credential but it is missing.

    The message names the environment variable the plugin looked at, which in
    a Flyte deployment corresponds to a `flyte.Secret` that needs to be
    created and requested by the task or app environment.
    """

    def __init__(self, env_var: str):
        self.env_var = env_var
        super().__init__(
            f"Jira credential not found: set the {env_var} environment variable. "
            f"On Flyte, create a secret (flyte create secret {env_var} ...) and "
            f"add it to your task or app environment's `secrets=[...]`."
        )


class WebhookSignatureError(JiraPluginError):
    """Raised when an incoming webhook payload fails token verification."""


class JiraAPIError(JiraPluginError):
    """Raised when the Jira REST API returns an error response.

    Args:
        status_code: HTTP status code returned by Jira.
        message: Error message extracted from the response body.
        url: Request URL.
        body: Parsed JSON body of the error response, if any.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        url: str = "",
        body: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.url = url
        self.body = body or {}
        super().__init__(f"Jira API error {status_code} for {url or '<unknown>'}: {message}")

    @property
    def is_rate_limited(self) -> bool:
        """Whether this error is a rate-limit response (HTTP 429)."""
        return self.status_code == 429
