"""Error types raised by the Linear plugin."""

from __future__ import annotations

from typing import Any


class LinearPluginError(Exception):
    """Base class for all errors raised by the Linear plugin."""


class MissingCredentialsError(LinearPluginError):
    """Raised when an operation requires a Linear API key but none is mounted.

    The message names the environment variable the plugin looked at, which in
    a Flyte deployment corresponds to a `flyte.Secret` that needs to be
    created and requested by the task or app environment.
    """

    def __init__(self, env_var: str):
        self.env_var = env_var
        super().__init__(
            f"Linear API key not found: set the {env_var} environment variable. "
            f"On Flyte, create a secret (flyte create secret {env_var} ...) and "
            f"add it to your task or app environment's `secrets=[...]`."
        )


class WebhookSignatureError(LinearPluginError):
    """Raised when an incoming webhook payload fails signature verification."""


class LinearAPIError(LinearPluginError):
    """Raised when the Linear GraphQL API returns errors or an HTTP failure.

    Args:
        message: First GraphQL error message, or the HTTP error.
        status_code: HTTP status code, if one was returned.
        errors: All GraphQL errors from the response, when present.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 200,
        errors: list[dict[str, Any]] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.errors = errors or []
        super().__init__(f"Linear API error: {message}")
