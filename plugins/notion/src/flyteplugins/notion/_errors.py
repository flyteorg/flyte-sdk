"""Error types raised by the Notion plugin."""

from __future__ import annotations

from typing import Any


class NotionPluginError(Exception):
    """Base class for all errors raised by the Notion plugin."""


class MissingCredentialsError(NotionPluginError):
    """Raised when an operation requires a Notion token but none is mounted.

    The message names the environment variable the plugin looked at, which in
    a Flyte deployment corresponds to a `flyte.Secret` that needs to be
    created and requested by the task or app environment.
    """

    def __init__(self, env_var: str):
        self.env_var = env_var
        super().__init__(
            f"Notion token not found: set the {env_var} environment variable. "
            f"On Flyte, create a secret (flyte create secret {env_var} ...) and "
            f"add it to your task or app environment's `secrets=[...]`."
        )


class NotionAPIError(NotionPluginError):
    """Raised when the Notion API returns an error response.

    Args:
        status_code: HTTP status code returned by Notion.
        code: Notion error code (e.g. `object_not_found`), when present.
        message: Error message from the response body.
        url: Request URL.
        body: Parsed JSON body of the error response, if any.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str = "",
        url: str = "",
        body: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.code = code
        self.url = url
        self.body = body or {}
        super().__init__(f"Notion API error {status_code} ({code or 'unknown'}) for {url or '<unknown>'}: {message}")

    @property
    def is_rate_limited(self) -> bool:
        """Whether this error is a rate-limit response (HTTP 429)."""
        return self.status_code == 429
