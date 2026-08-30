"""Configuration for the ClickUp plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.clickup.com/api/v2"
DEFAULT_TOKEN_ENV_VAR = "CLICKUP_TOKEN"
DEFAULT_WEBHOOK_SECRET_ENV_VAR = "CLICKUP_WEBHOOK_SECRET"
DEFAULT_CLIENT_ID = "flyteplugins-clickup"


@dataclass(frozen=True)
class Config:
    """Client and webhook configuration for the ClickUp plugin.

    Args:
        token_env: Name of the environment variable holding the ClickUp
            personal API token (Settings → Apps → API Token). Defaults to
            `CLICKUP_TOKEN`.
        webhook_secret_env: Name of the environment variable holding the
            webhook signing secret shown when a ClickUp webhook is created.
            Defaults to `CLICKUP_WEBHOOK_SECRET`.
        api_base_url: ClickUp REST API v2 base URL.
        client_id: Value of the `ClickUp-Client` header. ClickUp asks API
            clients to identify themselves.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and 429 responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    token_env: str = DEFAULT_TOKEN_ENV_VAR
    webhook_secret_env: str = DEFAULT_WEBHOOK_SECRET_ENV_VAR
    api_base_url: str = DEFAULT_API_BASE_URL
    client_id: str = DEFAULT_CLIENT_ID
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def token(self) -> str | None:
        """Read the API token from the environment, or None if unset."""
        return os.environ.get(self.token_env)

    def webhook_secret(self) -> str | None:
        """Read the webhook secret from the environment, or None if unset."""
        return os.environ.get(self.webhook_secret_env)


#: Module-level default configuration.
default_config = Config()
