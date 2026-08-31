"""Configuration for the Linear plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.linear.app/graphql"
DEFAULT_API_KEY_ENV_VAR = "LINEAR_API_KEY"
DEFAULT_WEBHOOK_SECRET_ENV_VAR = "LINEAR_WEBHOOK_SECRET"
DEFAULT_USER_AGENT = "flyteplugins-linear"


@dataclass(frozen=True)
class Config:
    """Client and webhook configuration for the Linear plugin.

    Args:
        api_key_env: Name of the environment variable holding the Linear API
            key (Settings → API → Personal API keys). Defaults to
            `LINEAR_API_KEY`.
        webhook_secret_env: Name of the environment variable holding the
            webhook signing secret shown when a Linear webhook is created.
            Defaults to `LINEAR_WEBHOOK_SECRET`.
        api_base_url: Linear GraphQL API endpoint.
        user_agent: Value of the `User-Agent` header.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and 429 responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    api_key_env: str = DEFAULT_API_KEY_ENV_VAR
    webhook_secret_env: str = DEFAULT_WEBHOOK_SECRET_ENV_VAR
    api_base_url: str = DEFAULT_API_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def api_key(self) -> str | None:
        """Read the API key from the environment, or None if unset."""
        return os.environ.get(self.api_key_env)

    def webhook_secret(self) -> str | None:
        """Read the webhook secret from the environment, or None if unset."""
        return os.environ.get(self.webhook_secret_env)


#: Module-level default configuration.
default_config = Config()
