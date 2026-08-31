"""Configuration for the GitHub plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.github.com"
DEFAULT_TOKEN_ENV_VAR = "GITHUB_TOKEN"
DEFAULT_WEBHOOK_SECRET_ENV_VAR = "GITHUB_WEBHOOK_SECRET"
DEFAULT_USER_AGENT = "flyteplugins-github"


@dataclass(frozen=True)
class Config:
    """Client and webhook configuration for the GitHub plugin.

    Args:
        token_env: Name of the environment variable holding the GitHub personal
            access token (or GitHub App installation token). Defaults to
            `GITHUB_TOKEN`.
        webhook_secret_env: Name of the environment variable holding the webhook
            secret configured in the GitHub repository/org webhook settings.
            Defaults to `GITHUB_WEBHOOK_SECRET`.
        api_base_url: GitHub REST API base URL. Override for GitHub Enterprise
            Server, e.g. `https://github.example.com/api/v3`.
        user_agent: Value of the `User-Agent` header. GitHub requires one.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and rate-limit responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    token_env: str = DEFAULT_TOKEN_ENV_VAR
    webhook_secret_env: str = DEFAULT_WEBHOOK_SECRET_ENV_VAR
    api_base_url: str = DEFAULT_API_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def token(self) -> str | None:
        """Read the API token from the environment, or None if unset."""
        return os.environ.get(self.token_env)

    def webhook_secret(self) -> str | None:
        """Read the webhook secret from the environment, or None if unset."""
        return os.environ.get(self.webhook_secret_env)


#: Module-level default configuration. Most users should not need to override
#: this; construct a custom `Config` only for GitHub Enterprise Server or
#: non-standard secret names.
default_config = Config()
