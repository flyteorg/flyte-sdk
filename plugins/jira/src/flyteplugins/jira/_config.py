"""Configuration for the Jira plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL_ENV_VAR = "JIRA_BASE_URL"
DEFAULT_EMAIL_ENV_VAR = "JIRA_EMAIL"
DEFAULT_API_TOKEN_ENV_VAR = "JIRA_API_TOKEN"
DEFAULT_WEBHOOK_TOKEN_ENV_VAR = "JIRA_WEBHOOK_TOKEN"
DEFAULT_API_PATH = "/rest/api/3"


@dataclass(frozen=True)
class Config:
    """Client and webhook configuration for the Jira Cloud plugin.

    Args:
        base_url_env: Environment variable holding the Jira Cloud site URL,
            e.g. `https://acme.atlassian.net`. Defaults to `JIRA_BASE_URL`.
        email_env: Environment variable holding the account email used for
            basic auth. Defaults to `JIRA_EMAIL`.
        api_token_env: Environment variable holding the API token created at
            id.atlassian.net. Defaults to `JIRA_API_TOKEN`.
        webhook_token_env: Environment variable holding the shared token the
            webhook receiver expects in the `X-Webhook-Token` header (Jira
            webhooks are not signed). Defaults to `JIRA_WEBHOOK_TOKEN`.
        api_path: REST API path prefix.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and 429 responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    base_url_env: str = DEFAULT_BASE_URL_ENV_VAR
    email_env: str = DEFAULT_EMAIL_ENV_VAR
    api_token_env: str = DEFAULT_API_TOKEN_ENV_VAR
    webhook_token_env: str = DEFAULT_WEBHOOK_TOKEN_ENV_VAR
    api_path: str = DEFAULT_API_PATH
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def base_url(self) -> str | None:
        """Read the site URL from the environment, or None if unset."""
        return os.environ.get(self.base_url_env)

    def email(self) -> str | None:
        """Read the account email from the environment, or None if unset."""
        return os.environ.get(self.email_env)

    def api_token(self) -> str | None:
        """Read the API token from the environment, or None if unset."""
        return os.environ.get(self.api_token_env)

    def webhook_token(self) -> str | None:
        """Read the webhook token from the environment, or None if unset."""
        return os.environ.get(self.webhook_token_env)


#: Module-level default configuration.
default_config = Config()
