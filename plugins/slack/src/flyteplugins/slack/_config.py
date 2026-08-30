"""Configuration for the Slack plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://slack.com/api"
DEFAULT_BOT_TOKEN_ENV_VAR = "SLACK_BOT_TOKEN"
DEFAULT_SIGNING_SECRET_ENV_VAR = "SLACK_SIGNING_SECRET"
DEFAULT_USER_AGENT = "flyteplugins-slack"


@dataclass(frozen=True)
class Config:
    """Client and webhook configuration for the Slack plugin.

    Args:
        bot_token_env: Name of the environment variable holding the Slack bot
            token (`xoxb-...`). Defaults to `SLACK_BOT_TOKEN`.
        signing_secret_env: Name of the environment variable holding the Slack
            app signing secret used to verify Events API requests. Defaults to
            `SLACK_SIGNING_SECRET`.
        api_base_url: Slack Web API base URL.
        user_agent: Value of the `User-Agent` header.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and 429 responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    bot_token_env: str = DEFAULT_BOT_TOKEN_ENV_VAR
    signing_secret_env: str = DEFAULT_SIGNING_SECRET_ENV_VAR
    api_base_url: str = DEFAULT_API_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def bot_token(self) -> str | None:
        """Read the bot token from the environment, or None if unset."""
        return os.environ.get(self.bot_token_env)

    def signing_secret(self) -> str | None:
        """Read the signing secret from the environment, or None if unset."""
        return os.environ.get(self.signing_secret_env)


#: Module-level default configuration.
default_config = Config()
