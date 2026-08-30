"""Configuration for the Notion plugin.

All credentials are resolved from environment variables, which in a Flyte
deployment are populated by mounting `flyte.Secret` objects onto the task or
app environment. The defaults match the standard secret names used throughout
the plugin documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.notion.com/v1"
DEFAULT_TOKEN_ENV_VAR = "NOTION_TOKEN"
DEFAULT_POLL_TOKEN_ENV_VAR = "NOTION_POLL_TOKEN"
DEFAULT_NOTION_VERSION = "2022-06-28"


@dataclass(frozen=True)
class Config:
    """Client and app configuration for the Notion plugin.

    Args:
        token_env: Name of the environment variable holding the Notion
            internal-integration token (`ntn_...` / `secret_...`). Defaults to
            `NOTION_TOKEN`.
        poll_token_env: Name of the environment variable holding the shared
            token that protects the app's `/api/poll` endpoint. Defaults to
            `NOTION_POLL_TOKEN`.
        api_base_url: Notion API base URL.
        notion_version: Value of the `Notion-Version` header.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries on transient failures
            (connection errors, 5xx, and 429 responses).
        retry_backoff: Base backoff in seconds between retries; grows
            exponentially.
    """

    token_env: str = DEFAULT_TOKEN_ENV_VAR
    poll_token_env: str = DEFAULT_POLL_TOKEN_ENV_VAR
    api_base_url: str = DEFAULT_API_BASE_URL
    notion_version: str = DEFAULT_NOTION_VERSION
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    def token(self) -> str | None:
        """Read the integration token from the environment, or None if unset."""
        return os.environ.get(self.token_env)

    def poll_token(self) -> str | None:
        """Read the poll-endpoint token from the environment, or None if unset."""
        return os.environ.get(self.poll_token_env)


#: Module-level default configuration.
default_config = Config()
