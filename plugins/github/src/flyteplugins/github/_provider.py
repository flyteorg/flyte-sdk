"""GitHub webhook verification and payload normalization."""

from __future__ import annotations

from typing import Any, Mapping

from flyte.extras.webhooks import (
    Provider,
    SignatureError,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)

#: Environment variable this provider reads its secret from by default.
DEFAULT_SECRET_ENV = "GITHUB_WEBHOOK_SECRET"


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Hub-Signature-256` HMAC over the raw body."""
    signature = lower_headers(headers).get("x-hub-signature-256")
    if not signature or not signature.startswith("sha256="):
        return False
    return constant_time_equals(hex_hmac_sha256(secret, body), signature.removeprefix("sha256="))


def handshake(headers: Mapping[str, str], body: bytes) -> dict[str, Any] | None:
    """Answer the `ping` GitHub sends when a webhook is created."""
    if lower_headers(headers).get("x-github-event") == "ping":
        return {"ok": True, "ping": True}
    return None


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a GitHub delivery into a `WebhookEvent`."""
    lowered = lower_headers(headers)
    event_type = lowered.get("x-github-event")
    if not event_type:
        raise SignatureError("missing X-GitHub-Event header")
    payload = json_body(body)

    repo = payload.get("repository") or {}
    issue_or_pr = payload.get("pull_request") or payload.get("issue") or {}
    # `comment` covers issue_comment / commit_comment / review comments; `review`
    # covers pull_request_review. Either identifies the event within its issue, so
    # two comments on one issue do not collapse onto a single dedupe key.
    comment = payload.get("comment") or payload.get("review") or {}
    number = issue_or_pr.get("number")

    resource = None
    if repo.get("full_name") and number is not None:
        resource = f"{repo['full_name']}#{number}"
        if comment.get("id") is not None:
            resource = f"{resource}:{comment['id']}"

    return WebhookEvent(
        provider="github",
        event_type=event_type,
        action=payload.get("action"),
        delivery_id=lowered.get("x-github-delivery", ""),
        resource_id=resource,
        occurred_at=issue_or_pr.get("updated_at"),
        scope=repo.get("full_name"),
        title=issue_or_pr.get("title"),
        url=comment.get("html_url") or issue_or_pr.get("html_url"),
        actor=(payload.get("sender") or {}).get("login"),
        payload=payload,
    )


class GitHubProvider(Provider):
    """GitHub's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.github import GitHubProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[GitHubProvider()])
    ```

    Args:
        secret_env: Environment variable holding the secret, mounted from a
            `flyte.Secret`. Override only if you store it under a non-standard
            name; the default is what the docs and examples assume.
    """

    def __init__(self, *, secret_env: str = DEFAULT_SECRET_ENV) -> None:
        super().__init__(
            name="github",
            secret_env=secret_env,
            verify=verify,
            parse=parse,
            handshake=handshake,
            setup_hint="repository Settings -> Webhooks -> Add webhook",
        )
