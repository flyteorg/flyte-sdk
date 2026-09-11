"""GitHub webhook verification and payload normalization.

GitHub delivers the same payload in two body shapes, chosen per webhook in the
*Add webhook* form and signed the same way:

* content type `application/json` — the JSON is the body;
* content type `application/x-www-form-urlencoded` — the form's *default* —
  the JSON arrives under a `payload=` form field.

`verify` covers both, since the HMAC signs the raw body regardless of encoding.
`parse` normalizes both into the same `WebhookEvent`, so a webhook left on the
default content type still works.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, ClassVar, Mapping

from flyte.extras.webhooks import (
    Provider,
    SignatureError,
    WebhookEvent,
    constant_time_equals,
    hex_hmac_sha256,
    json_body,
    lower_headers,
)


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Verify the `X-Hub-Signature-256` HMAC over the raw body."""
    signature = lower_headers(headers).get("x-hub-signature-256")
    if not signature or not signature.startswith("sha256="):
        return False
    return constant_time_equals(hex_hmac_sha256(secret, body), signature.removeprefix("sha256="))


def _form_payload(body: bytes) -> dict[str, Any] | None:
    """Decode a form-encoded delivery's `payload` field, or None when the body is JSON.

    GitHub's *Add webhook* form defaults the content type to
    `application/x-www-form-urlencoded`, which wraps the JSON in a `payload=`
    form field. Sniffing the body rather than trusting Content-Type keeps
    `parse` a pure function of the delivery, which is what the conformance
    harness replays.
    """
    if body[:1] in (b"{", b"["):
        return None
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "=" not in decoded.split("&", 1)[0]:
        return None
    fields = {key: values[0] for key, values in urllib.parse.parse_qs(decoded, keep_blank_values=True).items()}
    raw = fields.get("payload")
    if raw is None:
        raise SignatureError("form-encoded delivery carries no `payload` field")
    return json_body(raw.encode("utf-8"))


def handshake(headers: Mapping[str, str], body: bytes) -> dict[str, Any] | None:
    """Answer the `ping` GitHub sends when a webhook is created."""
    if lower_headers(headers).get("x-github-event") == "ping":
        return {"ok": True, "ping": True}
    return None


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a GitHub delivery — JSON or form-encoded — into a `WebhookEvent`."""
    lowered = lower_headers(headers)
    event_type = lowered.get("x-github-event")
    if not event_type:
        raise SignatureError("missing X-GitHub-Event header")
    form = _form_payload(body)
    payload = form if form is not None else json_body(body)

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

    Either content type in GitHub's *Add webhook* form works: `application/json`
    and the default `application/x-www-form-urlencoded` normalize identically.

    `WebhookAppEnvironment` mounts `default_secret_env` for you, so it does not
    need naming again in `secrets=`.

    Args:
        secret_env: Environment variable holding the secret. Pass one only to
            point this provider at a secret stored under a different name;
            otherwise `default_secret_env` applies.
    """

    default_secret_env: ClassVar[str] = "GITHUB_WEBHOOK_SECRET"

    def __init__(self, *, secret_env: str | None = None) -> None:
        super().__init__(
            name="github",
            secret_env=secret_env or self.default_secret_env,
            verify=verify,
            parse=parse,
            handshake=handshake,
            setup_hint="repository Settings -> Webhooks -> Add webhook",
        )
