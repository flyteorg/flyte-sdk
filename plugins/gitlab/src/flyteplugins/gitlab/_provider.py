"""GitLab webhook verification and payload normalization.

GitLab does **not** sign its webhooks. There is no HMAC over the body; the
*Secret token* you set on a webhook is a static shared token that GitLab echoes
verbatim in the `X-Gitlab-Token` header. This plugin authenticates with that
token (a constant-time comparison, exactly like `JiraProvider`), and reports
`signed=False` so the dashboard says so rather than implying a guarantee that
is absent.

(GitLab's newer *Signing token* mode instead signs the body with HMAC-SHA256 in
`X-Gitlab-Signature`; this plugin does not cover it — the legacy shared token is
what the webhook *Secret token* field sets, and it is what the sample delivery
exercises.)

`parse` normalizes the payload into a `WebhookEvent`: the event type comes from
`object_kind`, the action from `object_attributes.action` (or `status` for
pipelines), and the resource id follows GitLab's `!` / `#` convention — merge
requests use `!`, issues use `#`, on a `path_with_namespace` scope.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from flyte.extras.webhooks import (
    Provider,
    WebhookEvent,
    constant_time_equals,
    json_body,
    lower_headers,
)


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    """Compare the `X-Gitlab-Token` header against the shared secret token."""
    token = lower_headers(headers).get("x-gitlab-token")
    if not token:
        return False
    return constant_time_equals(token.strip(), secret)


def _build_resource(obj_kind: str, payload: dict[str, Any], attrs: dict[str, Any], namespace: str | None) -> str | None:
    """A stable, human-readable id for the resource an event is about."""
    if obj_kind in ("merge_request", "issue"):
        iid = attrs.get("iid")
        if namespace and iid is not None:
            marker = "!" if obj_kind == "merge_request" else "#"
            return f"{namespace}{marker}{iid}"
    elif obj_kind == "note":
        # A comment's parent is beside the note, not inside object_attributes.
        parent = payload.get("merge_request") or payload.get("issue") or {}
        parent_iid = parent.get("iid")
        note_id = attrs.get("id")
        if namespace and parent_iid is not None and note_id is not None:
            marker = "!" if payload.get("merge_request") else "#"
            return f"{namespace}{marker}{parent_iid}:{note_id}"
    return None


def parse(headers: Mapping[str, str], body: bytes) -> WebhookEvent:
    """Normalize a GitLab delivery into a `WebhookEvent`."""
    payload = json_body(body)
    lowered = lower_headers(headers)

    obj_kind = payload.get("object_kind") or payload.get("event_type") or "unknown"
    attrs = payload.get("object_attributes") or {}
    project = payload.get("project") or {}
    namespace = project.get("path_with_namespace") or project.get("name")

    # A note (comment) has no title of its own; the parent MR/issue does, which is
    # what the dashboard should show for a comment the way it would for the thread.
    title = attrs.get("title")
    if obj_kind == "note":
        parent = payload.get("merge_request") or payload.get("issue") or {}
        title = parent.get("title") or title

    user = payload.get("user")
    if isinstance(user, dict):
        actor = user.get("username") or user.get("name")
    else:
        # Push and tag-push hooks send the author as a plain string instead.
        actor = payload.get("user_username") or payload.get("user_name")

    # Pipelines and deployments report their state in `status`, not `action`.
    if obj_kind in ("pipeline", "deployment"):
        action = attrs.get("status")
    else:
        action = attrs.get("action")

    return WebhookEvent(
        provider="gitlab",
        event_type=obj_kind,
        action=action,
        delivery_id=lowered.get("x-gitlab-event-uuid", ""),
        resource_id=_build_resource(obj_kind, payload, attrs, namespace),
        occurred_at=attrs.get("updated_at") or attrs.get("created_at"),
        scope=namespace,
        title=title,
        url=attrs.get("url"),
        actor=actor,
        payload=payload,
    )


class GitLabProvider(Provider):
    """GitLab's webhook provider, with its defaults pre-wired.

    ```python
    from flyte.extras.webhooks import WebhookAppEnvironment
    from flyteplugins.gitlab import GitLabProvider

    app_env = WebhookAppEnvironment(name="webhooks", providers=[GitLabProvider()])
    ```

    GitLab does not sign its webhooks, so this provider authenticates with the
    shared token in `X-Gitlab-Token` and reports `signed=False` — which is what
    makes the dashboard say so rather than implying a guarantee that is absent.

    `WebhookAppEnvironment` mounts `default_secret_env` for you, so it does not
    need naming again in `secrets=`.

    Args:
        secret_env: Environment variable holding the secret. Pass one only to
            point this provider at a secret stored under a different name;
            otherwise `default_secret_env` applies.
    """

    default_secret_env: ClassVar[str] = "GITLAB_WEBHOOK_TOKEN"

    def __init__(self, *, secret_env: str | None = None) -> None:
        super().__init__(
            name="gitlab",
            secret_env=secret_env or self.default_secret_env,
            verify=verify,
            parse=parse,
            signed=False,
            setup_hint="Project Settings -> Webhooks (secret token sent verbatim in X-Gitlab-Token)",
        )
