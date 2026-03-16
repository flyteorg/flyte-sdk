from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Union, cast

from flyteidl2.notification import definition_pb2
from flyteidl2.task import run_pb2

from flyte.notify import Email, NamedDelivery, NamedRule, Notification, Slack, Teams, Webhook

_HTTP_METHOD_MAP = {
    "GET": definition_pb2.HTTP_METHOD_GET,
    "HEAD": definition_pb2.HTTP_METHOD_HEAD,
    "POST": definition_pb2.HTTP_METHOD_POST,
    "PUT": definition_pb2.HTTP_METHOD_PUT,
    "DELETE": definition_pb2.HTTP_METHOD_DELETE,
    "PATCH": definition_pb2.HTTP_METHOD_PATCH,
}


def _collect_phases(notifications: Tuple[Notification, ...]) -> Tuple[str, ...]:
    """Collect the unique phases across all notifications, preserving order."""
    seen: dict[str, None] = {}
    for n in notifications:
        phases = cast(Tuple, n.on_phase)  # always a tuple after __post_init__
        for phase in phases:
            seen.setdefault(phase.value, None)
    return tuple(seen)


def to_inline_rule(notifications: Union[Notification, Tuple[Notification, ...]]) -> run_pb2.InlineRule:
    """Convert one or more Notification objects into an InlineRule proto.

    Each notification becomes a DeliveryOption. Phases are collected from all
    notifications. Today the proto only supports phase_regex (a single string),
    so we set it to the first phase as a placeholder. Once the proto adds a
    repeated phases field, this will switch to populating that instead.
    """
    if isinstance(notifications, Notification):
        notifications = (notifications,)

    phases = _collect_phases(notifications)
    delivery_options = [to_delivery_option(n) for n in notifications]

    # TODO: switch to repeated phases field once available in the proto.
    # For now, set phase_regex to the first phase as a simple placeholder.
    return run_pb2.InlineRule(
        delivery_options=delivery_options,
        checks=run_pb2.InlineRuleChecks(phase_regex=phases[0]) if phases else None,
    )


def _to_rule_id(*, org: str, project: str, domain: str, name: str) -> definition_pb2.RuleId:
    return definition_pb2.RuleId(
        org=org,
        project=project,
        domain=domain,
        name=name,
    )


def resolve_notification_settings(
    notifications: NamedRule | Notification | Tuple[Notification, ...],
    *,
    org: str = "",
    project: str = "",
    domain: str = "",
) -> tuple[definition_pb2.RuleId | None, run_pb2.InlineRule | None]:
    """Resolve user-facing notification specs into proto fields for RunSpec.

    Returns (rule_id, inline_rule) — exactly one will be set, the other None.
    """
    if isinstance(notifications, NamedRule):
        return _to_rule_id(org=org, project=project, domain=domain, name=notifications.name), None
    return None, to_inline_rule(notifications)


def to_delivery_option(n: Notification) -> run_pb2.DeliveryOption:
    """Convert a single Notification into a DeliveryOption proto."""
    if isinstance(n, NamedDelivery):
        return run_pb2.DeliveryOption(
            config_id=definition_pb2.DeliveryConfigId(name=n.name),
        )
    return run_pb2.DeliveryOption(
        template=to_delivery_config_template(n),
    )


def to_delivery_config_template(n: Notification) -> definition_pb2.DeliveryConfigTemplate:
    """Convert a Notification into a DeliveryConfigTemplate proto."""
    if isinstance(n, Email):
        return definition_pb2.DeliveryConfigTemplate(
            email=definition_pb2.EmailDeliveryTemplate(
                subject=n.subject,
                to=list(n.recipients),
                text_template=n.body,
            ),
        )
    elif isinstance(n, Slack):
        body: Dict[str, Any] = {"blocks": list(n.blocks)} if n.blocks else {"text": n.message}
        return definition_pb2.DeliveryConfigTemplate(
            webhook=definition_pb2.WebhookDeliveryTemplate(
                url=n.webhook_url,
                method=definition_pb2.HTTP_METHOD_POST,
                headers={"Content-Type": "application/json"},
                body_template=json.dumps(body),
            ),
        )
    elif isinstance(n, Teams):
        teams_body: Dict[str, Any] = n.card or {"title": n.title, "text": n.message}
        return definition_pb2.DeliveryConfigTemplate(
            webhook=definition_pb2.WebhookDeliveryTemplate(
                url=n.webhook_url,
                method=definition_pb2.HTTP_METHOD_POST,
                headers={"Content-Type": "application/json"},
                body_template=json.dumps(teams_body),
            ),
        )
    elif isinstance(n, Webhook):
        return definition_pb2.DeliveryConfigTemplate(
            webhook=definition_pb2.WebhookDeliveryTemplate(
                url=n.url,
                method=_HTTP_METHOD_MAP.get(n.method, definition_pb2.HTTP_METHOD_POST),
                headers=n.headers or {},
                body_template=json.dumps(n.body) if n.body else "",
            ),
        )
    else:
        raise TypeError(f"Unsupported notification type: {type(n)}")
