from __future__ import annotations

import json
from typing import Any, Dict, Tuple, cast

from flyteidl2.common import phase_pb2
from flyteidl2.notification import definition_pb2
from flyteidl2.task import run_pb2

from flyte.models import ActionPhase
from flyte.notify import Email, NamedDelivery, NamedRule, Notification, Slack, Teams, Webhook

_HTTP_METHOD_MAP = {
    "GET": definition_pb2.HTTP_METHOD_GET,
    "HEAD": definition_pb2.HTTP_METHOD_HEAD,
    "POST": definition_pb2.HTTP_METHOD_POST,
    "PUT": definition_pb2.HTTP_METHOD_PUT,
    "DELETE": definition_pb2.HTTP_METHOD_DELETE,
    "CONNECT": definition_pb2.HTTP_METHOD_CONNECT,
    "OPTIONS": definition_pb2.HTTP_METHOD_OPTIONS,
    "TRACE": definition_pb2.HTTP_METHOD_TRACE,
    "PATCH": definition_pb2.HTTP_METHOD_PATCH,
}

_ACTION_PHASE_MAP: dict[ActionPhase, phase_pb2.ActionPhase.ValueType] = {
    ActionPhase.QUEUED: phase_pb2.ACTION_PHASE_QUEUED,
    ActionPhase.WAITING_FOR_RESOURCES: phase_pb2.ACTION_PHASE_WAITING_FOR_RESOURCES,
    ActionPhase.INITIALIZING: phase_pb2.ACTION_PHASE_INITIALIZING,
    ActionPhase.RUNNING: phase_pb2.ACTION_PHASE_RUNNING,
    ActionPhase.SUCCEEDED: phase_pb2.ACTION_PHASE_SUCCEEDED,
    ActionPhase.FAILED: phase_pb2.ACTION_PHASE_FAILED,
    ActionPhase.ABORTED: phase_pb2.ACTION_PHASE_ABORTED,
    ActionPhase.TIMED_OUT: phase_pb2.ACTION_PHASE_TIMED_OUT,
}


def resolve_notification_settings(
    notifications: NamedRule | Notification | Tuple[Notification, ...],
) -> tuple[str | None, run_pb2.InlineRuleList | None]:
    """Resolve user-facing notification specs into proto fields for RunSpec.

    Returns (notification_rule_name, notification_rules) — exactly one will be set.
    """
    if isinstance(notifications, NamedRule):
        return notifications.name, None
    return None, _to_inline_rule_list(notifications)


def _to_inline_rule_list(
    notifications: Notification | Tuple[Notification, ...],
) -> run_pb2.InlineRuleList:
    if isinstance(notifications, Notification):
        notifications = (notifications,)
    return run_pb2.InlineRuleList(rules=[_to_inline_rule(n) for n in notifications])


def _to_inline_rule(n: Notification) -> run_pb2.InlineRule:
    """Convert a single Notification into an InlineRule proto.

    Each Notification maps 1:1 to an InlineRule with its own phases and delivery.
    """
    on_phases = [_ACTION_PHASE_MAP[p] for p in cast(Tuple, n.on_phase)]

    if isinstance(n, NamedDelivery):
        return run_pb2.InlineRule(on_phases=on_phases, delivery_config_name=n.name)

    return run_pb2.InlineRule(on_phases=on_phases, delivery_template=_to_delivery_config_template(n))


def _to_delivery_config_template(n: Notification) -> definition_pb2.DeliveryConfigTemplate:
    if isinstance(n, Email):
        return definition_pb2.DeliveryConfigTemplate(
            email=definition_pb2.EmailDeliveryTemplate(
                to=[definition_pb2.EmailRecipient(address=r) for r in n.recipients],
                cc=[definition_pb2.EmailRecipient(address=r) for r in n.cc],
                bcc=[definition_pb2.EmailRecipient(address=r) for r in n.bcc],
                inline=definition_pb2.InlineEmailTemplate(
                    subject=n.subject,
                    text_template=n.body,
                    html_template=n.html_body or "",
                ),
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
                body_template=json.dumps(n.body) if n.body is not None else "",
            ),
        )
    else:
        raise TypeError(f"Unsupported notification type: {type(n)}")
