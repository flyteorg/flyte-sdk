"""Local notification sender for local execution mode.

Sends notifications using httpx (webhooks) and smtplib (email) with no
extra dependencies beyond what the SDK already has.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Tuple, cast

import httpx

from flyte.models import ActionPhase
from flyte.notify._notifiers import Email, NamedDelivery, NamedRule, Notification, Slack, Teams, Webhook

logger = logging.getLogger(__name__)


def _render(template: str, context: Dict[str, str]) -> str:
    """Render template variables like {task.name} using the context dict."""
    result = template
    for key, value in context.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _render_dict(d: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
    """Recursively render template variables in dict values."""
    rendered: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            rendered[k] = _render(v, context)
        elif isinstance(v, dict):
            rendered[k] = _render_dict(v, context)
        elif isinstance(v, list):
            rendered[k] = [
                _render_dict(i, context) if isinstance(i, dict) else _render(i, context) if isinstance(i, str) else i
                for i in v
            ]
        else:
            rendered[k] = v
    return rendered


def _build_context(
    *,
    task_name: str,
    run_name: str,
    phase: ActionPhase,
    error: str = "",
    duration: str = "",
    project: str = "",
    domain: str = "",
) -> Dict[str, str]:
    """Build the template variable context dict."""
    return {
        "task.name": task_name,
        "run.name": run_name,
        "run.phase": phase.value,
        "run.error": error,
        "run.duration": duration,
        "run.timestamp": "",
        "run.url": "",
        "project": project,
        "domain": domain,
    }


async def send_notifications(
    notifications: Notification | Tuple[Notification, ...],
    *,
    phase: ActionPhase,
    task_name: str,
    run_name: str,
    error: str = "",
    duration: str = "",
    project: str = "",
    domain: str = "",
) -> None:
    """Send notifications that match the given phase.

    Silently logs and continues on any send failure — never crashes the run.
    """
    if isinstance(notifications, Notification):
        notifications = (notifications,)

    context = _build_context(
        task_name=task_name,
        run_name=run_name,
        phase=phase,
        error=error,
        duration=duration,
        project=project,
        domain=domain,
    )

    for n in notifications:
        phases = cast(Tuple[ActionPhase, ...], n.on_phase)
        if phase not in phases:
            continue
        try:
            await _send_one(n, context)
        except Exception:
            logger.warning("Failed to send %s notification", type(n).__name__, exc_info=True)


async def _send_one(n: Notification, context: Dict[str, str]) -> None:
    if isinstance(n, (NamedDelivery, NamedRule)):
        logger.info("Skipping named notification %r in local mode (only available on remote)", n.name)
        return

    if isinstance(n, Email):
        await _send_email(n, context)
    elif isinstance(n, Slack):
        await _send_slack(n, context)
    elif isinstance(n, Teams):
        await _send_teams(n, context)
    elif isinstance(n, Webhook):
        await _send_webhook(n, context)


async def _send_email(n: Email, context: Dict[str, str]) -> None:
    subject = _render(n.subject, context)
    text_body = _render(n.body, context)

    msg = MIMEMultipart("alternative") if n.html_body else MIMEText(text_body)
    if n.html_body:
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(_render(n.html_body, context), "html"))

    msg["Subject"] = subject
    msg["To"] = ", ".join(n.recipients)
    if n.cc:
        msg["Cc"] = ", ".join(n.cc)

    all_recipients = list(n.recipients) + list(n.cc) + list(n.bcc)

    try:
        with smtplib.SMTP("localhost", 25, timeout=5) as smtp:
            smtp.sendmail(msg.get("From", "flyte@localhost"), all_recipients, msg.as_string())
        logger.info("Sent email notification to %s", n.recipients)
    except (ConnectionRefusedError, OSError, smtplib.SMTPException):
        logger.warning(
            "Could not send email notification — SMTP server not available at localhost:25. "
            "Recipients: %s, Subject: %s",
            n.recipients,
            subject,
        )


async def _send_slack(n: Slack, context: Dict[str, str]) -> None:
    if n.blocks:
        body = {"blocks": _render_dict({"b": list(n.blocks)}, context)["b"]}
    else:
        body = {"text": _render(n.message or "", context)}

    async with httpx.AsyncClient() as client:
        resp = await client.post(n.webhook_url, json=body, timeout=10)
        resp.raise_for_status()
    logger.info("Sent Slack notification to %s", n.webhook_url)


async def _send_teams(n: Teams, context: Dict[str, str]) -> None:
    if n.card:
        body = _render_dict(n.card, context)
    else:
        body = {"title": _render(n.title, context), "text": _render(n.message or "", context)}

    async with httpx.AsyncClient() as client:
        resp = await client.post(n.webhook_url, json=body, timeout=10)
        resp.raise_for_status()
    logger.info("Sent Teams notification to %s", n.webhook_url)


async def _send_webhook(n: Webhook, context: Dict[str, str]) -> None:
    url = _render(n.url, context)
    headers = {k: _render(v, context) for k, v in (n.headers or {}).items()}
    body = _render_dict(n.body, context) if n.body else None

    async with httpx.AsyncClient() as client:
        resp = await client.request(n.method, url, json=body, headers=headers, timeout=10)
        resp.raise_for_status()
    logger.info("Sent webhook notification to %s", url)
