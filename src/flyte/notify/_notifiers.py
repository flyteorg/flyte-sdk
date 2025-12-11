"""
Task Notifications API for Flyte 2.0

Supports multiple notification channels (Email, Slack, Teams, PagerDuty, Webhook)
with customizable messages and template variables.

Template Variables:
  - {task.name}: Task name
  - {run.name}: Run ID/name
  - {run.phase}: Current run phase
  - {run.error}: Error message (if failed)
  - {run.duration}: Run duration
  - {run.timestamp}: ISO 8601 timestamp
  - {run.url}: URL to run details page
  - {project}: Flyte project name
  - {domain}: Flyte domain name
  - {inputs.<param>}: Task input parameter
  - {outputs.<result>}: Task output (SUCCEEDED phase only)
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

from typing_extensions import get_args

# Phase type for type safety - matches proto definition
Phase = Literal["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED", "QUEUED", "RUNNING"]
_allowed_phases = set(get_args(Phase))


@dataclass(frozen=True, kw_only=True)
class Notification:
    """Base notification class.

    All notification types must specify phases when they should trigger.
    """

    on_phase: Union[Phase, Tuple[str, ...]]

    def __post_init__(self):
        # Normalize on_phase to a tuple
        if isinstance(self.on_phase, str):
            object.__setattr__(self, "on_phase", (self.on_phase,))
        elif isinstance(self.on_phase, (list, tuple)):
            # If any phase is incorrect, raise error
            configured_phases = set(self.on_phase)
            common_phases = configured_phases.intersection(_allowed_phases)
            if len(common_phases) != len(self.on_phase):
                raise ValueError(f"Phase has to be one of {_allowed_phases}")
            object.__setattr__(self, "on_phase", tuple(self.on_phase))

        if not self.on_phase:
            raise ValueError("At least one phase must be specified")


@dataclass(frozen=True, kw_only=True)
class Email(Notification):
    """Send email notifications.

    Example:
        ```python
        Email(
            on_phase="FAILED",
            recipients=["oncall@example.com"],
            subject="Alert: Task {task.name} failed",
            body="Error: {run.error}\nDetails: {run.url}"
        )
        ```

    Args:
        on_phase: Phase(s) to trigger notification (e.g., "FAILED" or ("FAILED", "TIMED_OUT"))
        recipients: Tuple of email addresses
        subject: Email subject template (supports template variables)
        body: Email body template (supports template variables)
    """

    recipients: Tuple[str, ...]
    subject: str = "Task {task.name} {run.phase}"
    body: str = (
        "Task: {task.name}\n"
        "Run: {run.name}\n"
        "Project/Domain: {project}/{domain}\n"
        "Phase: {run.phase}\n"
        "Duration: {run.duration}\n"
        "Details: {run.url}\n"
    )

    def __post_init__(self):
        super().__post_init__()
        # Normalize recipients to tuple
        if isinstance(self.recipients, list):
            object.__setattr__(self, "recipients", tuple(self.recipients))

        if not self.recipients:
            raise ValueError("At least one recipient must be specified")


@dataclass(frozen=True, kw_only=True)
class Slack(Notification):
    """Send Slack notifications with optional Block Kit formatting.

    Example:
        ```python
        Slack(
            on_phase="FAILED",
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            message="🚨 Task {task.name} failed: {run.error}\n{run.url}",
            channel="#alerts"
        )
        ```

    Args:
        on_phase: Phase(s) to trigger notification
        webhook_url: Slack webhook URL
        message: Simple text message (supports template variables)
        channel: Optional channel override (e.g., "#alerts" or "@username")
        blocks: Optional Slack Block Kit blocks for rich formatting
            (if provided, message is ignored). See: https://api.slack.com/block-kit
    """

    webhook_url: str
    message: Optional[str] = None
    channel: Optional[str] = None
    blocks: Optional[Tuple[Dict[str, Any], ...]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.webhook_url:
            raise ValueError("webhook_url is required")

        # Normalize blocks to tuple if provided as list
        if self.blocks is not None and isinstance(self.blocks, list):
            object.__setattr__(self, "blocks", tuple(self.blocks))

        # Default message
        if self.message is None and self.blocks is None:
            object.__setattr__(
                self,
                "message",
                ("Task `{task.name}` {run.phase}\nRun: {run.name} | {project}/{domain}\n<{run.url}|View Details>"),
            )


@dataclass(frozen=True, kw_only=True)
class Teams(Notification):
    """Send Microsoft Teams notifications with optional Adaptive Cards.

    Example:
        ```python
        Teams(
            on_phase="SUCCEEDED",
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
            title="✅ Task Complete",
            message="Task {task.name} completed in {run.duration}\n[View Details]({run.url})"
        )
        ```

    Args:
        on_phase: Phase(s) to trigger notification
        webhook_url: Microsoft Teams webhook URL
        title: Message card title (supports template variables)
        message: Simple text message (supports template variables)
        card: Optional Adaptive Card for rich formatting
            (if provided, title and message are ignored).
            See: https://adaptivecards.io/designer/
    """

    webhook_url: str
    title: str = "Task {task.name} {run.phase}"
    message: Optional[str] = None
    card: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.webhook_url:
            raise ValueError("webhook_url is required")

        # Default message
        if self.message is None and self.card is None:
            object.__setattr__(
                self,
                "message",
                (
                    "Run: {run.name}\n"
                    "Project/Domain: {project}/{domain}\n"
                    "Phase: {run.phase}\n"
                    "Duration: {run.duration}\n"
                    "[View Details]({run.url})"
                ),
            )


@dataclass(frozen=True, kw_only=True)
class Webhook(Notification):
    """Send custom HTTP webhook notifications (most flexible option).

    Example:
        ```python
        Webhook(
            on_phase="FAILED",
            url="https://api.example.com/alerts",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": "secret"},
            body={
                "event": "task_failed",
                "task": "{task.name}",
                "run": "{run.name}",
                "project": "{project}",
                "domain": "{domain}",
                "error": "{run.error}",
                "url": "{run.url}",
                "inputs": {"data_path": "{inputs.data_path}"}
            }
        )
        ```

    Args:
        on_phase: Phase(s) to trigger notification
        url: Webhook URL (supports template variables)
        method: HTTP method (default: "POST")
        headers: Optional HTTP headers (values support template variables)
        body: Optional request body as dict
            (all string values support template variables recursively)
    """

    url: str
    method: Literal["POST", "PUT", "PATCH", "GET", "DELETE"] = "POST"
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.url:
            raise ValueError("url is required")

        # Default body
        if self.body is None:
            object.__setattr__(
                self,
                "body",
                {
                    "task": "{task.name}",
                    "run": "{run.name}",
                    "project": "{project}",
                    "domain": "{domain}",
                    "phase": "{run.phase}",
                    "url": "{run.url}",
                    "timestamp": "{run.timestamp}",
                },
            )
