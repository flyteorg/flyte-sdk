"""
Task Notifications API for Flyte 2.0

Supports multiple notification channels (Email, Slack, Teams, Webhook)
with customizable messages and template variables.

Template Variables:
    - {{.Run.Project}}: Run project name
    - {{.Run.Domain}}: Run domain name
    - {{.Run.Name}}: Run ID/name
    - {{.Phase}}: Terminal run phase
    - {{.Error}}: Error message (if failed)
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

from flyte.models import ActionPhase

_SUPPORTED_PHASES = {ActionPhase.FAILED, ActionPhase.SUCCEEDED, ActionPhase.ABORTED, ActionPhase.TIMED_OUT}


@dataclass(frozen=True)
class NamedRule:
    """Reference a pre-defined notification rule by name.

    Use this when your Flyte admin has configured a named notification rule
    that you want to apply to your runs. Named rules define both the phases
    to monitor and the delivery channels to use.

    Example:
        ```python
        # As a trigger notification
        flyte.Trigger(
            name="hourly",
            automation=flyte.Cron("0 * * * *"),
            notifications=flyte.notify.NamedRule("oncall-alerts"),
        )

        # In with_runcontext
        flyte.with_runcontext(
            notifications=flyte.notify.NamedRule("oncall-alerts"),
        ).run(my_task, x=1)
        ```

    Args:
        name: The name of the pre-defined rule (scoped to project/domain).
    """

    name: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("Rule name must not be empty")


@dataclass(frozen=True, kw_only=True)
class Notification:
    """Base notification class.

    All notification types must specify phases when they should trigger.
    """

    on_phase: Union[ActionPhase, Tuple[ActionPhase, ...]]

    def __post_init__(self):
        # Normalize on_phase to a tuple
        if isinstance(self.on_phase, str):
            object.__setattr__(self, "on_phase", (self.on_phase,))
        elif isinstance(self.on_phase, (list, tuple)):
            # If any phase is incorrect, raise error
            object.__setattr__(self, "on_phase", tuple(self.on_phase))

        if not self.on_phase:
            raise ValueError("At least one phase must be specified")

        for p in self.on_phase:
            if p not in _SUPPORTED_PHASES:
                raise ValueError(f"Notification on phase {p} is not supported.")


@dataclass(frozen=True, kw_only=True)
class NamedDelivery(Notification):
    """Use a pre-configured delivery channel by name.

    Use this when your Flyte admin has configured a named delivery config
    (e.g., a shared Slack webhook or email list) that you want to reference
    without specifying the delivery details inline.

    Example:
        ```python
        flyte.notify.NamedDelivery(
            on_phase=ActionPhase.FAILED,
            name="slack-oncall",
        )

        # Combine with inline notifications
        notifications=(
            flyte.notify.NamedDelivery(on_phase=ActionPhase.FAILED, name="slack-oncall"),
            flyte.notify.Email(
                on_phase=ActionPhase.SUCCEEDED,
                recipients=["team@example.com"],
            ),
        )
        ```

    Args:
        on_phase: ActionPhase(s) to trigger notification.
        name: The name of the pre-configured delivery config (scoped to project/domain).
    """

    name: str

    def __post_init__(self):
        super().__post_init__()
        if not self.name:
            raise ValueError("Delivery config name must not be empty")


@dataclass(frozen=True, kw_only=True)
class Email(Notification):
    """Send email notifications.

    Example:
        ```python
        Email(
            on_phase=ActionPhase.FAILED,
            recipients=["oncall@example.com"],
            subject="Alert: Task {task.name} failed",
            body="Error: {{.Error}}"
        )
        ```

    Args:
        on_phase: ActionPhase(s) to trigger notification
            (e.g., ActionPhase.FAILED or (ActionPhase.FAILED, ActionPhase.TIMED_OUT))
        recipients: Email addresses for the "to" field.
        cc: Optional email addresses for the "cc" field.
        bcc: Optional email addresses for the "bcc" field.
        subject: Email subject template (supports template variables).
        body: Plain text body template (supports template variables).
        html_body: Optional HTML body template (supports template variables).
            When provided, the email is sent as multipart with both plain text and HTML.
    """

    recipients: Tuple[str, ...]
    cc: Tuple[str, ...] = ()
    bcc: Tuple[str, ...] = ()
    subject: str = "Run {{.Run.Name}} {{.Phase}}"
    body: str = (
        "Run: {{.Run.Name}}\nProject/Domain: {{.Run.Project}}/{{.Run.Domain}}\nPhase: {{.Phase}}\nError: {{.Error}}\n"
    )
    html_body: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Normalize recipients to tuple
        if isinstance(self.recipients, list):
            object.__setattr__(self, "recipients", tuple(self.recipients))
        if isinstance(self.cc, list):
            object.__setattr__(self, "cc", tuple(self.cc))
        if isinstance(self.bcc, list):
            object.__setattr__(self, "bcc", tuple(self.bcc))

        if not self.recipients:
            raise ValueError("At least one recipient must be specified")


@dataclass(frozen=True, kw_only=True)
class Slack(Notification):
    """Send Slack notifications with optional Block Kit formatting.

    Example:
        ```python
        Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            message="🚨 Run {{.Run.Name}} failed: {{.Error}}",
        )
        ```

    Args:
        on_phase:ActionPhase(s) to trigger notification
        webhook_url: Slack webhook URL
        message: Simple text message (supports template variables)
        blocks: Optional Slack Block Kit blocks for rich formatting
            (if provided, message is ignored). See: https://api.slack.com/block-kit
    """

    webhook_url: str
    message: Optional[str] = None
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
                ("Run: {{.Run.Name}} finished with phase {{.Phase}}\n | {{.Run.Project}}/{{.Run.Domain}}\n"),
            )


@dataclass(frozen=True, kw_only=True)
class Teams(Notification):
    """Send Microsoft Teams notifications with optional Adaptive Cards.

    Example:
        ```python
        Teams(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
            title="✅ Task Complete",
            message="Run {{.Run.Name}} completed"
        )
        ```

    Args:
        on_phase:ActionPhase(s) to trigger notification
        webhook_url: Microsoft Teams webhook URL
        title: Message card title (supports template variables)
        message: Simple text message (supports template variables)
        card: Optional Adaptive Card for rich formatting
            (if provided, title and message are ignored).
            See: https://adaptivecards.io/designer/
    """

    webhook_url: str
    title: str = "Run {{.Run.Name}} {{.Phase}}"
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
                    "Run: {{.Run.Name}}\n"
                    "Project/Domain: {{.Run.Project}}/{{.Run.Domain}}\n"
                    "Phase: {{.Phase}}\n"
                    "Error: {{.Error}}\n"
                ),
            )


@dataclass(frozen=True, kw_only=True)
class Webhook(Notification):
    """Send custom HTTP webhook notifications (most flexible option).

    Example:
        ```python
        Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.example.com/alerts",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": "secret"},
            body={
                "phase": "{{.Phase}}",
                "run": "{{.Run.Name}}",
                "project": "{{.Run.Project}}",
                "domain": "{{.Run.Domain}}",
                "error": "{{.Error}}",
            }
        )
        ```

    Args:
        on_phase:ActionPhase(s) to trigger notification
        url: Webhook URL (supports template variables)
        method: HTTP method (default: "POST")
        headers: Optional HTTP headers (values support template variables)
        body: Optional request body as dict
            (all string values support template variables recursively)
    """

    url: str
    method: Literal["POST", "PUT", "PATCH", "GET", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"] = "POST"
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
                    "run": "{{.Run.Name}}",
                    "project": "{{.Run.Project}}",
                    "domain": "{{.Run.Domain}}",
                    "phase": "{{.Phase}}",
                    "error": "{{.Error}}",
                },
            )
