"""
Task Notifications API for Flyte 2.0

Send notifications when tasks reach specific execution phases.
Supports Email, Slack, Teams, and custom Webhooks.

Quick Start:
    ```python
    import flyte
    import flyte.notify as notify

    @flyte.task(
        trigger=flyte.Trigger(
            name="daily_report",
            automation=flyte.Cron("0 0 * * *"),
            notify=[
                notify.Email(
                    on_phase="FAILED",
                    recipients=["oncall@example.com"]
                ),
                notify.Slack(
                    on_phase="SUCCEEDED",
                    webhook_url="https://hooks.slack.com/...",
                    message="Daily report completed! {run.url}"
                )
            ]
        )
    )
    def daily_report():
        # Your task logic here
        pass
    ```

Available Notification Types:
    - Email: Send email notifications
    - Slack: Send Slack messages (with optional Block Kit)
    - Teams: Send Microsoft Teams messages (with optional Adaptive Cards)
    - Webhook: Send custom HTTP requests (most flexible)

Supported Phases:
    - "SUCCEEDED": Task completed successfully
    - "FAILED": Task failed
    - "TIMED_OUT": Task timed out
    - "ABORTED": Task was aborted
    - "QUEUED": Task was queued
    - "RUNNING": Task started running

Template Variables:
    All notification messages support template variables:
    - {task.name}: Task name
    - {run.name}: Run ID/name
    - {run.phase}: Current run phase
    - {run.error}: Error message (if failed)
    - {run.duration}: Run duration
    - {run.timestamp}: ISO 8601 timestamp
    - {run.url}: URL to run details page
    - {project}: Flyte project name
    - {domain}: Flyte domain name
"""

from flyte.notify._notifiers import (
    Email,
    Notification,
    Phase,
    Slack,
    Teams,
    Webhook,
)

__all__ = [
    "Email",
    "Notification",
    "Phase",
    "Slack",
    "Teams",
    "Webhook",
]
