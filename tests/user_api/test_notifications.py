"""Unit tests for notification classes."""

import pytest

import flyte.notify as notify
from flyte.models import ActionPhase


class TestEmail:
    """Tests for Email notification class."""

    def test_email_basic(self):
        """Test basic Email notification creation"""
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("oncall@example.com",),
        )

        assert email.on_phase == (ActionPhase.FAILED,)
        assert email.recipients == ("oncall@example.com",)
        assert "Task {task.name} {run.phase}" in email.subject
        assert "{run.url}" in email.body

    def test_email_custom_subject_and_body(self):
        """Test Email notification with custom subject and body"""
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("oncall@example.com",),
            subject="Alert: Task {task.name} failed",
            body="Error: {run.error}\nDetails: {run.url}",
        )

        assert email.subject == "Alert: Task {task.name} failed"
        assert email.body == "Error: {run.error}\nDetails: {run.url}"

    def test_email_multiple_recipients(self):
        """Test Email notification with multiple recipients"""
        email = notify.Email(
            on_phase=ActionPhase.SUCCEEDED,
            recipients=("user1@example.com", "user2@example.com", "user3@example.com"),
        )

        assert len(email.recipients) == 3
        assert "user1@example.com" in email.recipients
        assert "user2@example.com" in email.recipients
        assert "user3@example.com" in email.recipients

    def test_email_recipients_list_to_tuple(self):
        """Test Email notification converts recipients list to tuple"""
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=["oncall@example.com", "backup@example.com"],  # type: ignore
        )

        assert isinstance(email.recipients, tuple)
        assert email.recipients == ("oncall@example.com", "backup@example.com")

    def test_email_multiple_phases(self):
        """Test Email notification with multiple trigger phases"""
        email = notify.Email(
            on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
            recipients=("oncall@example.com",),
        )

        assert email.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT)

    def test_email_no_recipients_error(self):
        """Test Email notification raises error with no recipients"""
        with pytest.raises(ValueError, match="At least one recipient must be specified"):
            notify.Email(
                on_phase=ActionPhase.FAILED,
                recipients=(),
            )

    def test_email_no_phase_error(self):
        """Test Email notification raises error with no phase"""
        with pytest.raises(ValueError, match="At least one phase must be specified"):
            notify.Email(
                on_phase=(),
                recipients=("oncall@example.com",),
            )


class TestSlack:
    """Tests for Slack notification class."""

    def test_slack_basic(self):
        """Test basic Slack notification creation"""
        slack = notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        )

        assert slack.on_phase == (ActionPhase.FAILED,)
        assert slack.webhook_url == "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        assert slack.message is not None  # Default message should be set
        assert slack.blocks is None

    def test_slack_custom_message(self):
        """Test Slack notification with custom message"""
        slack = notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            message="🚨 Task {task.name} failed: {run.error}\n{run.url}",
        )

        assert slack.message == "🚨 Task {task.name} failed: {run.error}\n{run.url}"

    def test_slack_with_blocks(self):
        """Test Slack notification with Block Kit blocks"""
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "🚨 Run Failed"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "*Workflow:*\n{run.name}"},
                    {"type": "mrkdwn", "text": "*Error:*\n{run.error}"},
                ],
            },
        ]

        slack = notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            blocks=blocks,  # type: ignore
        )

        assert isinstance(slack.blocks, tuple)
        assert len(slack.blocks) == 2
        assert slack.blocks[0]["type"] == "header"

    def test_slack_blocks_list_to_tuple(self):
        """Test Slack notification converts blocks list to tuple"""
        slack = notify.Slack(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            blocks=[{"type": "header"}],  # type: ignore
        )

        assert isinstance(slack.blocks, tuple)

    def test_slack_no_webhook_url_error(self):
        """Test Slack notification raises error without webhook_url"""
        with pytest.raises(ValueError, match="webhook_url is required"):
            notify.Slack(
                on_phase=ActionPhase.FAILED,
                webhook_url="",
            )

    def test_slack_default_message_when_none_provided(self):
        """Test Slack notification sets default message when none provided"""
        slack = notify.Slack(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        )

        assert slack.message is not None
        assert "{task.name}" in slack.message
        assert "{run.url}" in slack.message

    def test_slack_multiple_phases(self):
        """Test Slack notification with multiple trigger phases"""
        slack = notify.Slack(
            on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        )

        assert slack.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT)


class TestTeams:
    """Tests for Teams notification class."""

    def test_teams_basic(self):
        """Test basic Teams notification creation"""
        teams = notify.Teams(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
        )

        assert teams.on_phase == (ActionPhase.SUCCEEDED,)
        assert teams.webhook_url == "https://outlook.office.com/webhook/YOUR_WEBHOOK_URL"
        assert teams.title == "Task {task.name} {run.phase}"
        assert teams.message is not None  # Default message should be set
        assert teams.card is None

    def test_teams_custom_title_and_message(self):
        """Test Teams notification with custom title and message"""
        teams = notify.Teams(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
            title="✅ Task Complete",
            message="Task {task.name} completed in {run.duration}\n[View Details]({run.url})",
        )

        assert teams.title == "✅ Task Complete"
        assert teams.message == "Task {task.name} completed in {run.duration}\n[View Details]({run.url})"

    def test_teams_with_adaptive_card(self):
        """Test Teams notification with Adaptive Card"""
        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "Task completed!"},
            ],
        }

        teams = notify.Teams(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
            card=card,
        )

        assert teams.card == card
        assert teams.card["type"] == "AdaptiveCard"

    def test_teams_no_webhook_url_error(self):
        """Test Teams notification raises error without webhook_url"""
        with pytest.raises(ValueError, match="webhook_url is required"):
            notify.Teams(
                on_phase=ActionPhase.SUCCEEDED,
                webhook_url="",
            )

    def test_teams_default_message_when_none_provided(self):
        """Test Teams notification sets default message when none provided"""
        teams = notify.Teams(
            on_phase=ActionPhase.FAILED,
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
        )

        assert teams.message is not None
        assert "{run.name}" in teams.message
        assert "{run.url}" in teams.message

    def test_teams_multiple_phases(self):
        """Test Teams notification with multiple trigger phases"""
        teams = notify.Teams(
            on_phase=(ActionPhase.FAILED, ActionPhase.ABORTED),
            webhook_url="https://outlook.office.com/webhook/YOUR_WEBHOOK_URL",
        )

        assert teams.on_phase == (ActionPhase.FAILED, ActionPhase.ABORTED)


class TestWebhook:
    """Tests for Webhook notification class."""

    def test_webhook_basic(self):
        """Test basic Webhook notification creation"""
        webhook = notify.Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.example.com/alerts",
        )

        assert webhook.on_phase == (ActionPhase.FAILED,)
        assert webhook.url == "https://api.example.com/alerts"
        assert webhook.method == "POST"
        assert webhook.body is not None  # Default body should be set
        assert webhook.headers is None

    def test_webhook_custom_method(self):
        """Test Webhook notification with custom HTTP method"""
        webhook = notify.Webhook(
            on_phase=ActionPhase.SUCCEEDED,
            url="https://api.example.com/alerts",
            method="PUT",
        )

        assert webhook.method == "PUT"

    def test_webhook_with_headers(self):
        """Test Webhook notification with custom headers"""
        webhook = notify.Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.example.com/alerts",
            headers={"Content-Type": "application/json", "X-API-Key": "secret"},
        )

        assert webhook.headers == {"Content-Type": "application/json", "X-API-Key": "secret"}

    def test_webhook_custom_body(self):
        """Test Webhook notification with custom body"""
        custom_body = {
            "event": "task_failed",
            "task": "{task.name}",
            "run": "{run.name}",
            "project": "{project}",
            "domain": "{domain}",
            "error": "{run.error}",
            "url": "{run.url}",
        }

        webhook = notify.Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.example.com/alerts",
            body=custom_body,
        )

        assert webhook.body == custom_body
        assert webhook.body["event"] == "task_failed"
        assert webhook.body["task"] == "{task.name}"

    def test_webhook_default_body(self):
        """Test Webhook notification has default body with template variables"""
        webhook = notify.Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.example.com/alerts",
        )

        assert webhook.body is not None
        assert "{task.name}" in webhook.body.values()
        assert "{run.name}" in webhook.body.values()
        assert "{project}" in webhook.body.values()

    def test_webhook_no_url_error(self):
        """Test Webhook notification raises error without url"""
        with pytest.raises(ValueError, match="url is required"):
            notify.Webhook(
                on_phase=ActionPhase.FAILED,
                url="",
            )

    def test_webhook_all_http_methods(self):
        """Test Webhook notification supports all HTTP methods"""
        methods = ["POST", "PUT", "PATCH", "GET", "DELETE"]

        for method in methods:
            webhook = notify.Webhook(
                on_phase=ActionPhase.FAILED,
                url="https://api.example.com/alerts",
                method=method,  # type: ignore
            )
            assert webhook.method == method

    def test_webhook_multiple_phases(self):
        """Test Webhook notification with multiple trigger phases"""
        webhook = notify.Webhook(
            on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT, ActionPhase.ABORTED),
            url="https://api.example.com/alerts",
        )

        assert webhook.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT, ActionPhase.ABORTED)


class TestNotificationBase:
    """Tests for base Notification class behavior."""

    def test_single_phase_normalization(self):
        """Test that a single phase is normalized to a tuple"""
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("oncall@example.com",),
        )

        assert isinstance(email.on_phase, tuple)
        assert email.on_phase == (ActionPhase.FAILED,)

    def test_list_phase_normalization(self):
        """Test that a list of phases is normalized to a tuple"""
        email = notify.Email(
            on_phase=[ActionPhase.FAILED, ActionPhase.TIMED_OUT],  # type: ignore
            recipients=("oncall@example.com",),
        )

        assert isinstance(email.on_phase, tuple)
        assert email.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT)

    def test_tuple_phase_preserved(self):
        """Test that a tuple of phases is preserved"""
        email = notify.Email(
            on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
            recipients=("oncall@example.com",),
        )

        assert isinstance(email.on_phase, tuple)
        assert email.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT)

    def test_empty_phase_error(self):
        """Test that empty phase tuple raises error"""
        with pytest.raises(ValueError, match="At least one phase must be specified"):
            notify.Email(
                on_phase=(),
                recipients=("oncall@example.com",),
            )

    def test_frozen_dataclass(self):
        """Test that notification instances are frozen"""
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("oncall@example.com",),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            email.recipients = ("new@example.com",)


class TestNamedRule:
    """Tests for NamedRule notification class."""

    def test_named_rule_basic(self):
        rule = notify.NamedRule("oncall-alerts")
        assert rule.name == "oncall-alerts"

    def test_named_rule_empty_name_error(self):
        with pytest.raises(ValueError, match="Rule name must not be empty"):
            notify.NamedRule("")

    def test_named_rule_frozen(self):
        rule = notify.NamedRule("test")
        with pytest.raises(Exception):
            rule.name = "other"


class TestNamedDelivery:
    """Tests for NamedDelivery notification class."""

    def test_named_delivery_basic(self):
        nd = notify.NamedDelivery(on_phase=ActionPhase.FAILED, name="slack-oncall")
        assert nd.name == "slack-oncall"
        assert nd.on_phase == (ActionPhase.FAILED,)

    def test_named_delivery_multiple_phases(self):
        nd = notify.NamedDelivery(
            on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
            name="slack-oncall",
        )
        assert nd.on_phase == (ActionPhase.FAILED, ActionPhase.TIMED_OUT)

    def test_named_delivery_empty_name_error(self):
        with pytest.raises(ValueError, match="Delivery config name must not be empty"):
            notify.NamedDelivery(on_phase=ActionPhase.FAILED, name="")

    def test_named_delivery_unsupported_phase_error(self):
        with pytest.raises(ValueError, match="not supported"):
            notify.NamedDelivery(on_phase=ActionPhase.RUNNING, name="test")

    def test_named_delivery_frozen(self):
        nd = notify.NamedDelivery(on_phase=ActionPhase.FAILED, name="test")
        with pytest.raises(Exception):
            nd.name = "other"


class TestEmailExtended:
    """Tests for Email cc, bcc, and html_body fields."""

    def test_email_cc(self):
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            cc=("cc1@b.com", "cc2@b.com"),
        )
        assert email.cc == ("cc1@b.com", "cc2@b.com")

    def test_email_bcc(self):
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            bcc=("bcc@b.com",),
        )
        assert email.bcc == ("bcc@b.com",)

    def test_email_cc_bcc_default_empty(self):
        email = notify.Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",))
        assert email.cc == ()
        assert email.bcc == ()

    def test_email_cc_list_to_tuple(self):
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            cc=["cc@b.com"],  # type: ignore
        )
        assert isinstance(email.cc, tuple)

    def test_email_bcc_list_to_tuple(self):
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            bcc=["bcc@b.com"],  # type: ignore
        )
        assert isinstance(email.bcc, tuple)

    def test_email_html_body(self):
        email = notify.Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            html_body="<h1>Alert: {task.name}</h1>",
        )
        assert email.html_body == "<h1>Alert: {task.name}</h1>"

    def test_email_html_body_default_none(self):
        email = notify.Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",))
        assert email.html_body is None


class TestWebhookExtendedMethods:
    """Tests for extended HTTP methods on Webhook."""

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "GET", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"])
    def test_all_http_methods(self, method):
        webhook = notify.Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://test",
            method=method,  # type: ignore
        )
        assert webhook.method == method


class TestNotificationIntegration:
    """Integration tests for notifications with multiple types."""

    def test_multiple_notification_types(self):
        """Test creating multiple notification types together"""
        notifications = (
            notify.Slack(
                on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
                webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                message="Task {task.name} failed",
            ),
            notify.Webhook(
                on_phase=ActionPhase.FAILED,
                url="https://api.example.com/alerts",
                headers={"Content-Type": "application/json"},
                body={"task": "{task.name}"},
            ),
            notify.Email(
                on_phase=ActionPhase.SUCCEEDED,
                recipients=("success@example.com",),
                subject="Task {task.name} succeeded",
            ),
        )

        assert len(notifications) == 3
        assert isinstance(notifications[0], notify.Slack)
        assert isinstance(notifications[1], notify.Webhook)
        assert isinstance(notifications[2], notify.Email)

    def test_all_supported_phases(self):
        """Test notifications can use all supported action phases"""
        phases = [
            ActionPhase.SUCCEEDED,
            ActionPhase.FAILED,
            ActionPhase.TIMED_OUT,
            ActionPhase.ABORTED,
        ]

        for phase in phases:
            email = notify.Email(
                on_phase=phase,
                recipients=("test@example.com",),
            )
            assert email.on_phase == (phase,)

    def test_unsupported_phases_raise_error(self):
        """Test that unsupported phases raise ValueError"""
        unsupported_phases = [
            ActionPhase.QUEUED,
            ActionPhase.RUNNING,
        ]

        for phase in unsupported_phases:
            with pytest.raises(ValueError, match="not supported"):
                notify.Email(
                    on_phase=phase,
                    recipients=("test@example.com",),
                )
