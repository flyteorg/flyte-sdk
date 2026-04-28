"""Unit tests for notifications_serde — proto serialization of notification objects."""

import json

import pytest

pytest.importorskip("flyteidl2.notification", reason="flyteidl2 notification protos not available in this build")

from flyteidl2.common import phase_pb2
from flyteidl2.notification import definition_pb2

from flyte._internal.runtime.notifications_serde import (
    _to_delivery_config_template,
    _to_inline_rule,
    _to_inline_rule_list,
    resolve_notification_settings,
)
from flyte.models import ActionPhase
from flyte.notify import Email, NamedDelivery, NamedRule, Slack, Teams, Webhook


class TestResolveNotificationSettings:
    def test_named_rule_returns_name(self):
        name, rules = resolve_notification_settings(NamedRule("oncall-alerts"))
        assert name == "oncall-alerts"
        assert rules is None

    def test_single_notification_returns_inline_rules(self):
        name, rules = resolve_notification_settings(
            Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi")
        )
        assert name is None
        assert rules is not None
        assert len(rules.rules) == 1

    def test_tuple_of_notifications_returns_inline_rules(self):
        name, rules = resolve_notification_settings(
            (
                Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi"),
                Email(on_phase=ActionPhase.SUCCEEDED, recipients=("a@b.com",)),
            )
        )
        assert name is None
        assert len(rules.rules) == 2


class TestToInlineRuleList:
    def test_single_notification_wrapped(self):
        rule_list = _to_inline_rule_list(Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi"))
        assert len(rule_list.rules) == 1

    def test_multiple_notifications(self):
        rule_list = _to_inline_rule_list(
            (
                Slack(on_phase=ActionPhase.FAILED, webhook_url="https://slack"),
                Email(on_phase=ActionPhase.SUCCEEDED, recipients=("a@b.com",)),
                Webhook(on_phase=ActionPhase.ABORTED, url="https://hook"),
            )
        )
        assert len(rule_list.rules) == 3


class TestToInlineRule:
    def test_phases_mapped_correctly(self):
        rule = _to_inline_rule(
            Slack(on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT), webhook_url="https://test", message="hi")
        )
        assert list(rule.on_phases) == [phase_pb2.ACTION_PHASE_FAILED, phase_pb2.ACTION_PHASE_TIMED_OUT]

    def test_single_phase(self):
        rule = _to_inline_rule(Email(on_phase=ActionPhase.SUCCEEDED, recipients=("a@b.com",)))
        assert list(rule.on_phases) == [phase_pb2.ACTION_PHASE_SUCCEEDED]

    def test_named_delivery_uses_config_name(self):
        rule = _to_inline_rule(NamedDelivery(on_phase=ActionPhase.FAILED, name="slack-oncall"))
        assert rule.delivery_config_name == "slack-oncall"
        assert rule.WhichOneof("delivery") == "delivery_config_name"

    def test_inline_notification_uses_template(self):
        rule = _to_inline_rule(Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi"))
        assert rule.WhichOneof("delivery") == "delivery_template"
        assert rule.delivery_template.HasField("webhook")


class TestEmailTemplate:
    def test_basic_email(self):
        tmpl = _to_delivery_config_template(Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",)))
        assert tmpl.HasField("email")
        email = tmpl.email
        assert len(email.to) == 1
        assert email.to[0].address == "a@b.com"
        assert email.inline.subject == "Task {task.name} {run.phase}"
        assert "{task.name}" in email.inline.text_template

    def test_multiple_recipients(self):
        tmpl = _to_delivery_config_template(Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com", "c@d.com")))
        assert len(tmpl.email.to) == 2
        assert tmpl.email.to[0].address == "a@b.com"
        assert tmpl.email.to[1].address == "c@d.com"

    def test_cc_and_bcc(self):
        tmpl = _to_delivery_config_template(
            Email(
                on_phase=ActionPhase.FAILED,
                recipients=("a@b.com",),
                cc=("cc@b.com",),
                bcc=("bcc@b.com",),
            )
        )
        assert len(tmpl.email.cc) == 1
        assert tmpl.email.cc[0].address == "cc@b.com"
        assert len(tmpl.email.bcc) == 1
        assert tmpl.email.bcc[0].address == "bcc@b.com"

    def test_empty_cc_bcc_by_default(self):
        tmpl = _to_delivery_config_template(Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",)))
        assert len(tmpl.email.cc) == 0
        assert len(tmpl.email.bcc) == 0

    def test_html_body(self):
        tmpl = _to_delivery_config_template(
            Email(
                on_phase=ActionPhase.FAILED,
                recipients=("a@b.com",),
                html_body="<h1>{task.name}</h1>",
            )
        )
        assert tmpl.email.inline.html_template == "<h1>{task.name}</h1>"

    def test_no_html_body(self):
        tmpl = _to_delivery_config_template(Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",)))
        assert tmpl.email.inline.html_template == ""

    def test_custom_subject_and_body(self):
        tmpl = _to_delivery_config_template(
            Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",), subject="Alert!", body="Details here")
        )
        assert tmpl.email.inline.subject == "Alert!"
        assert tmpl.email.inline.text_template == "Details here"


class TestSlackTemplate:
    def test_slack_message(self):
        tmpl = _to_delivery_config_template(
            Slack(on_phase=ActionPhase.FAILED, webhook_url="https://hooks.slack.com/test", message="alert!")
        )
        assert tmpl.HasField("webhook")
        wh = tmpl.webhook
        assert wh.url == "https://hooks.slack.com/test"
        assert wh.method == definition_pb2.HTTP_METHOD_POST
        assert wh.headers["Content-Type"] == "application/json"
        body = json.loads(wh.body_template)
        assert body == {"text": "alert!"}

    def test_slack_blocks(self):
        blocks = ({"type": "header", "text": {"type": "plain_text", "text": "hi"}},)
        tmpl = _to_delivery_config_template(
            Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", blocks=blocks)
        )
        body = json.loads(tmpl.webhook.body_template)
        assert "blocks" in body
        assert body["blocks"][0]["type"] == "header"

    def test_slack_default_message(self):
        tmpl = _to_delivery_config_template(Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test"))
        body = json.loads(tmpl.webhook.body_template)
        assert "text" in body
        assert "{task.name}" in body["text"]


class TestTeamsTemplate:
    def test_teams_message(self):
        tmpl = _to_delivery_config_template(
            Teams(on_phase=ActionPhase.FAILED, webhook_url="https://teams.test", message="alert!")
        )
        wh = tmpl.webhook
        assert wh.url == "https://teams.test"
        body = json.loads(wh.body_template)
        assert body["text"] == "alert!"
        assert "title" in body

    def test_teams_card(self):
        card = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "hi"}]}
        tmpl = _to_delivery_config_template(
            Teams(on_phase=ActionPhase.FAILED, webhook_url="https://teams.test", card=card)
        )
        body = json.loads(tmpl.webhook.body_template)
        assert body["type"] == "AdaptiveCard"

    def test_teams_default_message(self):
        tmpl = _to_delivery_config_template(Teams(on_phase=ActionPhase.FAILED, webhook_url="https://teams.test"))
        body = json.loads(tmpl.webhook.body_template)
        assert "{run.name}" in body["text"]


class TestWebhookTemplate:
    def test_webhook_basic(self):
        tmpl = _to_delivery_config_template(Webhook(on_phase=ActionPhase.FAILED, url="https://api.test/hook"))
        wh = tmpl.webhook
        assert wh.url == "https://api.test/hook"
        assert wh.method == definition_pb2.HTTP_METHOD_POST

    def test_webhook_custom_method(self):
        tmpl = _to_delivery_config_template(Webhook(on_phase=ActionPhase.FAILED, url="https://test", method="PUT"))
        assert tmpl.webhook.method == definition_pb2.HTTP_METHOD_PUT

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("GET", definition_pb2.HTTP_METHOD_GET),
            ("HEAD", definition_pb2.HTTP_METHOD_HEAD),
            ("POST", definition_pb2.HTTP_METHOD_POST),
            ("PUT", definition_pb2.HTTP_METHOD_PUT),
            ("DELETE", definition_pb2.HTTP_METHOD_DELETE),
            ("PATCH", definition_pb2.HTTP_METHOD_PATCH),
            ("OPTIONS", definition_pb2.HTTP_METHOD_OPTIONS),
            ("TRACE", definition_pb2.HTTP_METHOD_TRACE),
            ("CONNECT", definition_pb2.HTTP_METHOD_CONNECT),
        ],
    )
    def test_all_http_methods(self, method, expected):
        tmpl = _to_delivery_config_template(
            Webhook(on_phase=ActionPhase.FAILED, url="https://test", method=method)  # type: ignore[arg-type]
        )
        assert tmpl.webhook.method == expected

    def test_webhook_headers(self):
        tmpl = _to_delivery_config_template(
            Webhook(
                on_phase=ActionPhase.FAILED,
                url="https://test",
                headers={"X-Key": "secret", "Accept": "application/json"},
            )
        )
        assert tmpl.webhook.headers["X-Key"] == "secret"
        assert tmpl.webhook.headers["Accept"] == "application/json"

    def test_webhook_custom_body(self):
        tmpl = _to_delivery_config_template(
            Webhook(on_phase=ActionPhase.FAILED, url="https://test", body={"event": "fail", "task": "{task.name}"})
        )
        body = json.loads(tmpl.webhook.body_template)
        assert body == {"event": "fail", "task": "{task.name}"}

    def test_webhook_no_body(self):
        tmpl = _to_delivery_config_template(Webhook(on_phase=ActionPhase.FAILED, url="https://test", body={}))
        assert tmpl.webhook.body_template == "{}"

    def test_webhook_empty_headers(self):
        tmpl = _to_delivery_config_template(Webhook(on_phase=ActionPhase.FAILED, url="https://test"))
        assert dict(tmpl.webhook.headers) == {}


class TestUnsupportedType:
    def test_unsupported_notification_type_raises(self):
        from flyte.notify._notifiers import Notification

        # Create a bare Notification (not a concrete subclass)
        n = Notification(on_phase=ActionPhase.FAILED)
        with pytest.raises(TypeError, match="Unsupported notification type"):
            _to_delivery_config_template(n)
