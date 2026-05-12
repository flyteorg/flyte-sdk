"""Unit tests for the local notification sender."""

import smtplib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyte.models import ActionPhase
from flyte.notify import Email, NamedDelivery, Slack, Teams, Webhook
from flyte.notify._sender import (
    _build_context,
    _render,
    _render_dict,
    send_notifications,
)


class TestRender:
    def test_simple_variable(self):
        assert _render("{task.name} failed", {"task.name": "my_task"}) == "my_task failed"

    def test_multiple_variables(self):
        ctx = {"task.name": "t", "run.name": "r", "run.phase": "failed"}
        assert _render("{task.name}/{run.name} is {run.phase}", ctx) == "t/r is failed"

    def test_no_variables(self):
        assert _render("plain text", {"task.name": "t"}) == "plain text"

    def test_missing_variable_left_as_is(self):
        assert _render("{unknown.var}", {"task.name": "t"}) == "{unknown.var}"

    def test_empty_string(self):
        assert _render("", {"task.name": "t"}) == ""


class TestRenderDict:
    def test_flat_dict(self):
        ctx = {"task.name": "t", "run.name": "r"}
        result = _render_dict({"task": "{task.name}", "run": "{run.name}"}, ctx)
        assert result == {"task": "t", "run": "r"}

    def test_nested_dict(self):
        ctx = {"task.name": "t"}
        result = _render_dict({"outer": {"inner": "{task.name}"}}, ctx)
        assert result == {"outer": {"inner": "t"}}

    def test_list_values(self):
        ctx = {"task.name": "t"}
        result = _render_dict({"items": ["{task.name}", "literal"]}, ctx)
        assert result == {"items": ["t", "literal"]}

    def test_list_of_dicts(self):
        ctx = {"task.name": "t"}
        result = _render_dict({"items": [{"name": "{task.name}"}]}, ctx)
        assert result == {"items": [{"name": "t"}]}

    def test_non_string_values_preserved(self):
        ctx = {"task.name": "t"}
        result = _render_dict({"count": 42, "flag": True, "name": "{task.name}"}, ctx)
        assert result == {"count": 42, "flag": True, "name": "t"}


class TestBuildContext:
    def test_all_fields(self):
        ctx = _build_context(
            task_name="my_task",
            run_name="run-123",
            phase=ActionPhase.FAILED,
            error="OOM",
            duration="5m",
            project="proj",
            domain="dev",
        )
        assert ctx["task.name"] == "my_task"
        assert ctx["run.name"] == "run-123"
        assert ctx["run.phase"] == "failed"
        assert ctx["run.error"] == "OOM"
        assert ctx["run.duration"] == "5m"
        assert ctx["project"] == "proj"
        assert ctx["domain"] == "dev"

    def test_defaults(self):
        ctx = _build_context(task_name="t", run_name="r", phase=ActionPhase.SUCCEEDED)
        assert ctx["run.error"] == ""
        assert ctx["run.duration"] == ""
        assert ctx["project"] == ""
        assert ctx["domain"] == ""


class TestSendNotificationsPhaseFiltering:
    @pytest.mark.asyncio
    async def test_matching_phase_sends(self):
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock) as mock:
            n = Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi")
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_matching_phase_skips(self):
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock) as mock:
            n = Slack(on_phase=ActionPhase.SUCCEEDED, webhook_url="https://test", message="hi")
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_notifications_filtered(self):
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock) as mock:
            notifications = (
                Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="fail"),
                Email(on_phase=ActionPhase.SUCCEEDED, recipients=("a@b.com",)),
                Webhook(on_phase=ActionPhase.FAILED, url="https://hook"),
            )
            await send_notifications(notifications, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            assert mock.call_count == 2  # Slack + Webhook, not Email

    @pytest.mark.asyncio
    async def test_multi_phase_notification_matches(self):
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock) as mock:
            n = Slack(
                on_phase=(ActionPhase.FAILED, ActionPhase.TIMED_OUT),
                webhook_url="https://test",
                message="hi",
            )
            await send_notifications(n, phase=ActionPhase.TIMED_OUT, task_name="t", run_name="r")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_notification_not_tuple(self):
        """Test passing a single notification (not wrapped in tuple)."""
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock) as mock:
            n = Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",))
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            mock.assert_called_once()


class TestSendNotificationsErrorHandling:
    @pytest.mark.asyncio
    async def test_send_failure_does_not_raise(self):
        with patch("flyte.notify._sender._send_one", new_callable=AsyncMock, side_effect=Exception("boom")):
            n = Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi")
            # Should not raise
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")

    @pytest.mark.asyncio
    async def test_one_failure_does_not_block_others(self):
        call_order = []

        async def side_effect(n, ctx):
            call_order.append(type(n).__name__)
            if isinstance(n, Slack):
                raise Exception("slack down")

        with patch("flyte.notify._sender._send_one", side_effect=side_effect):
            notifications = (
                Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", message="hi"),
                Webhook(on_phase=ActionPhase.FAILED, url="https://hook"),
            )
            await send_notifications(notifications, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            assert call_order == ["Slack", "Webhook"]


class TestSendNamedNotifications:
    @pytest.mark.asyncio
    async def test_named_delivery_skipped_locally(self):
        """NamedDelivery should be silently skipped in local mode (no HTTP call made)."""
        n = NamedDelivery(on_phase=ActionPhase.FAILED, name="slack-oncall")
        with patch("flyte.notify._sender.httpx.AsyncClient") as mock_client_cls:
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")
            mock_client_cls.assert_not_called()


class TestSendSlack:
    @pytest.mark.asyncio
    async def test_slack_posts_message(self):
        n = Slack(on_phase=ActionPhase.FAILED, webhook_url="https://hooks.slack.com/test", message="{task.name} failed")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("flyte.notify._sender.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="my_task", run_name="r")

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://hooks.slack.com/test"
            assert call_args[1]["json"] == {"text": "my_task failed"}

    @pytest.mark.asyncio
    async def test_slack_posts_blocks(self):
        blocks = ({"type": "section", "text": {"type": "mrkdwn", "text": "*Task:* {task.name}"}},)
        n = Slack(on_phase=ActionPhase.FAILED, webhook_url="https://test", blocks=blocks)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("flyte.notify._sender.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="my_task", run_name="r")

            body = mock_client.post.call_args[1]["json"]
            assert "blocks" in body
            assert body["blocks"][0]["text"]["text"] == "*Task:* my_task"


class TestSendTeams:
    @pytest.mark.asyncio
    async def test_teams_posts_message(self):
        n = Teams(on_phase=ActionPhase.FAILED, webhook_url="https://teams.test", message="{task.name} failed")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("flyte.notify._sender.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="my_task", run_name="r")

            body = mock_client.post.call_args[1]["json"]
            assert body["text"] == "my_task failed"


class TestSendWebhook:
    @pytest.mark.asyncio
    async def test_webhook_sends_request(self):
        n = Webhook(
            on_phase=ActionPhase.FAILED,
            url="https://api.test/{project}/alerts",
            method="PUT",
            headers={"X-Key": "{domain}"},
            body={"task": "{task.name}"},
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("flyte.notify._sender.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await send_notifications(
                n, phase=ActionPhase.FAILED, task_name="t", run_name="r", project="proj", domain="dev"
            )

            call_args = mock_client.request.call_args
            assert call_args[0] == ("PUT", "https://api.test/proj/alerts")
            assert call_args[1]["headers"] == {"X-Key": "dev"}
            assert call_args[1]["json"] == {"task": "t"}


class TestSendEmail:
    @pytest.mark.asyncio
    async def test_email_sends_via_smtp(self):
        n = Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",), subject="{task.name} failed")

        mock_smtp = MagicMock()
        with patch("flyte.notify._sender.smtplib.SMTP", return_value=mock_smtp) as smtp_cls:
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=False)

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="my_task", run_name="r")

            smtp_cls.assert_called_once_with("localhost", 25, timeout=5)
            mock_smtp.sendmail.assert_called_once()
            call_args = mock_smtp.sendmail.call_args
            assert call_args[0][1] == ["a@b.com"]  # recipients
            assert "my_task failed" in call_args[0][2]  # message contains rendered subject

    @pytest.mark.asyncio
    async def test_email_includes_cc_and_bcc_recipients(self):
        n = Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            cc=("cc@b.com",),
            bcc=("bcc@b.com",),
        )

        mock_smtp = MagicMock()
        with patch("flyte.notify._sender.smtplib.SMTP", return_value=mock_smtp):
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=False)

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")

            all_recipients = mock_smtp.sendmail.call_args[0][1]
            assert "a@b.com" in all_recipients
            assert "cc@b.com" in all_recipients
            assert "bcc@b.com" in all_recipients

    @pytest.mark.asyncio
    async def test_email_smtp_unavailable_does_not_raise(self):
        """When SMTP is unavailable, email sending logs a warning but doesn't crash."""
        n = Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",))

        with patch("flyte.notify._sender.smtplib.SMTP", side_effect=ConnectionRefusedError):
            # Should not raise
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")

    @pytest.mark.asyncio
    async def test_email_smtp_error_does_not_raise(self):
        n = Email(on_phase=ActionPhase.FAILED, recipients=("a@b.com",))

        with patch("flyte.notify._sender.smtplib.SMTP", side_effect=smtplib.SMTPException("fail")):
            # Should not raise
            await send_notifications(n, phase=ActionPhase.FAILED, task_name="t", run_name="r")

    @pytest.mark.asyncio
    async def test_email_html_body(self):
        n = Email(
            on_phase=ActionPhase.FAILED,
            recipients=("a@b.com",),
            html_body="<b>{task.name}</b>",
        )

        mock_smtp = MagicMock()
        with patch("flyte.notify._sender.smtplib.SMTP", return_value=mock_smtp):
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=False)

            await send_notifications(n, phase=ActionPhase.FAILED, task_name="my_task", run_name="r")

            message_str = mock_smtp.sendmail.call_args[0][2]
            assert "<b>my_task</b>" in message_str
            assert "multipart/alternative" in message_str
