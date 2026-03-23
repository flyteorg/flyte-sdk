"""Test that notifications are sent during local execution via with_runcontext."""

from unittest.mock import AsyncMock, patch

import pytest

import flyte
import flyte.notify as notify
from flyte.models import ActionPhase

env = flyte.TaskEnvironment(name="notification_test")


@env.task
def succeeding_task(x: int = 1) -> int:
    return x * 2


@env.task
def failing_task(x: int = 1) -> int:
    raise ValueError("something went wrong")


def test_notification_sent_on_success():
    flyte.init_from_config(None)

    slack = notify.Slack(
        on_phase=ActionPhase.SUCCEEDED,
        webhook_url="https://hooks.slack.com/test",
        message="Task {task.name} succeeded",
    )

    with patch("flyte.notify._sender.send_notifications", new_callable=AsyncMock) as mock_send:
        result = flyte.with_runcontext(mode="local", notifications=slack).run(succeeding_task, x=5)

        assert result.outputs()[0] == 10
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["phase"] == ActionPhase.SUCCEEDED
        assert "succeeding_task" in kwargs["task_name"]


def test_notification_sent_on_failure():
    flyte.init_from_config(None)

    slack = notify.Slack(
        on_phase=ActionPhase.FAILED,
        webhook_url="https://hooks.slack.com/test",
        message="Task {task.name} failed: {run.error}",
    )

    with patch("flyte.notify._sender.send_notifications", new_callable=AsyncMock) as mock_send:
        with pytest.raises(Exception, match="something went wrong"):
            flyte.with_runcontext(mode="local", notifications=slack).run(failing_task, x=1)

        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["phase"] == ActionPhase.FAILED
        assert "something went wrong" in kwargs["error"]


def test_multiple_notifications():
    flyte.init_from_config(None)

    notifications = (
        notify.Slack(
            on_phase=ActionPhase.SUCCEEDED,
            webhook_url="https://hooks.slack.com/test",
            message="success",
        ),
        notify.Email(
            on_phase=ActionPhase.SUCCEEDED,
            recipients=("team@example.com",),
        ),
    )

    with patch("flyte.notify._sender.send_notifications", new_callable=AsyncMock) as mock_send:
        result = flyte.with_runcontext(mode="local", notifications=notifications).run(succeeding_task, x=3)

        assert result.outputs()[0] == 6
        mock_send.assert_called_once()
        # The entire tuple is passed through
        args = mock_send.call_args[0]
        assert args[0] == notifications


def test_no_notifications_means_no_send():
    flyte.init_from_config(None)

    with patch("flyte.notify._sender.send_notifications", new_callable=AsyncMock) as mock_send:
        result = flyte.with_runcontext(mode="local").run(succeeding_task, x=2)

        assert result.outputs()[0] == 4
        mock_send.assert_not_called()
