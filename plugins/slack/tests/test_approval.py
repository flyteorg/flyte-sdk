"""approval: the block message, the decision handler, and what it ignores."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

from flyteplugins.slack import approval, parse

SECRET = "slack-secret"


def _headers(body: bytes) -> dict:
    ts = str(int(time.time()))
    base = b"v0:" + ts.encode() + b":" + body
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": "v0=" + hmac.new(SECRET.encode(), base, hashlib.sha256).hexdigest(),
    }


def _click(action_id: str, value: dict | None = None) -> object:
    """A block_actions WebhookEvent, built through the real parser."""
    payload = {
        "type": "block_actions",
        "trigger_id": "111.222.333",
        "user": {"id": "U0DECIDER"},
        "channel": {"id": "C1"},
        "container": {"type": "message", "channel_id": "C1", "message_ts": "1.0"},
        "actions": [
            {
                "action_id": action_id,
                "action_ts": str(time.time()),
                **({"value": json.dumps(value)} if value is not None else {}),
            }
        ],
        "response_url": "https://hooks.slack.com/actions/T1/1/a",
    }
    body = urlencode({"payload": json.dumps(payload)}).encode()
    return parse(_headers(body), body)


def test_the_blocks_give_each_option_its_own_action_id():
    built = approval.blocks(
        "Deploy?", ("approve", "reject"), request_id="r-1", response_path="s3://bucket/hitl/r-1.json"
    )
    buttons = built[-1]["elements"]
    action_ids = [b["action_id"] for b in buttons]
    assert action_ids == ["hitl-decision:approve", "hitl-decision:reject"]
    # The value is everything the webhook app needs to answer: no config there.
    assert json.loads(buttons[1]["value"]) == {
        "request_id": "r-1",
        "response_path": "s3://bucket/hitl/r-1.json",
        "choice": "reject",
    }


async def test_a_decision_click_answers_the_event_and_retires_the_buttons(monkeypatch):
    answered, responded = {}, {}

    async def fake_answer(request_id, response_path, choice):
        answered.update(request_id=request_id, response_path=response_path, choice=choice)

    async def fake_respond(response_url, text=None, **kwargs):
        responded.update(response_url=response_url, text=text, **kwargs)

    monkeypatch.setattr(approval, "_answer", fake_answer)
    monkeypatch.setattr(approval.notify, "respond", fake_respond)

    event = _click(
        "hitl-decision:approve",
        {"request_id": "r-1", "response_path": "s3://bucket/hitl/r-1.json", "choice": "approve"},
    )
    result = await approval._on_decision(event)

    assert answered == {"request_id": "r-1", "response_path": "s3://bucket/hitl/r-1.json", "choice": "approve"}
    assert responded["response_url"] == "https://hooks.slack.com/actions/T1/1/a"
    assert responded["replace_original"] is True
    assert "U0DECIDER" in responded["text"]
    assert result == {"request_id": "r-1", "choice": "approve"}


async def test_other_apps_buttons_are_left_alone(monkeypatch):
    async def explode(*args, **kwargs):
        raise AssertionError("must not answer a foreign button")

    monkeypatch.setattr(approval, "_answer", explode)
    assert await approval._on_decision(_click("send_to_customer")) is None


def test_register_subscribes_to_every_block_action():
    class FakeAppEnv:
        def __init__(self):
            self.registered = []

        def on_event(self, pattern):
            def decorator(fn):
                self.registered.append((pattern, fn))
                return fn

            return decorator

    app_env = FakeAppEnv()
    approval.register(app_env)
    [(pattern, handler)] = app_env.registered
    # The catch-all pattern, because each option carries its own action_id;
    # the handler filters on the hitl-decision prefix itself.
    assert pattern == "block_actions"
    assert handler is approval._on_decision
