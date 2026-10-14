"""approval: the block message, the decision handler, and what it ignores."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

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


#: What a button carries: everything needed to find the parked condition.
DECISION = {"condition": "slack-approval", "run": "r-1", "action": "a-1", "choice": "approve"}


def test_the_blocks_give_each_option_its_own_action_id():
    built = approval.blocks(
        "Deploy?", ("approve", "reject"), condition="slack-approval", run_name="r-1", action_name="a-1"
    )
    buttons = built[-1]["elements"]
    action_ids = [b["action_id"] for b in buttons]
    assert action_ids == ["flyte-condition:approve", "flyte-condition:reject"]
    # The value is everything the webhook app needs to signal: no config there.
    assert json.loads(buttons[1]["value"]) == {
        "condition": "slack-approval",
        "run": "r-1",
        "action": "a-1",
        "choice": "reject",
    }


async def test_a_decision_click_signals_the_condition_and_retires_the_buttons(monkeypatch):
    answered, responded = {}, {}

    async def fake_answer(condition, run_name, action_name, choice):
        answered.update(condition=condition, run=run_name, action=action_name, choice=choice)

    async def fake_respond(response_url, text=None, **kwargs):
        responded.update(response_url=response_url, text=text, **kwargs)

    monkeypatch.setattr(approval, "_answer", fake_answer)
    monkeypatch.setattr(approval.notify, "respond", fake_respond)

    result = await approval._on_decision(_click("flyte-condition:approve", DECISION))

    assert answered == {"condition": "slack-approval", "run": "r-1", "action": "a-1", "choice": "approve"}
    assert responded["response_url"] == "https://hooks.slack.com/actions/T1/1/a"
    assert responded["replace_original"] is True
    assert "U0DECIDER" in responded["text"]
    assert result == {"condition": "slack-approval", "run": "r-1", "choice": "approve"}


async def test_other_apps_buttons_are_left_alone(monkeypatch):
    async def explode(*args, **kwargs):
        raise AssertionError("must not signal a foreign button")

    monkeypatch.setattr(approval, "_answer", explode)
    assert await approval._on_decision(_click("send_to_customer")) is None


async def test_signaling_looks_the_condition_up_by_run_and_action(monkeypatch):
    """The button's value is the whole lookup: name, run, and parent action."""
    import flyte.remote as remote

    signaled = {}
    looked_up = {}

    class FakeCondition:
        async def signal_aio(self, payload):
            signaled["payload"] = payload

        signal = property(lambda self: SimpleNamespace(aio=self.signal_aio))

    async def fake_get(name, run_name=None, action_name=None):
        looked_up.update(name=name, run_name=run_name, action_name=action_name)
        return FakeCondition()

    monkeypatch.setattr(approval, "_ensure_initialized", _noop)
    monkeypatch.setattr(remote.Condition, "get", SimpleNamespace(aio=fake_get), raising=False)

    await approval._answer("slack-approval", "r-1", "a-1", "approve")
    assert looked_up == {"name": "slack-approval", "run_name": "r-1", "action_name": "a-1"}
    assert signaled["payload"] == "approve"


async def test_a_condition_that_is_gone_says_so(monkeypatch):
    """Already signaled, timed out, or the run ended — the click must not fail silently."""
    import flyte.remote as remote

    async def fake_get(name, run_name=None, action_name=None):
        return None

    monkeypatch.setattr(approval, "_ensure_initialized", _noop)
    monkeypatch.setattr(remote.Condition, "get", SimpleNamespace(aio=fake_get), raising=False)

    with pytest.raises(RuntimeError, match="not found"):
        await approval._answer("gone", "r-1", "a-1", "approve")


async def _noop() -> None:
    return None


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
    # the handler filters on the flyte-condition prefix itself.
    assert pattern == "block_actions"
    assert handler is approval._on_decision
