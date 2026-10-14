"""ClickUp-specific normalization, beyond conformance."""

from __future__ import annotations

import hashlib
import hmac
import json

from flyteplugins.clickup import events, parse, verify

SECRET = "clickup-secret"


def _parse(payload: dict):
    body = json.dumps(payload).encode()
    return parse({}, body)


def test_the_event_name_is_the_qualified_type():
    assert _parse({"event": "taskStatusUpdated", "task_id": "t1"}).qualified_type == events.Task.STATUS_UPDATED


def test_the_list_id_falls_back_to_the_nested_task():
    """Task-scoped events carry it only there; a scopes allowlist needs it."""
    event = _parse({"event": "taskCreated", "task_id": "t1", "task": {"id": "t1", "list": {"id": "l7"}}})
    assert event.scope == "l7"


def test_a_top_level_list_id_wins():
    event = _parse({"event": "listCreated", "list_id": "l1", "task": {"list": {"id": "l7"}}})
    assert event.scope == "l1"


def test_later_updates_to_one_task_get_their_own_keys():
    def update(timestamp: int):
        return _parse({"event": "taskUpdated", "task_id": "t1", "timestamp": timestamp})

    assert update(1700000000000).dedupe_key() != update(1700000009999).dedupe_key()


def test_verify_reads_the_header_clickup_actually_sends():
    """ClickUp sends a bare `X-Signature`, not a product-prefixed name.

    See https://developer.clickup.com/docs/webhooksignature.

    Asserted on the literal wire name because nothing else can: the conformance
    round trip signs `SAMPLE_DELIVERY` with whatever header this module reads, so
    a wrong name verifies against itself while every genuine delivery 401s.
    """
    body = b'{"event": "taskCreated", "task_id": "t1"}'
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

    assert verify(body, {"X-Signature": digest}, SECRET) is True
    assert verify(body, {"X-Clickup-Signature": digest}, SECRET) is False
