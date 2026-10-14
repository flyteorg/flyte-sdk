"""Tests for the normalized event and its dedupe key."""

from __future__ import annotations

from flyte.extras.webhooks import WebhookEvent


def _event(**kwargs) -> WebhookEvent:
    base = {"provider": "stub", "event_type": "thing", "action": "created", "resource_id": "r1"}
    return WebhookEvent(**{**base, **kwargs})


def test_qualified_type_joins_type_and_action_only_when_there_is_one():
    assert _event().qualified_type == "thing.created"
    assert _event(action=None).qualified_type == "thing"


def test_the_same_delivery_dedupes():
    assert _event().dedupe_key() == _event().dedupe_key()


def test_distinct_resources_get_distinct_keys():
    assert _event(resource_id="r1").dedupe_key() != _event(resource_id="r2").dedupe_key()


def test_a_later_change_to_one_resource_gets_its_own_key():
    """Without the timestamp, every update after the first would never launch."""
    first = _event(action="updated", occurred_at="2024-01-01T00:00:00Z")
    later = _event(action="updated", occurred_at="2024-06-01T00:00:00Z")
    assert first.dedupe_key() != later.dedupe_key()


def test_events_without_a_resource_fall_back_to_the_delivery_id():
    a = _event(resource_id=None, delivery_id="d1")
    b = _event(resource_id=None, delivery_id="d2")
    assert a.dedupe_key() != b.dedupe_key()


def test_keys_never_collide_across_providers():
    """Two products can use the same resource id for unrelated things."""
    a = _event(provider="github")
    b = _event(provider="linear")
    assert a.dedupe_key() != b.dedupe_key()


def test_distinct_event_types_on_one_resource_get_distinct_keys():
    assert _event(action="created").dedupe_key() != _event(action="updated").dedupe_key()
