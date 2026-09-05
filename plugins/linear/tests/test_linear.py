"""Linear-specific normalization, beyond conformance."""

from __future__ import annotations

import hashlib
import hmac
import json

from flyteplugins.linear import events, parse

SECRET = "linear-secret"


def _parse(payload: dict):
    body = json.dumps(payload).encode()
    return parse({"X-Linear-Signature": hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()}, body)


def test_entity_and_action_join_into_the_constant():
    event = _parse({"action": "create", "type": "Issue", "data": {"id": "i1"}})
    assert event.qualified_type == events.Issue.CREATE


def test_a_later_update_gets_its_own_key():
    """Keyed on the entity alone, only the first update would ever launch."""

    def update(updated_at: str):
        return _parse({"action": "update", "type": "Issue", "data": {"id": "i1", "updatedAt": updated_at}})

    assert update("2024-01-01T00:00:00Z").dedupe_key() != update("2024-06-01T00:00:00Z").dedupe_key()


def test_the_team_id_is_found_nested_on_a_comment():
    """Comment payloads carry the team only on the nested issue."""
    event = _parse(
        {"action": "create", "type": "Comment", "data": {"id": "c1", "issue": {"id": "i1", "team": {"id": "team-9"}}}}
    )
    assert event.scope == "team-9"


def test_a_payload_with_no_entity_timestamp_falls_back_to_the_delivery_time():
    event = _parse({"action": "create", "type": "Issue", "createdAt": "2024-01-01T00:00:00Z", "data": {"id": "i1"}})
    assert event.occurred_at == "2024-01-01T00:00:00Z"
