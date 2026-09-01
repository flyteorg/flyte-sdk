"""Tests for event normalization, dedupe keys, and the typed constants."""

from __future__ import annotations

import enum

import pytest
from conftest import (
    ALL_PROVIDERS,
    body_of,
    clickup_headers,
    clickup_payload,
    github_headers,
    jira_headers,
    jira_payload,
    linear_headers,
    linear_payload,
    pr_payload,
)

from flyteplugins.webhooks import PROVIDERS, WebhookEvent, events

PROVIDER_MODULES = [getattr(events, name) for name in events.__all__ if name != "EventType"]


def _parse(provider: str, headers_for, payload):
    body = body_of(payload)
    return PROVIDERS[provider].parse(headers_for(body), body)


# ----------------------------------------------------------------------
# dedupe semantics
# ----------------------------------------------------------------------


def test_redelivery_of_one_event_dedupes():
    first = _parse("github", github_headers, pr_payload())
    again = _parse("github", github_headers, pr_payload())
    assert first.dedupe_key() == again.dedupe_key()


def test_distinct_resources_get_distinct_keys():
    a = _parse("github", github_headers, pr_payload(number=1))
    b = _parse("github", github_headers, pr_payload(number=2))
    assert a.dedupe_key() != b.dedupe_key()


def test_a_later_change_to_one_resource_gets_its_own_key():
    """Without the timestamp, every update after the first would never launch."""
    first = linear_payload()
    second = linear_payload()
    second["data"]["updatedAt"] = "2024-06-01T00:00:00Z"
    assert _parse("linear", linear_headers, first).dedupe_key() != _parse("linear", linear_headers, second).dedupe_key()


def test_distinct_comments_on_one_issue_do_not_collapse():
    """Keyed on the issue alone, every comment after the first looks like a redelivery."""

    def comment(comment_id: int) -> dict:
        return {
            "action": "created",
            "issue": {"number": 9, "title": "A bug"},
            "comment": {"id": comment_id, "html_url": "https://example.invalid"},
            "repository": {"full_name": "octo/repo"},
        }

    a = _parse("github", lambda b: github_headers(b, event="issue_comment"), comment(1))
    b = _parse("github", lambda b: github_headers(b, event="issue_comment"), comment(2))
    assert a.resource_id != b.resource_id
    assert a.dedupe_key() != b.dedupe_key()


def test_events_without_a_resource_fall_back_to_the_delivery_id():
    push = {"repository": {"full_name": "octo/repo"}}
    a = _parse("github", lambda b: github_headers(b, event="push", delivery="x"), push)
    b = _parse("github", lambda b: github_headers(b, event="push", delivery="y"), push)
    assert a.resource_id is None
    assert a.dedupe_key() != b.dedupe_key()


def test_keys_never_collide_across_providers():
    keys = {_parse(p, h, mk()).dedupe_key() for p, h, mk, _ in ALL_PROVIDERS}
    assert len(keys) == len(ALL_PROVIDERS)


# ----------------------------------------------------------------------
# normalization details worth pinning
# ----------------------------------------------------------------------


def test_linear_finds_the_team_id_nested_on_a_comment():
    """Comment payloads carry the team only on the nested issue."""
    payload = {
        "action": "create",
        "type": "Comment",
        "webhookId": "wh-1",
        "createdAt": "2024-01-01T00:00:00Z",
        "data": {"id": "c1", "issue": {"id": "i1", "team": {"id": "team-9"}}},
    }
    assert _parse("linear", linear_headers, payload).scope == "team-9"


def test_clickup_falls_back_to_the_nested_task_list():
    payload = clickup_payload()
    del payload["list_id"]
    payload["task"]["list"] = {"id": "l7"}
    assert _parse("clickup", clickup_headers, payload).scope == "l7"


def test_jira_reads_the_project_key_from_issue_fields():
    assert _parse("jira", jira_headers, jira_payload()).scope == "PROJ"


def test_qualified_type_joins_type_and_action_only_when_there_is_one():
    assert WebhookEvent(provider="p", event_type="issues", action="opened").qualified_type == "issues.opened"
    assert WebhookEvent(provider="p", event_type="taskCreated").qualified_type == "taskCreated"


# ----------------------------------------------------------------------
# typed constants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", PROVIDER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_constants_are_plain_strings_that_render_as_their_value(module):
    for name in module.__all__:
        for member in getattr(module, name):
            assert isinstance(member, str)
            assert member == member.value
            # 3.11+ would otherwise render "Class.MEMBER" into the dashboard.
            assert str(member) == member.value
            assert f"{member}" == member.value


@pytest.mark.parametrize("module", PROVIDER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_exported_class_is_an_event_enum(module):
    assert module.__all__
    for name in module.__all__:
        cls = getattr(module, name)
        assert issubclass(cls, enum.Enum) and issubclass(cls, str)
        assert len(cls) > 0


@pytest.mark.parametrize("module", PROVIDER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_any_is_the_bare_type_its_actions_prefix(module):
    for name in module.__all__:
        cls = getattr(module, name)
        if "ANY" not in cls.__members__:
            continue
        bare = cls.ANY.value
        assert "." not in bare, f"{name}.ANY should be a bare event type, got {bare}"
        for member in cls:
            if member is not cls.ANY:
                assert member.value.startswith(f"{bare}."), f"{member.value} is not an action of {bare}"


@pytest.mark.parametrize("module", PROVIDER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_duplicate_values_within_a_provider(module):
    seen: dict[str, str] = {}
    for name in module.__all__:
        for member in getattr(module, name):
            assert member.value not in seen, f"{member.value} in both {seen.get(member.value)} and {name}"
            seen[member.value] = name


@pytest.mark.parametrize(("provider", "headers_for", "payload_for", "expected"), ALL_PROVIDERS)
def test_constants_match_what_the_parsers_actually_produce(provider, headers_for, payload_for, expected):
    """The point of the constants: they must equal the parsed qualified_type."""
    event = _parse(provider, headers_for, payload_for())
    assert event.qualified_type == expected

    module = getattr(events, provider)
    values = {m.value for name in module.__all__ for m in getattr(module, name)}
    assert expected in values, f"{expected} is not spelled by any {provider} constant"
