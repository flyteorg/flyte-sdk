"""Tests for the typed event-type constants."""

from __future__ import annotations

import enum

import pytest

from flyteplugins.github import events


@pytest.fixture
def app():
    from flyteplugins.github import GitHubAppEnvironment

    return GitHubAppEnvironment(name="events-test")


def test_constants_are_plain_strings():
    """`str` subclasses, so they drop into any API that takes a pattern string."""
    for name in events.__all__:
        for member in getattr(events, name):
            assert isinstance(member, str)
            assert member == member.value


def test_constants_render_as_their_wire_value():
    """Python 3.11+ would otherwise render members as "Class.MEMBER"."""
    for name in events.__all__:
        for member in getattr(events, name):
            assert str(member) == member.value
            assert f"{member}" == member.value


def test_every_exported_class_is_an_event_enum():
    assert events.__all__, "no event classes exported"
    for name in events.__all__:
        cls = getattr(events, name)
        assert issubclass(cls, enum.Enum)
        assert issubclass(cls, str)
        assert len(cls) > 0


def test_no_duplicate_values_across_classes():
    """A value in two classes means one of them is wrong."""
    seen: dict[str, str] = {}
    for name in events.__all__:
        for member in getattr(events, name):
            assert member.value not in seen, f"{member.value} in both {seen.get(member.value)} and {name}"
            seen[member.value] = name


def test_constants_match_what_the_parser_produces(app):
    """The whole point: a constant must equal the parsed event's qualified_type."""
    from flyteplugins.github import GitHubEvent

    event = GitHubEvent(event_type="pull_request", action="opened", delivery_id="d")
    assert event.qualified_type == events.PullRequest.OPENED
    assert app._matches(events.PullRequest.OPENED, event)
    assert app._matches(events.PullRequest.ANY, event), "ANY must match every action"
    assert not app._matches(events.PullRequest.CLOSED, event)
    assert not app._matches(events.Issues.OPENED, event), "must not cross event types"


def test_any_is_the_bare_event_type():
    """`ANY` has to equal the type alone, since that is what `_matches` compares."""
    for name in events.__all__:
        cls = getattr(events, name)
        if "ANY" not in cls.__members__:
            continue
        bare = cls.ANY.value
        assert "." not in bare, f"{name}.ANY should be a bare event type, got {bare}"
        for member in cls:
            if member is cls.ANY:
                continue
            assert member.value.startswith(f"{bare}."), f"{member.value} is not an action of {bare}"


def test_handlers_register_with_a_constant(app):
    from flyteplugins.github import GitHubEvent

    @app.on_event(events.IssueComment.CREATED)
    async def handler(event):  # pragma: no cover - never invoked
        return None

    event = GitHubEvent(event_type="issue_comment", action="created", delivery_id="d")
    assert any(app._matches(pattern, event) for pattern, _ in app.event_handlers)
