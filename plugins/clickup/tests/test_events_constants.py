"""Tests for the typed event-type constants."""

from __future__ import annotations

import enum

from flyteplugins.clickup import events


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


def test_constants_match_what_the_parser_produces():
    """The whole point: a constant must equal the parsed event's qualified_type."""
    from flyteplugins.clickup import ClickUpAppEnvironment, ClickUpEvent

    app = ClickUpAppEnvironment(name="events-test")

    event = ClickUpEvent(event="taskStatusUpdated")
    assert event.qualified_type == events.Task.STATUS_UPDATED
    assert app._matches(events.Task.STATUS_UPDATED, event)

    other = ClickUpEvent(event="listCreated")
    assert other.qualified_type == events.List.CREATED


def test_handlers_register_with_a_constant():
    from flyteplugins.clickup import ClickUpAppEnvironment, ClickUpEvent

    app = ClickUpAppEnvironment(name="events-test")

    @app.on_event(events.Task.STATUS_UPDATED)
    async def handler(event):  # pragma: no cover - never invoked
        return None

    event = ClickUpEvent(event="taskStatusUpdated")
    assert any(app._matches(pattern, event) for pattern, _ in app.event_handlers)
