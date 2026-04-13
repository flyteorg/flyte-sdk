import pytest

from flyte.app import Link


def test_link_basic():
    link = Link(path="/docs", title="API Docs")
    assert link.path == "/docs"
    assert link.title == "API Docs"
    assert link.is_relative is False


def test_link_relative():
    link = Link(path="/health", title="Health", is_relative=True)
    assert link.is_relative is True


def test_link_absolute():
    link = Link(path="https://example.com/docs", title="External Docs", is_relative=False)
    assert link.path == "https://example.com/docs"
    assert link.is_relative is False


def test_link_frozen():
    link = Link(path="/test", title="Test")
    with pytest.raises(AttributeError):
        link.path = "/changed"


def test_link_equality():
    l1 = Link(path="/docs", title="Docs")
    l2 = Link(path="/docs", title="Docs")
    assert l1 == l2


def test_link_inequality():
    l1 = Link(path="/docs", title="Docs")
    l2 = Link(path="/health", title="Health")
    assert l1 != l2
