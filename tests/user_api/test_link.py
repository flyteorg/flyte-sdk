from flyte._link import Link


class MyLink:
    name: str = "test-link"
    icon_uri: str = ""

    def get_link(self, run_name, project, domain, context, parent_action_name, action_name, pod_name, **kwargs):
        return f"https://example.com/{project}/{domain}/{run_name}/{action_name}"


def test_link_has_protocol_shape():
    link = MyLink()
    assert hasattr(link, "name")
    assert hasattr(link, "get_link")
    assert callable(link.get_link)


def test_link_get_link():
    link = MyLink()
    result = link.get_link(
        run_name="run-1",
        project="my-project",
        domain="development",
        context={},
        parent_action_name="parent",
        action_name="child",
        pod_name="pod-1",
    )
    assert "my-project" in result
    assert "development" in result
    assert "run-1" in result
    assert "child" in result


def test_link_protocol_is_protocol():
    from typing import Protocol

    assert issubclass(Link, Protocol)
