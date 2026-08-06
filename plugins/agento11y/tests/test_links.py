"""The jump from a Flyte run to its conversation in Grafana Agent Observability."""

import flyte

from flyteplugins.agento11y import GrafanaAgentObservability

ARGS = {
    "run_name": "run-abc",
    "project": "proj",
    "domain": "dev",
    "context": {},
    "parent_action_name": "a0",
    "action_name": "a1",
    "pod_name": "pod-1",
}


def test_the_link_opens_the_conversation_itself():
    """Matched against a URL copied out of the app, so the shape is not a guess."""
    url = GrafanaAgentObservability(host="https://myorg.grafana.net").get_link(**ARGS)
    assert url == (
        "https://myorg.grafana.net/a/grafana-agento11y-app/conversations/run-abc/explore"
        "?returnTo=%2Fa%2Fgrafana-agento11y-app%2Fconversations%3FconversationId%3Drun-abc"
    )


def test_the_back_navigation_can_be_dropped():
    link = GrafanaAgentObservability(host="https://x.grafana.net", return_to=False)
    assert link.get_link(**ARGS) == "https://x.grafana.net/a/grafana-agento11y-app/conversations/run-abc/explore"


def test_the_app_id_and_paths_are_overridable():
    """The app was renamed from grafana-sigil-app once already, so this must not be fixed."""
    link = GrafanaAgentObservability(
        host="https://x.grafana.net",
        app_id="grafana-sigil-app",
        conversation_path="threads/{conversation_id}",
        list_path="threads",
        return_to=False,
    )
    assert link.get_link(**ARGS) == "https://x.grafana.net/a/grafana-sigil-app/threads/run-abc"


def test_without_a_run_it_falls_back_to_the_list():
    link = GrafanaAgentObservability(host="https://x.grafana.net")
    assert link.get_link(**{**ARGS, "run_name": ""}) == "https://x.grafana.net/a/grafana-agento11y-app/conversations"


def test_run_names_are_url_encoded():
    url = GrafanaAgentObservability(host="https://x.grafana.net", return_to=False).get_link(
        **{**ARGS, "run_name": "a b/c"}
    )
    assert url.endswith("/conversations/a%20b%2Fc/explore")


def test_flyte_template_placeholders_survive_url_encoding():
    """Flyte substitutes {{.runName}} by string replacement on the finished URI.

    Percent-encoding the braces makes that replacement miss, and the app would then open a
    conversation literally named "{{.runName}}". Both the path and the back-navigation
    parameter carry the id, so both have to keep it literal.
    """
    url = GrafanaAgentObservability(host="https://x.grafana.net").get_link(**{**ARGS, "run_name": "{{.runName}}"})
    assert "/conversations/{{.runName}}/explore" in url
    assert url.endswith("conversations%3FconversationId%3D{{.runName}}")
    assert "%7B%7B" not in url


def test_the_link_can_be_attached_to_a_task():
    """The actual contract: usable as `links=` and surfaced on the task."""
    link = GrafanaAgentObservability(host="https://x.grafana.net")
    env = flyte.TaskEnvironment(name="ao11y_linked")

    @env.task(links=(link,))
    async def t() -> int:
        return 1

    assert tuple(t.links) == (link,)


def test_by_run_false_lands_on_the_conversations_list():
    """Google ADK and Pydantic AI name conversations themselves, so the run is not the id.

    Addressing by run there would produce a URL that resolves to nothing, which is worse than
    an extra click to the list.
    """
    link = GrafanaAgentObservability(host="https://x.grafana.net", by_run=False)
    assert link.get_link(**ARGS) == "https://x.grafana.net/a/grafana-agento11y-app/conversations"
