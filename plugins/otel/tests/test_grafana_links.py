"""The jump from a Flyte run to its telemetry in Grafana."""

import json
from urllib.parse import parse_qs, urlparse

import flyte

from flyteplugins.otel.grafana import GrafanaTrace

ARGS = {
    "run_name": "run-abc",
    "project": "proj",
    "domain": "dev",
    "context": {},
    "parent_action_name": "a0",
    "action_name": "a1",
    "pod_name": "pod-1",
}


def explore_state(url):
    panes = json.loads(parse_qs(urlparse(url).query)["panes"][0])
    return next(iter(panes.values()))


def test_the_trace_link_queries_on_the_run_not_a_trace_id():
    """Querying survives however trace ids end up being assigned; addressing by id does not."""
    url = GrafanaTrace(host="https://myorg.grafana.net", datasource_uid="tempo-uid").get_link(**ARGS)
    state = explore_state(url)
    assert state["queries"][0]["query"] == '{.flyte.run_name="run-abc"}'
    assert state["queries"][0]["queryType"] == "traceql"
    assert state["datasource"] == "tempo-uid"


def test_the_trace_link_carries_a_time_range():
    """Explore defaults to the last hour, so a link to an older run would open empty."""
    state = explore_state(GrafanaTrace(host="https://x.grafana.net", datasource_uid="t").get_link(**ARGS))
    assert state["range"] == {"from": "now-7d", "to": "now"}


def test_the_trace_link_can_be_narrowed_to_one_action():
    url = GrafanaTrace(host="https://x.grafana.net", datasource_uid="t", action_scoped=True).get_link(**ARGS)
    query = explore_state(url)["queries"][0]["query"]
    assert query == '{.flyte.run_name="run-abc" && .flyte.action_name="a1"}'


def test_a_trailing_slash_on_the_host_does_not_double_up():
    url = GrafanaTrace(host="https://x.grafana.net/", datasource_uid="t").get_link(**ARGS)
    assert url.startswith("https://x.grafana.net/explore?")


def test_the_trace_link_can_be_attached_to_a_task():
    """The actual contract: usable as `links=` and surfaced on the task."""
    links = (GrafanaTrace(host="https://x.grafana.net", datasource_uid="t"),)
    env = flyte.TaskEnvironment(name="linked")

    @env.task(links=links)
    async def t() -> int:
        return 1

    assert tuple(t.links) == links
    assert [link.name for link in t.links] == ["Grafana trace"]


def test_flyte_template_placeholders_survive_url_encoding():
    """Flyte substitutes {{.runName}} by string replacement on the finished URI.

    Percent-encoding the braces makes that replacement miss, and the placeholder reaches
    Grafana verbatim, so the query searches for a run literally named "{{.runName}}".
    """
    url = GrafanaTrace(host="https://x.grafana.net", datasource_uid="t", action_scoped=True).get_link(
        **{**ARGS, "run_name": "{{.runName}}", "action_name": "{{.actionName}}"}
    )
    assert "{{.runName}}" in url
    assert "{{.actionName}}" in url
    assert "%7B%7B" not in url

    query = explore_state(url)["queries"][0]["query"]
    assert query == '{.flyte.run_name="{{.runName}}" && .flyte.action_name="{{.actionName}}"}'


def test_ordinary_run_names_are_still_encoded():
    """Only the placeholders are exempt; a real name with reserved characters must be escaped."""
    url = GrafanaTrace(host="https://x.grafana.net", datasource_uid="t").get_link(**ARGS)
    assert "%22" in url, "the query JSON should still be percent-encoded"
    assert explore_state(url)["queries"][0]["query"] == '{.flyte.run_name="run-abc"}'
