"""Links from a Flyte action into Grafana.

Flyte renders a task's `links` in its UI, so this is the jump from a run to the telemetry
it produced. These are pure URL builders with no Grafana dependency, and they work off the
`flyte.*` span attributes this plugin already stamps.

The trace link queries rather than addressing a trace by id. That matters: a query on
`flyte.run_name` finds a run's spans whatever their trace ids turn out to be, so it works
before the run-derived trace id is involved, and keeps working after. Addressing by id would
depend on the derivation and break the moment anything upstream propagated a trace context.

Only the Tempo link lives here. A link to a run's conversation in Grafana Agent Observability
belongs with `flyteplugins-agento11y`, since it is that package's identity binding that
makes a run addressable by conversation id at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode

from flyte import Link

__all__ = ["GrafanaTrace"]

# Flyte hands links placeholders like {{.runName}} at serialization time and substitutes them
# by string replacement on the finished URI. Percent-encoding turns "{{" into "%7B%7B", which
# that replacement never matches, so the placeholder would reach Grafana verbatim and the
# query would search for a run literally named "{{.runName}}".
_ENCODED_TEMPLATE = re.compile(r"%7B%7B([.\w]+)%7D%7D")


def _encode_preserving_templates(params: Dict[str, object]) -> str:
    """urlencode, then put Flyte's template placeholders back in literal form."""
    return _ENCODED_TEMPLATE.sub(r"{{\1}}", urlencode(params))


def _explore_url(*, host: str, datasource_uid: str, query: str, time_from: str, time_to: str) -> str:
    """Build a Grafana Explore deep link running a TraceQL query.

    Explore encodes its state as a JSON `panes` parameter. The time range has to be part of
    it because Explore otherwise defaults to the last hour, and a link to yesterday's run
    would open on an empty pane and read as broken.
    """
    panes = {
        "flyte": {
            "datasource": datasource_uid,
            "queries": [
                {
                    "refId": "A",
                    "datasource": {"type": "tempo", "uid": datasource_uid},
                    "queryType": "traceql",
                    "query": query,
                }
            ],
            "range": {"from": time_from, "to": time_to},
        }
    }
    params = _encode_preserving_templates(
        {"schemaVersion": 1, "panes": json.dumps(panes, separators=(",", ":")), "orgId": 1}
    )
    return f"{host.rstrip('/')}/explore?{params}"


@dataclass
class GrafanaTrace(Link):
    """A link to this run's spans in Grafana Tempo.

    Args:
        host: Stack URL, e.g. `https://myorg.grafana.net`.
        datasource_uid: UID of the Tempo datasource. Per-stack and not guessable, so there is
            no useful default; find it under Connections, Data sources in Grafana.
        name: Label shown in the Flyte UI.
        lookback: Start of the Explore time range, in Grafana's relative syntax. The link
            protocol gives no timestamps, so the window is relative and deliberately wide.
        action_scoped: Narrow the query to the single action rather than the whole run.
    """

    host: str
    datasource_uid: str
    name: str = "Grafana trace"
    icon_uri: Optional[str] = ""
    lookback: str = "now-7d"
    action_scoped: bool = False

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: Dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
        **kwargs,
    ) -> str:
        selectors = [f'.flyte.run_name="{run_name}"']
        if self.action_scoped and action_name:
            selectors.append(f'.flyte.action_name="{action_name}"')
        query = "{" + " && ".join(selectors) + "}"
        return _explore_url(
            host=self.host,
            datasource_uid=self.datasource_uid,
            query=query,
            time_from=self.lookback,
            time_to="now",
        )
