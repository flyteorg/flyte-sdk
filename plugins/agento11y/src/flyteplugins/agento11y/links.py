"""Links from a Flyte action into Grafana Agent Observability.

This lives here rather than in ``flyteplugins-otel`` because it is only correct when this
package is in use: it addresses a run by conversation id, and it is
`flyteplugins.agento11y.FlyteIdentityBinding` that makes the run name the conversation
id in the first place. Shipping it alongside the generic tracing plugin would mean a link that
silently points at nothing for anyone not running the bridge.

For a link to the run's spans in Tempo, which needs nothing from this package, use
``flyteplugins.otel.grafana.GrafanaTrace``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import quote

from flyte import Link

__all__ = ["GrafanaAgentObservability"]

# Flyte hands links placeholders such as {{.runName}} at serialization time and substitutes
# them by string replacement on the finished URI. Percent-encoding the braces makes that
# replacement miss, and the app would then open a conversation literally named "{{.runName}}".
_ENCODED_TEMPLATE = re.compile(r"%7B%7B([.\w]+)%7D%7D")


def _quote_preserving_templates(value: str) -> str:
    """Percent-encode a value, leaving any Flyte template placeholders literal."""
    return _ENCODED_TEMPLATE.sub(r"{{\1}}", quote(value, safe=""))


@dataclass
class GrafanaAgentObservability(Link):
    """A link to this run's conversation in Grafana Agent Observability.

    Opens the conversation itself rather than the filtered list, since the run maps one to one
    onto a conversation and the list is an extra click. ``return_to`` fills the app's back
    navigation with the list filtered to the same run, which is what the app itself does.

    This resolves whenever the run really is the conversation id, which every framework this
    plugin instruments now arranges. ``by_run=False`` remains for anything that names
    conversations itself — a framework added later, or an agent whose conversations
    deliberately span several runs — and lands on the conversations list rather than a URL
    that resolves to nothing.

    Args:
        host: Stack URL, e.g. ``https://myorg.grafana.net``.
        name: Label shown in the Flyte UI.
        app_id: Grafana app plugin id. The app moved from ``grafana-sigil-app`` to
            ``grafana-agento11y-app``; the old id still resolves but is deprecated.
        conversation_path: Path template within the app, given the conversation id.
        list_path: Path of the conversations list, used for the back navigation.
        return_to: Include the back-navigation parameter. Set False for a bare link.
        by_run: Address the conversation by the Flyte run. Set False when something else owns
            the conversation id, so the link goes to the list rather than a dead conversation.
    """

    host: str
    name: str = "Grafana Agent Observability"
    icon_uri: Optional[str] = ""
    app_id: str = "grafana-agento11y-app"
    conversation_path: str = "conversations/{conversation_id}/explore"
    list_path: str = "conversations"
    return_to: bool = True
    by_run: bool = True

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
        app = f"{self.host.rstrip('/')}/a/{self.app_id}"
        if not self.by_run or not run_name:
            return f"{app}/{self.list_path.strip('/')}"

        conversation = _quote_preserving_templates(run_name)
        url = f"{app}/{self.conversation_path.strip('/').format(conversation_id=conversation)}"
        if not self.return_to:
            return url

        # The app percent-encodes this whole path as one query value, so the encoding is
        # applied to the already-built string rather than to its parts.
        back = f"/a/{self.app_id}/{self.list_path.strip('/')}?conversationId={conversation}"
        return f"{url}?returnTo={_quote_preserving_templates(back)}"
