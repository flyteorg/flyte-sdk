"""ArtifactLineageAppEnvironment — dashboard app for artifact lineage.

Renders, for any published artifact, the full chain of producers and
consumers around it: walk upstream to the original run/artifact that started
the chain, and downstream through every run, artifact, and app that consumed
it (directly or transitively). See `_lineage.py` for how the graph is built
and why two signals (bound-input literals, and the `__upstream_artifact__`
label) are both needed.

Endpoints:

- `GET /lineage` — list of published artifacts (click through to a graph)
- `GET /lineage/artifact/{name}` — lineage graph for one artifact, as HTML
  (`?version=` selects a version; defaults to latest)
- `GET /lineage/graph/{name}` — the same lineage graph as JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import fastapi

from flyte.app.extras import FastAPIAppEnvironment

from ._lineage import LABEL_UPSTREAM_ARTIFACT, build_artifact_lineage

logger = logging.getLogger(__name__)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_artifact_list_html(groups: list, title: str) -> str:
    rows = "".join(
        f"""<a class="row" href="/lineage/artifact/{_html_escape(g["name"])}">
              <span class="name">{_html_escape(g["name"])}</span>
              <span class="version mono">{_html_escape(g["latest_version"])}</span>
              <span class="count">{g["versions"]} version{"s" if g["versions"] != 1 else ""}</span>
              <span class="desc">{_html_escape(g["description"])}</span>
            </a>"""
        for g in groups
    )
    empty = '<p class="empty">No artifacts published yet.</p>' if not groups else ""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_html_escape(title)}</title>
<style>
  :root {{ --bg: #0a0a0f; --panel: #131318; --card: #1c1c22; --card-border: #33333c;
           --text: #e7e7ea; --muted: #8a8a94; --accent: #a98fd1; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; background: var(--bg); color: var(--text); }}
  header {{ padding: 1.2rem 2rem 0.6rem; }}
  h1 {{ font-size: 1.15rem; margin: 0; font-weight: 600; }}
  .sub {{ color: var(--muted); font-size: 0.8rem; margin: 0.35rem 0 0; }}
  .sub code {{ background: #1e1e25; padding: 0 0.3rem; border-radius: 4px; }}
  main {{ padding: 0.8rem 2rem 2rem; max-width: 920px; }}
  .row {{ display: flex; align-items: center; gap: 14px; padding: 12px 14px;
          background: var(--card); border: 1px solid var(--card-border); border-radius: 10px;
          margin-bottom: 8px; text-decoration: none; color: var(--text); transition: border-color 0.15s; }}
  .row:hover {{ border-color: var(--accent); }}
  .name {{ font-weight: 600; font-size: 13.5px; min-width: 220px; }}
  .version {{ color: var(--muted); font-size: 11.5px; min-width: 90px; }}
  .count {{ color: var(--muted); font-size: 11px; background: #26262e; border-radius: 999px;
            padding: 2px 9px; flex: none; }}
  .desc {{ color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .empty {{ color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>{_html_escape(title)}</h1>
  <p class="sub">Click an artifact to trace its lineage — every producer back to the origin,
  every consumer downstream. Built live from artifact provenance and the
  <code>{LABEL_UPSTREAM_ARTIFACT}</code> label — no stored lineage state.</p>
</header>
<main>{empty}{rows}</main>
</body>
</html>"""


def _render_graph_html(graph_json: dict, title: str, console_base: str, graph_url: str) -> str:
    base = console_base.rstrip("/")
    boot_json = json.dumps(graph_json).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_html_escape(title)}</title>
<link rel="stylesheet" href="https://unpkg.com/@xyflow/react@12.8.2/dist/style.css"/>
<style>
  :root {{
    --bg: #0a0a0f; --panel: #131318; --card: #1c1c22; --card-border: #33333c;
    --text: #e7e7ea; --muted: #8a8a94;
    --artifact: #a98fd1; --run: #f59e0b; --app: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0; background: var(--bg); color: var(--text); }}
  header {{ padding: 1.2rem 2rem 0.6rem; display: flex; align-items: center; gap: 14px; }}
  h1 {{ font-size: 1.15rem; margin: 0; font-weight: 600; }}
  .back {{ color: var(--muted); text-decoration: none; font-size: 0.8rem; border: 1px solid #2a2a33;
           border-radius: 8px; padding: 5px 12px; }}
  .back:hover {{ color: var(--text); border-color: #4d4d59; }}
  .legend {{ margin-left: auto; display: flex; gap: 14px; font-size: 11.5px; color: var(--muted); }}
  .legend .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 999px; margin-right: 5px; }}
  main {{ padding: 0.8rem 2rem 1.2rem; }}
  #app {{ height: 78vh; min-height: 460px; border: 1px solid #23232b; border-radius: 12px; background: #101015; }}
  .fallback {{ display: flex; height: 100%; align-items: center; justify-content: center;
               color: var(--muted); font-size: 0.85rem; }}
  .react-flow__controls {{ box-shadow: none; border: 1px solid #2a2a33; border-radius: 8px; overflow: hidden; }}
  .react-flow__controls-button {{ background: var(--card); border-bottom: 1px solid #2a2a33; fill: var(--muted); }}
  .react-flow__edge-path {{ stroke: #4a4a55; }}
  .react-flow__edge-textbg {{ fill: #101015; }}
  .react-flow__edge-text {{ fill: var(--muted); font-size: 10px; }}
  .react-flow__attribution {{ background: transparent; color: #55555f; }}
  .lin-card {{ width: 236px; background: var(--card); border: 1px solid var(--card-border);
               border-radius: 10px; padding: 10px 12px; cursor: pointer; border-left-width: 3px; }}
  .lin-card:hover {{ border-color: #4d4d59; }}
  .lin-card.artifact {{ border-left-color: var(--artifact); }}
  .lin-card.artifact.root {{ border-color: var(--artifact); }}
  .lin-card.run {{ border-left-color: var(--run); }}
  .lin-card.app {{ border-left-color: var(--app); }}
  .lin-card .kind {{ font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .lin-card .title {{ font-size: 12.5px; font-weight: 600; margin-top: 2px; white-space: nowrap;
                       overflow: hidden; text-overflow: ellipsis; }}
  .lin-card .sub {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px;
                     color: var(--muted); margin-top: 5px; white-space: nowrap; overflow: hidden;
                     text-overflow: ellipsis; }}
  .lin-card .pill {{ display: inline-block; margin-top: 6px; font-size: 9.5px; padding: 1px 7px;
                      border-radius: 999px; text-transform: capitalize; }}
  .pill.succeeded {{ background: rgba(22,163,74,0.15); color: #4ade80; }}
  .pill.failed, .pill.aborted, .pill.timed_out {{ background: rgba(220,38,38,0.15); color: #f87171; }}
  .pill.running {{ background: rgba(2,132,199,0.18); color: #60a5fa; }}
  .pill.queued, .pill.initializing, .pill.waiting_for_resources {{
    background: rgba(148,148,160,0.15); color: #a1a1ab;
  }}
</style>
</head>
<body>
<header>
  <a class="back" href="/lineage">&larr; All artifacts</a>
  <h1>{_html_escape(title)}</h1>
  <div class="legend">
    <span><span class="dot" style="background:var(--artifact)"></span>artifact</span>
    <span><span class="dot" style="background:var(--run)"></span>run</span>
    <span><span class="dot" style="background:var(--app)"></span>app</span>
  </div>
</header>
<main>
  <div id="app"><div class="fallback">Lineage graph could not render (CDN unreachable?) — see the
    <a href="{_html_escape(graph_url)}">raw graph JSON</a>.</div></div>
</main>
<script type="importmap">
{{
  "imports": {{
    "react": "https://esm.sh/react@18.3.1",
    "react-dom/client": "https://esm.sh/react-dom@18.3.1/client?deps=react@18.3.1",
    "@xyflow/react": "https://esm.sh/@xyflow/react@12.8.2?deps=react@18.3.1,react-dom@18.3.1",
    "@dagrejs/dagre": "https://esm.sh/@dagrejs/dagre@1.1.4",
    "htm": "https://esm.sh/htm@3.1.1"
  }}
}}
</script>
<script id="lineage-data" type="application/json">{boot_json}</script>
<script type="module">
try {{
  const React = (await import("react")).default;
  const {{ createRoot }} = await import("react-dom/client");
  const {{ ReactFlow, Background, BackgroundVariant, Controls, Handle, Position, MarkerType }} =
    await import("@xyflow/react");
  const dagre = (await import("@dagrejs/dagre")).default;
  const htm = (await import("htm")).default;
  const html = htm.bind(React.createElement);

  const GRAPH = JSON.parse(document.getElementById("lineage-data").textContent);
  const CONSOLE_BASE = {json.dumps(base)};
  const NODE_W = 236, NODE_H = 74;

  const LinNode = ({{ data }}) => html`
    <div class="lin-card ${{data.kind}} ${{data.isRoot ? "root" : ""}}"
         title=${{data.title}}
         onClick=${{() => data.url && window.open(CONSOLE_BASE + data.url, "_blank")}}>
      <${{Handle}} type="target" position=${{Position.Left}} style=${{{{ opacity: 0 }}}} />
      <div class="kind">${{data.kind}}</div>
      <div class="title">${{data.title}}</div>
      <div class="sub">${{data.sub}}</div>
      ${{data.phase ? html`<span class="pill ${{data.phase}}">${{data.phase}}</span>` : null}}
      <${{Handle}} type="source" position=${{Position.Right}} style=${{{{ opacity: 0 }}}} />
    </div>`;
  const nodeTypes = {{ lineage: LinNode }};

  function layout(nodesById, edges, rootId) {{
    const g = new dagre.graphlib.Graph();
    g.setGraph({{ rankdir: "LR", nodesep: 30, ranksep: 90 }});
    g.setDefaultEdgeLabel(() => ({{}}));
    Object.keys(nodesById).forEach((id) => g.setNode(id, {{ width: NODE_W, height: NODE_H }}));
    edges.forEach((e) => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const flowNodes = Object.entries(nodesById).map(([id, n]) => {{
      const p = g.node(id);
      let title = "", sub = "", url = "";
      if (n.kind === "artifact") {{ title = n.name; sub = n.version; url = n.url; }}
      else if (n.kind === "run") {{ title = (n.task_name || "?").split(".").pop(); sub = n.run_name; url = n.url; }}
      else {{ title = n.app_name; sub = n.endpoint || ""; url = n.url; }}
      return {{
        id, type: "lineage", position: {{ x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 }},
        data: {{ kind: n.kind, title, sub, url, phase: n.phase || "", isRoot: id === rootId }},
      }};
    }});
    const flowEdges = edges.map((e, i) => ({{
      id: "e" + i, source: e.source, target: e.target, type: "smoothstep",
      label: e.relation, style: e.relation === "consumes" ? {{ strokeDasharray: "4 3" }} : undefined,
      markerEnd: {{ type: MarkerType.ArrowClosed, color: "#4a4a55", width: 16, height: 16 }},
    }}));
    return {{ flowNodes, flowEdges }};
  }}

  const {{ flowNodes, flowEdges }} = layout(GRAPH.nodes, GRAPH.edges, GRAPH.root);
  const App = () => html`
    <${{ReactFlow}} nodes=${{flowNodes}} edges=${{flowEdges}} nodeTypes=${{nodeTypes}}
        fitView fitViewOptions=${{{{ padding: 0.15, maxZoom: 1 }}}} minZoom=${{0.2}}
        proOptions=${{{{ hideAttribution: true }}}} nodesConnectable=${{false}} colorMode="dark">
      <${{Background}} variant=${{BackgroundVariant.Dots}} gap=${{22}} size=${{1.4}} color="#26262e" />
      <${{Controls}} showInteractive=${{false}} />
    <//>`;
  createRoot(document.getElementById("app")).render(html`<${{App}} />`);
}} catch (err) {{
  console.error(err);
}}
</script>
</body>
</html>"""


def _derive_console_base() -> str:
    try:
        from flyte._initialize import get_client

        url = get_client().console.run_url(project="p", domain="d", run_name="r")
        return url.split("/v2/", 1)[0]
    except Exception:
        return ""


def _ensure_client() -> None:
    from flyte._initialize import ensure_client

    ensure_client()


@dataclass(kw_only=True)
class ArtifactLineageAppEnvironment(FastAPIAppEnvironment):
    """AppEnvironment serving a dashboard of artifact producer/consumer lineage.

    Given the name of a published artifact, renders the graph of everything
    connected to it: every run that produced an ancestor artifact (traced back
    to the original run/artifact in the chain) and every run, artifact, or app
    downstream of it. See `_lineage.build_artifact_lineage` for how the graph
    is assembled, and the module docstring in `_lineage.py` for why some
    consumers need the `__upstream_artifact__` label to be discoverable at all.

    ```python
    lineage_app = ArtifactLineageAppEnvironment(
        name="artifact-lineage",
        watched_tasks=["artifact_lineage_example.consume", "artifact_lineage_example.re_export"],
        watched_apps=["artifact-lineage-consumer-app"],
    )
    ```
    """

    app: fastapi.FastAPI = field(default_factory=fastapi.FastAPI)
    # Task/app names to include in the bound-input consumer scan, beyond what the
    # `__upstream_artifact__` label already finds on its own (see `_lineage.py`).
    watched_tasks: list[str] = field(default_factory=list)
    watched_apps: list[str] = field(default_factory=list)
    scan_limit: int = 50
    max_depth: int = 25
    console_base: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "app", self._build_fastapi_app())
        super().__post_init__()

    def _console_base(self) -> str:
        return self.console_base or _derive_console_base()

    async def _artifact_groups(self) -> list[dict]:
        from flyte.remote import Artifact

        groups = []
        async for group in Artifact.list_names.aio():
            groups.append(
                {
                    "name": group.name,
                    "latest_version": group.latest.version,
                    "versions": group.versions,
                    "description": group.pb2.latest.spec.info.description or "",
                }
            )
        return groups

    async def _resolve_artifact(self, name: str, version: str = "latest"):
        from flyte.remote import Artifact

        return await Artifact.get.aio(name=name, version=version)

    async def _graph_json(self, name: str, version: str) -> dict:
        from dataclasses import asdict

        _ensure_client()
        artifact = await self._resolve_artifact(name, version)
        graph = await build_artifact_lineage(
            artifact,
            watched_tasks=self.watched_tasks,
            watched_apps=self.watched_apps,
            scan_limit=self.scan_limit,
            max_depth=self.max_depth,
        )
        return asdict(graph)

    def _build_fastapi_app(self):
        app = fastapi.FastAPI(title="ArtifactLineage")
        env = self

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {
                "status": "ok",
                "watched_tasks": str(len(env.watched_tasks)),
                "watched_apps": str(len(env.watched_apps)),
            }

        @app.get("/", include_in_schema=False)
        async def index() -> fastapi.responses.RedirectResponse:
            return fastapi.responses.RedirectResponse(url="/lineage")

        @app.get("/lineage", response_class=fastapi.responses.HTMLResponse)
        async def lineage_list() -> str:
            try:
                _ensure_client()
                groups = await env._artifact_groups()
            except Exception:
                logger.exception("Failed to list artifacts")
                return "<html><body><p>Could not list artifacts — check app logs.</p></body></html>"
            return _render_artifact_list_html(groups, title=f"{env.name} — artifacts")

        @app.get("/lineage/graph/{name}")
        async def lineage_graph(name: str, version: str = "latest") -> dict:
            return await env._graph_json(name, version)

        @app.get("/lineage/artifact/{name}", response_class=fastapi.responses.HTMLResponse)
        async def lineage_artifact(name: str, version: str = "latest") -> str:
            try:
                graph = await env._graph_json(name, version)
            except Exception:
                logger.exception("Failed to build lineage graph for %s", name)
                return (
                    "<html><body><p>Could not build lineage graph — is the artifact "
                    "name/version correct? Check app logs.</p></body></html>"
                )
            graph_url = f"/lineage/graph/{name}" + (f"?version={version}" if version != "latest" else "")
            return _render_graph_html(
                graph,
                title=f"{env.name} — {name}",
                console_base=env._console_base(),
                graph_url=graph_url,
            )

        return app
