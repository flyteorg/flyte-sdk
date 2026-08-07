"""ArtifactLineageAppEnvironment — dashboard app for artifact lineage.

Renders, for any published artifact, the full chain of producers and
consumers around it: walk upstream to the original run/artifact that started
the chain, and downstream through every run, artifact, and app that consumed
it (directly or transitively). See `_lineage.py` for how the graph is built
and why two signals (bound-input literals, and the `upstream-artifact-name`/
`upstream-artifact-version` labels) are both needed.

A single page (`GET /lineage`) does everything: a searchable sidebar lists
every published artifact, and clicking one renders its lineage graph inline
in the same view — no page navigation, just a client-side fetch + re-render.

Endpoints:

- `GET /lineage` — the dashboard (`?artifact=` preselects one, `&version=`
  optionally pins a version; `GET /lineage/artifact/{name}` is a shorthand
  redirect to the same thing, for direct links)
- `GET /lineage/artifacts` — every published artifact, as JSON (sidebar data)
- `GET /lineage/graph/{name}` — one artifact's lineage graph, as JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import fastapi

from flyte.app.extras import FastAPIAppEnvironment

from ._lineage import LABEL_UPSTREAM_ARTIFACT_NAME, LABEL_UPSTREAM_ARTIFACT_VERSION, build_artifact_lineage

logger = logging.getLogger(__name__)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_dashboard_html(
    title: str,
    console_base: str,
    *,
    artifacts_url: str = "/lineage/artifacts",
    graph_url_base: str = "/lineage/graph",
    preselect: dict | None = None,
) -> str:
    base = console_base.rstrip("/")
    boot = {"preselect": preselect, "artifactsUrl": artifacts_url, "graphUrlBase": graph_url_base}
    boot_json = json.dumps(boot).replace("</", "<\\/")
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
    --text: #e7e7ea; --muted: #8a8a94; --accent: #a98fd1;
    --artifact: #a98fd1; --run: #f59e0b; --app: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; background: var(--bg); color: var(--text);
  }}
  code {{ background: #1e1e25; padding: 0 0.3rem; border-radius: 4px; }}
  a {{ color: #93c5fd; }}
  #shell {{ display: flex; height: 100vh; }}
  .sidebar {{
    width: 300px; flex: none; background: var(--panel); border-right: 1px solid #23232b;
    display: flex; flex-direction: column; min-height: 0;
  }}
  .sidebar-head {{ padding: 1rem 1rem 0.7rem; border-bottom: 1px solid #23232b; }}
  .sidebar-head h1 {{ font-size: 1rem; margin: 0 0 2px; font-weight: 600; }}
  .sidebar-head .sub {{ color: var(--muted); font-size: 0.72rem; line-height: 1.5; margin: 6px 0 10px; }}
  .search {{
    width: 100%; background: #0e0e13; border: 1px solid #23232b; border-radius: 8px;
    color: var(--text); font-size: 12.5px; padding: 7px 10px; outline: none;
  }}
  .search:focus {{ border-color: #4d4d59; }}
  .sidebar-list {{ flex: 1; overflow-y: auto; padding: 8px; }}
  .art-item {{
    display: flex; flex-direction: column; gap: 3px; padding: 9px 11px; border-radius: 9px;
    cursor: pointer; border: 1px solid transparent; margin-bottom: 3px;
  }}
  .art-item:hover {{ background: #1a1a20; }}
  .art-item.active {{ background: #22222b; border-color: var(--accent); }}
  .art-row1 {{ display: flex; align-items: center; gap: 8px; }}
  .art-name {{ font-weight: 600; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .art-count {{
    margin-left: auto; flex: none; font-size: 9px; color: var(--muted); background: #26262e;
    border-radius: 999px; padding: 1px 7px;
  }}
  .art-desc {{
    color: var(--muted); font-size: 10.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .empty-list {{ color: var(--muted); font-size: 12px; padding: 10px 6px; }}
  .main {{ flex: 1; display: flex; flex-direction: column; min-width: 0; }}
  .main-head {{
    padding: 1rem 1.5rem; border-bottom: 1px solid #23232b; display: flex; align-items: center; gap: 14px;
    min-height: 32px;
  }}
  .main-head h2 {{ font-size: 1.05rem; margin: 0; font-weight: 600; }}
  .main-head .version {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: var(--muted);
    background: #1c1c22; border: 1px solid #2a2a33; border-radius: 999px; padding: 2px 9px;
  }}
  .main-head .desc {{ color: var(--muted); font-size: 12px; }}
  .main-head .console-link {{
    margin-left: auto; font-size: 0.75rem; color: var(--muted); text-decoration: none;
    border: 1px solid #2a2a33; border-radius: 8px; padding: 5px 12px; flex: none;
  }}
  .main-head .console-link:hover {{ color: var(--text); border-color: #4d4d59; }}
  .legend {{ display: flex; gap: 12px; font-size: 10.5px; color: var(--muted); flex: none; }}
  .legend .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 999px; margin-right: 4px; }}
  .canvas {{ flex: 1; position: relative; background: #101015; }}
  .placeholder {{
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.85rem; text-align: center; padding: 2rem; flex-direction: column; gap: 6px;
  }}
  .react-flow__controls {{ box-shadow: none; border: 1px solid #2a2a33; border-radius: 8px; overflow: hidden; }}
  .react-flow__controls-button {{ background: var(--card); border-bottom: 1px solid #2a2a33; fill: var(--muted); }}
  .react-flow__edge-path {{ stroke: #4a4a55; }}
  .react-flow__edge-textbg {{ fill: #101015; }}
  .react-flow__edge-text {{ fill: var(--muted); font-size: 10px; }}
  .react-flow__attribution {{ background: transparent; color: #55555f; }}
  .lin-card {{
    width: 226px; background: var(--card); border: 1px solid var(--card-border);
    border-radius: 10px; padding: 10px 12px; cursor: pointer; border-left-width: 3px;
    transition: border-color 0.12s ease;
  }}
  .lin-card:hover {{ border-color: #4d4d59; }}
  .lin-card.artifact {{ border-left-color: var(--artifact); }}
  .lin-card.artifact.root {{ border-color: var(--artifact); box-shadow: 0 0 0 1px var(--artifact); }}
  .lin-card.run {{ border-left-color: var(--run); }}
  .lin-card.app {{ border-left-color: var(--app); }}
  .lin-card .kind {{ font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .lin-card .title {{
    font-size: 12.5px; font-weight: 600; margin-top: 2px; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }}
  .lin-card .sub {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px; color: var(--muted);
    margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .lin-card .pill {{
    display: inline-block; margin-top: 6px; font-size: 9.5px; padding: 1px 7px; border-radius: 999px;
    text-transform: capitalize;
  }}
  .pill.succeeded {{ background: rgba(22,163,74,0.15); color: #4ade80; }}
  .pill.failed, .pill.aborted, .pill.timed_out {{ background: rgba(220,38,38,0.15); color: #f87171; }}
  .pill.running {{ background: rgba(2,132,199,0.18); color: #60a5fa; }}
  .pill.queued, .pill.initializing, .pill.waiting_for_resources {{
    background: rgba(148,148,160,0.15); color: #a1a1ab;
  }}
</style>
</head>
<body>
<div id="shell">
  <div class="placeholder" style="position:static;height:100%;">Loading…</div>
</div>
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

  const BOOT = JSON.parse(document.getElementById("lineage-data").textContent);
  const CONSOLE_BASE = {json.dumps(base)};
  const NODE_W = 226, NODE_H = 74;

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

  const Legend = () => html`
    <div class="legend">
      <span><span class="dot" style="background:var(--artifact)"></span>artifact</span>
      <span><span class="dot" style="background:var(--run)"></span>run</span>
      <span><span class="dot" style="background:var(--app)"></span>app</span>
    </div>`;

  const Sidebar = ({{ artifacts, filter, onFilter, selected, onSelect }}) => {{
    const q = filter.trim().toLowerCase();
    const visible = q ? artifacts.filter((a) => a.name.toLowerCase().includes(q)) : artifacts;
    return html`
      <aside class="sidebar">
        <div class="sidebar-head">
          <h1>Artifact lineage</h1>
          <p class="sub">Every producer back to the origin, every consumer downstream —
            built live from artifact provenance and the
            <code>{LABEL_UPSTREAM_ARTIFACT_NAME}</code>/<code>{LABEL_UPSTREAM_ARTIFACT_VERSION}</code> labels.</p>
          <input class="search" placeholder="Filter artifacts…" value=${{filter}}
                 onInput=${{(e) => onFilter(e.target.value)}} />
        </div>
        <div class="sidebar-list">
          ${{visible.length === 0
            ? html`<div class="empty-list">${{artifacts.length === 0
                ? "No artifacts published yet." : "No artifacts match your filter."}}</div>`
            : visible.map((a) => html`
                <div key=${{a.name}} class="art-item ${{selected && selected.name === a.name ? "active" : ""}}"
                     onClick=${{() => onSelect(a.name)}}>
                  <div class="art-row1">
                    <span class="art-name">${{a.name}}</span>
                    <span class="art-count">${{a.versions}}</span>
                  </div>
                  <span class="art-desc">${{a.description || a.latest_version}}</span>
                </div>`)}}
        </div>
      </aside>`;
  }};

  const App = () => {{
    const [artifacts, setArtifacts] = React.useState([]);
    const [filter, setFilter] = React.useState("");
    const [selected, setSelected] = React.useState(BOOT.preselect || null);
    const [graph, setGraph] = React.useState(null);
    const [status, setStatus] = React.useState(selected ? "loading" : "idle");

    React.useEffect(() => {{
      fetch(BOOT.artifactsUrl, {{ cache: "no-store" }})
        .then((r) => r.json()).then(setArtifacts)
        .catch((e) => console.warn("artifact list fetch failed", e));
    }}, []);

    const select = React.useCallback((name, version) => {{
      setSelected({{ name, version: version || "latest" }});
      const qs = version && version !== "latest" ? `?version=${{encodeURIComponent(version)}}` : "";
      history.replaceState(null, "", `/lineage/artifact/${{encodeURIComponent(name)}}${{qs}}`);
    }}, []);

    React.useEffect(() => {{
      if (!selected) return;
      setStatus("loading");
      const v = selected.version && selected.version !== "latest" ? `?version=${{selected.version}}` : "";
      fetch(`${{BOOT.graphUrlBase}}/${{encodeURIComponent(selected.name)}}${{v}}`, {{ cache: "no-store" }})
        .then((r) => {{ if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }})
        .then((g) => {{ setGraph(g); setStatus("ok"); }})
        .catch((e) => {{ console.warn("graph fetch failed", e); setStatus("error"); }});
    }}, [selected]);

    const {{ flowNodes, flowEdges }} = React.useMemo(
      () => (graph ? layout(graph.nodes, graph.edges, graph.root) : {{ flowNodes: [], flowEdges: [] }}),
      [graph]
    );
    const rootNode = graph ? graph.nodes[graph.root] : null;

    return html`
      <${{Sidebar}} artifacts=${{artifacts}} filter=${{filter}} onFilter=${{setFilter}}
          selected=${{selected}} onSelect=${{(name) => select(name)}} />
      <div class="main">
        <div class="main-head">
          ${{rootNode ? html`
            <h2>${{rootNode.name}}</h2>
            <span class="version">${{rootNode.version}}</span>
            ${{rootNode.description ? html`<span class="desc">${{rootNode.description}}</span>` : null}}
            ${{rootNode.url ? html`<a class="console-link" target="_blank" rel="noopener"
                href=${{CONSOLE_BASE + rootNode.url}}>Open in console ↗</a>` : null}}
          ` : html`<h2>Select an artifact</h2>`}}
          <${{Legend}} />
        </div>
        <div class="canvas">
          ${{status === "idle" ? html`<div class="placeholder">
              Select an artifact from the sidebar to view its lineage.</div>` : null}}
          ${{status === "loading" ? html`<div class="placeholder">Loading lineage…</div>` : null}}
          ${{status === "error" ? html`<div class="placeholder">Could not build the lineage graph —
              is the artifact name/version correct? Check app logs.</div>` : null}}
          ${{status === "ok" ? html`
            <${{ReactFlow}} key=${{selected.name + ":" + selected.version}} nodes=${{flowNodes}}
                edges=${{flowEdges}} nodeTypes=${{nodeTypes}}
                fitView fitViewOptions=${{{{ padding: 0.15, maxZoom: 1 }}}} minZoom=${{0.2}}
                proOptions=${{{{ hideAttribution: true }}}} nodesConnectable=${{false}} colorMode="dark">
              <${{Background}} variant=${{BackgroundVariant.Dots}} gap=${{22}} size=${{1.4}} color="#26262e" />
              <${{Controls}} showInteractive=${{false}} />
            <//>` : null}}
        </div>
      </div>`;
  }};
  createRoot(document.getElementById("shell")).render(html`<${{App}} />`);
}} catch (err) {{
  console.error(err);
  document.getElementById("shell").innerHTML =
    '<div class="placeholder" style="position:static;height:100%;">Dashboard failed to load ' +
    '(CDN unreachable?) — see <a href="/lineage/artifacts">/lineage/artifacts</a> and ' +
    '<a href="/lineage/graph/&lt;name&gt;">/lineage/graph/&lt;name&gt;</a> for the raw JSON.</div>';
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
    consumers need the `upstream-artifact-name`/`upstream-artifact-version`
    labels to be discoverable at all.

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
    # `upstream-artifact-name`/`upstream-artifact-version` labels already find (see `_lineage.py`).
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
        async def lineage_page(artifact: str = "", version: str = "latest") -> str:
            preselect = {"name": artifact, "version": version} if artifact else None
            return _render_dashboard_html(env.name, env._console_base(), preselect=preselect)

        @app.get("/lineage/artifacts")
        async def lineage_artifacts() -> list[dict]:
            try:
                _ensure_client()
                return await env._artifact_groups()
            except Exception:
                logger.exception("Failed to list artifacts")
                return []

        @app.get("/lineage/graph/{name}")
        async def lineage_graph(name: str, version: str = "latest") -> dict:
            return await env._graph_json(name, version)

        @app.get("/lineage/artifact/{name}", response_class=fastapi.responses.HTMLResponse)
        async def lineage_artifact(name: str, version: str = "latest") -> str:
            return _render_dashboard_html(env.name, env._console_base(), preselect={"name": name, "version": version})

        return app
