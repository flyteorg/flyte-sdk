"""Flyte v2 usage metering task with an interactive HTML report.

Sweeps every project/domain/run/action reachable on the backend, records
requested CPU/memory/GPU and per-phase durations for each action (including
traces and plugin tasks), and renders an interactive report (flyte.report)
with:

  - filters: date range, project, domain, user, task type
  - group-by: project / domain / user / run / task / task type, bucketed by
    day / month / year
  - editable assumptions: default requests for tasks that declared none
    (k8s admission defaults are not visible via the Flyte API)
  - drill-down: group -> runs -> full action tree (sub-actions and traces)
  - fanout aggregation: identical leaf actions under one parent collapse into
    a single counted row (metrics stay exact; memory and report size stay
    bounded even for 200k-wide fanouts)
  - crash recovery: the scope list, the run list, and each 50-run sweep batch
    are @flyte.trace checkpoints replayed on retries; if the sweep OOMs, the
    report task re-runs it once in a bigger container

Run it (the report appears in the Flyte UI):
    flyte run utilization_report.py usage_report [--start YYYY-MM-DD] [--end YYYY-MM-DD]
        [--project P] [--domain D] [--all-scopes]
or with raw python (init from a config, then submit the same task):
    python utilization_report.py --config ~/.union/config.yaml [--all] [--start ...]

Default scope is the project/domain the task runs in; default window is the
last 12 full months plus the current month.
"""

import argparse
import asyncio
import json
import os
import re
import sys

import flyte
import flyte.errors
import flyte.report

env = flyte.TaskEnvironment(
    name="usage_profiler",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

CONCURRENCY = 24  # concurrent action-detail RPCs per batch
RUN_CONCURRENCY = 8  # concurrent runs being processed per batch
BATCH_RUNS = 50  # runs per checkpointed sweep batch (one @flyte.trace each)
BATCH_CONCURRENCY = 4  # sweep batches in flight at once (traces gather safely)
SCOPE_CONCURRENCY = 8  # project/domain run listings in flight at once
RUNS_PAGE = 500  # list_runs page size
ACTIONS_PAGE = 1000  # list_actions page size — the server honors large pages
# at ~the same per-page latency as 100, so this cuts a 200k-action run from
# 2000 sequential round trips to 200
OOM_RETRY_RESOURCES = flyte.Resources(cpu=2, memory="8Gi")

_MEM_UNITS = {
    "": 1,
    "k": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "Pi": 2**50,
}


def parse_cpu(v: str) -> float:
    if not v:
        return 0.0
    v = v.strip()
    if v.endswith("m"):
        return float(v[:-1]) / 1000
    return float(v)


def parse_mem_gib(v: str) -> float:
    if not v:
        return 0.0
    m = re.fullmatch(r"([0-9.]+)\s*([A-Za-z]*)", v.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _MEM_UNITS.get(m.group(2), 1) / 2**30


def extract_resources(detail_dict: dict) -> dict:
    """Requested resources from the action's task template.

    cpu/mem are None when the task declared no request (k8s namespace
    defaults apply at admission and are invisible to the Flyte API) —
    the report substitutes editable assumed values for those.
    """
    tmpl = (detail_dict.get("task") or {}).get("taskTemplate") or {}
    container = tmpl.get("container") or {}
    out = {
        "cpu": None,
        "mem": None,
        "gpu": 0.0,
        "gd": "",
        "hc": bool(container),
        "tmpl_type": str(tmpl.get("type", "")),
        "has_tmpl": bool(tmpl),
    }
    for r in (container.get("resources") or {}).get("requests") or []:
        name, value = r.get("name"), r.get("value", "")
        if name == "CPU":
            out["cpu"] = parse_cpu(value)
        elif name == "MEMORY":
            out["mem"] = parse_mem_gib(value)
        elif name == "GPU":
            out["gpu"] = float(value or 0)
    acc = (tmpl.get("extendedResources") or {}).get("gpuAccelerator") or {}
    out["gd"] = acc.get("device", "")
    return out


def resolve_window(start: str = "", end: str = ""):
    """Parse YYYY-MM-DD bounds; default = last 12 full months + current month."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    if start:
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    else:
        y, m = now.year, now.month - 12
        if m <= 0:
            y, m = y - 1, m + 12
        start_dt = datetime(y, m, 1, tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None
    return start_dt, end_dt


@flyte.trace
async def load_scopes(project: str = "", domain: str = "") -> list[list[str]]:
    """Project/domain pairs to sweep.

    Traced so retries replay the exact same scope list — a re-listed set could
    shift the run list and misalign the sweep-batch checkpoints downstream.
    """
    from flyte.remote import Project

    scopes: list[list[str]] = []
    async for p in Project.listall.aio():
        pd = p.to_dict()
        pname = str(pd.get("id") or pd.get("name") or "")
        if not pname or (project and pname != project):
            continue
        for d in pd.get("domains") or []:
            dname = str(d.get("id") or "")
            if not dname or (domain and dname != domain):
                continue
            scopes.append([pname, dname])
    return scopes


@flyte.trace
async def load_runs(scopes: list[list[str]], start: str = "", end: str = "") -> dict:
    """All in-window runs across the scopes, plus resolved user names.

    start/end are ISO dates bounding by run start time (run listings are
    newest-first, so pages older than `start` are never fetched). Traced: the
    run list is pinned at first success, so the checkpointed sweep batches
    keep covering exactly these runs across retries.
    """
    from datetime import datetime, timezone

    from flyteidl2.common import identifier_pb2, list_pb2
    from flyteidl2.workflow import run_service_pb2

    from flyte._initialize import get_client, get_init_config
    from flyte.remote import User

    org = get_init_config().org
    cutoff = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

    user_names: dict[str, str] = {}
    try:
        me = await User.get.aio()
        subj = me.subject() if callable(me.subject) else me.subject
        name = me.name() if callable(me.name) else me.name
        if subj:
            user_names[str(subj)] = str(name or subj)
    except Exception:
        pass

    # List runs with the raw client: the SDK's Run.listall pagination stalls
    # after the first page (re-yields it), silently dropping older runs.
    # Scopes are independent, so they list concurrently; pagination within a
    # scope stays sequential (token chain). Results merge in scope order.
    scope_sem = asyncio.Semaphore(SCOPE_CONCURRENCY)

    async def list_scope(proj: str, dom: str) -> list[list[str]]:
        scope_runs: list[list[str]] = []
        seen: set[str] = set()
        token, pages = None, 0
        async with scope_sem:
            try:
                while True:
                    # timeout + repeated-token guard: a hung or looping page
                    # must not stall the whole sweep silently
                    resp = await asyncio.wait_for(
                        get_client().run_service.list_runs(
                            run_service_pb2.ListRunsRequest(
                                request=list_pb2.ListRequest(limit=RUNS_PAGE, token=token or ""),
                                org=org,
                                project_id=identifier_pb2.ProjectIdentifier(organization=org, domain=dom, name=proj),
                            )
                        ),
                        60,
                    )
                    pages += 1
                    if pages % 20 == 0:
                        print(f"  {proj}/{dom}: page {pages}, {len(scope_runs)} runs kept …", flush=True)
                    page_all_old = bool(resp.runs) and cutoff is not None
                    for r in resp.runs:
                        if r.action.status.HasField("start_time"):
                            started = r.action.status.start_time.ToDatetime(tzinfo=timezone.utc)
                            if cutoff is not None and started < cutoff:
                                continue
                            page_all_old = False
                            if end_dt is not None and started > end_dt:
                                continue
                        else:
                            page_all_old = False
                        if r.action.id.run.name in seen:
                            continue
                        seen.add(r.action.id.run.name)
                        eb = r.action.metadata.executed_by
                        user = eb.user.id.subject or eb.application.id.subject or "unknown"
                        # run protos carry the user's profile — resolve names for everyone
                        sp = eb.user.spec
                        label = f"{sp.first_name} {sp.last_name}".strip() or sp.email or sp.user_handle
                        if eb.user.id.subject and label:
                            user_names.setdefault(eb.user.id.subject, label)
                        elif eb.application.id.subject and eb.application.spec.name:
                            user_names.setdefault(eb.application.id.subject, eb.application.spec.name)
                        scope_runs.append([proj, dom, r.action.id.run.name, user])
                    if resp.token and resp.token == token:
                        print(f"  ! {proj}/{dom}: server repeated page token, stopping this scope", file=sys.stderr)
                        break
                    token = resp.token
                    # newest-first: once a whole page is older than the cutoff, stop
                    if not token or page_all_old:
                        break
            except Exception as e:
                print(f"  ! listing runs {proj}/{dom}: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"  {proj}/{dom}: {len(scope_runs)} runs", flush=True)
        return scope_runs

    per_scope = await asyncio.gather(*(list_scope(proj, dom) for proj, dom in scopes))
    runs: list[list[str]] = [r for scope_runs in per_scope for r in scope_runs]
    return {"runs": runs, "user_names": user_names}


def aggregate_fanout(out: list[dict]) -> list[dict]:
    """Collapse identical leaf actions under one parent into a counted row.

    Only actions that are nobody's parent are aggregated — the run's full
    action set is listed before this is called, so "has children" is decidable
    from the rows' parent links. A 200k-wide fanout of one task collapses to a
    single row carrying `cnt` and summed durations (day granularity is part of
    the group key), so every metric stays exact while memory and report size
    stay bounded. Anything with children keeps its own row for drill-down.
    """
    parents = {r["pa"] for r in out if r["pa"]}
    kept: list[dict] = []
    groups: dict[tuple, dict] = {}
    for r in out:
        if r["an"] in parents:
            kept.append(r)
            continue
        key = (
            r["pa"],
            r["at"],
            r["tt"],
            r["tn"],
            r["ph"],
            r["cs"],
            r["us"],
            r["cpu"],
            r["mem"],
            r["gpu"],
            r["gd"],
            r["hc"],
            r["rs"] > 0,
            (r["st"] or "")[:10],
        )
        g = groups.get(key)
        if g is None:
            groups[key] = {**r, "cnt": 1}
        else:
            g["cnt"] += 1
            for f in ("qs", "ins", "rs", "ts"):
                g[f] += r[f]
            g["att"] = max(g["att"], r["att"])
            if r["st"] and (not g["st"] or r["st"] < g["st"]):
                g["st"] = r["st"]
    return kept + list(groups.values())


@flyte.trace
async def sweep_batch(index: int, total: int, batch: list[list[str]]) -> list[dict]:
    """Sweep one batch of runs into fanout-aggregated report rows.

    Traced: each batch is a recovery checkpoint — on a task retry, completed
    batches replay from their recorded results instead of re-hitting the API.
    """
    import time

    from flyteidl2.common import identifier_pb2, list_pb2, phase_pb2
    from flyteidl2.workflow import run_definition_pb2, run_service_pb2

    from flyte._initialize import get_client, get_init_config
    from flyte.remote._action import ActionDetails

    org = get_init_config().org
    sem = asyncio.Semaphore(CONCURRENCY)
    run_sem = asyncio.Semaphore(RUN_CONCURRENCY)

    try:
        from flyteidl2.core import catalog_pb2

        CACHE_HIT = catalog_pb2.CatalogCacheStatus.Value("CACHE_HIT")
    except Exception:
        CACHE_HIT = 2  # flyteidl2.core.CatalogCacheStatus.CACHE_HIT

    def proto_common(a) -> dict:
        md, stt = a.metadata, a.status
        atype = (
            run_definition_pb2.ActionType.Name(md.action_type).replace("ACTION_TYPE_", "") if md.action_type else "TASK"
        )
        try:
            phase = phase_pb2.ActionPhase.Name(stt.phase).replace("ACTION_PHASE_", "")
        except Exception:
            phase = str(stt.phase)
        return {
            "an": a.id.name,
            "pa": md.parent,
            "at": atype,
            "tn": md.task.id.name,
            "tt": md.task.task_type,
            "ph": phase,
            "att": stt.attempts,
            "st": stt.start_time.ToJsonString() if stt.HasField("start_time") else "",
            "cs": stt.cache_status == CACHE_HIT,
        }

    def needs_details(c: dict) -> bool:
        # traces, conditions, and engine-run orchestration primitives never run a
        # container — the list proto already tells us everything billable. Cache
        # hits skip details only when the listing already names their type.
        if c["at"] != "TASK" or c["tt"].startswith("core-"):
            return False
        return not (c["cs"] and c["tt"])

    RPC_TIMEOUT = 60  # a single hung RPC must not wedge the sweep

    async def list_actions(proj: str, dom: str, run_name: str) -> list:
        run_id = identifier_pb2.RunIdentifier(org=org, project=proj, domain=dom, name=run_name)
        token, out = None, []
        while True:
            req = list_pb2.ListRequest(limit=ACTIONS_PAGE, token=token)
            resp = await asyncio.wait_for(
                get_client().run_service.list_actions(run_service_pb2.ListActionsRequest(request=req, run_id=run_id)),
                RPC_TIMEOUT,
            )
            out.extend(resp.actions)
            if resp.token and resp.token == token:
                print(f"  ! {run_name}: server repeated action page token", file=sys.stderr)
                break
            token = resp.token
            if not token:
                break
        return out

    stats = {"details": 0, "fast": 0}

    async def detail_row(proj, dom, run_name, user, common) -> dict | None:
        ident = identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(org=org, project=proj, domain=dom, name=run_name),
            name=common["an"],
        )
        async with sem:
            try:
                d = await asyncio.wait_for(ActionDetails.get_details.aio(ident), RPC_TIMEOUT)
            except Exception as e:
                print(f"  ! {run_name}/{common['an']}: {type(e).__name__}: {e}", file=sys.stderr)
                return None
        stats["details"] += 1
        dd = d.to_dict()
        meta = dd.get("metadata") or {}
        status = dd.get("status") or {}
        res = extract_resources(dd)
        # some backends return skeleton list protos; prefer detail metadata
        # (str() everywhere: unknown enum values dict-ify as ints on newer servers)
        action_type = str(meta.get("actionType") or "").replace("ACTION_TYPE_", "") or common["at"]
        # resolve the action type through every signal we have: detail metadata,
        # the list proto, the task template's own type, and finally "function"
        # for mapped/traced python functions (task id + funtionName, no template)
        task_type = (
            str((meta.get("task") or {}).get("taskType", ""))
            or common["tt"]
            or res["tmpl_type"]
            or ("function" if not res["has_tmpl"] and (meta.get("funtionName") or meta.get("task")) else "")
            or (action_type.lower() if action_type != "TASK" else "")
        )
        if action_type != "TASK" or task_type.startswith("core-"):
            res["hc"] = False
        if common["cs"] or status.get("cacheStatus") == "CACHE_HIT":
            res["hc"] = False

        def secs(prop):
            # phase-duration properties raise on UNSPECIFIED-phase actions
            try:
                td = getattr(d, prop)
                return round(td.total_seconds(), 3) if td else 0.0
            except Exception:
                return 0.0

        total_s = secs("runtime")
        if not total_s and status.get("durationMs"):
            total_s = round(int(status["durationMs"]) / 1000, 3)
        return {
            "pj": proj,
            "dm": dom,
            "rn": run_name,
            "an": common["an"],
            "pa": meta.get("parent", "") or common["pa"],
            "at": action_type,
            "tt": task_type,
            "tn": ((meta.get("task") or {}).get("id") or {}).get("name", "") or common["tn"],
            "ph": str(status.get("phase") or "").replace("ACTION_PHASE_", "") or common["ph"],
            "att": status.get("attempts", 0) or common["att"],
            "st": status.get("startTime", "") or common["st"],
            "cs": common["cs"] or (status.get("cacheStatus") == "CACHE_HIT"),
            "qs": secs("queued_time"),
            "ins": secs("initializing_time"),
            "rs": secs("running_time"),
            "ts": total_s,
            "cpu": res["cpu"],
            "mem": res["mem"],
            "gpu": res["gpu"],
            "gd": res["gd"],
            "hc": res["hc"],
            "us": user,
        }

    def fast_row(proj, dom, run_name, user, a, common) -> dict:
        stats["fast"] += 1
        return {
            "pj": proj,
            "dm": dom,
            "rn": run_name,
            **common,
            "tt": common["tt"] or (common["at"].lower() if common["at"] != "TASK" else ""),
            "qs": 0.0,
            "ins": 0.0,
            "rs": 0.0,
            "ts": round(a.status.duration_ms / 1000, 3) if a.status.duration_ms else 0.0,
            "cpu": None,
            "mem": None,
            "gpu": 0.0,
            "gd": "",
            "hc": False,
            "us": user,
        }

    rows: list[dict] = []
    done = {"runs": 0}
    t0 = time.monotonic()

    async def process_run(proj, dom, run_name, user):
        async with run_sem:
            try:
                actions = await list_actions(proj, dom, run_name)
            except Exception as e:
                print(f"  ! actions {run_name}: {e}", file=sys.stderr)
                actions = []
        pending, out = [], []
        for a in actions:
            common = proto_common(a)
            if needs_details(common):
                pending.append(common)
            else:
                out.append(fast_row(proj, dom, run_name, user, a, common))
        results = await asyncio.gather(*(detail_row(proj, dom, run_name, user, c) for c in pending))
        out.extend(r for r in results if r)
        n_raw = len(out)
        out = aggregate_fanout(out)
        if n_raw > 1000:
            print(f"    aggregated {run_name}: {n_raw} actions -> {len(out)} rows", flush=True)
        rows.extend(out)
        done["runs"] += 1
        if done["runs"] % 20 == 0 or done["runs"] == len(batch):
            el = time.monotonic() - t0
            print(
                f"  batch {index + 1}/{total}: {done['runs']}/{len(batch)} runs · {len(rows)} rows "
                f"({stats['details']} detail RPCs, {stats['fast']} fast) · {el:.0f}s",
                flush=True,
            )

    async def heartbeat():
        while True:
            await asyncio.sleep(20)
            el = time.monotonic() - t0
            print(
                f"  heartbeat: batch {index + 1}/{total} · {done['runs']}/{len(batch)} runs done · "
                f"{len(rows)} rows · {stats['details']} detail RPCs · {el:.0f}s",
                flush=True,
            )

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(process_run(*r) for r in batch))
    finally:
        hb.cancel()
    return rows


@env.task(retries=2)
async def collect_actions(start: str = "", end: str = "", project: str = "", domain: str = "") -> dict:
    """Sweep runs/actions into fanout-aggregated report rows.

    Built for crash recovery: the scope list, the run list, and every sweep
    batch are @flyte.trace checkpoints, so on a retry (retries=2 covers
    crashes and other system errors) completed steps replay from recorded
    results instead of re-hitting the API. OOM is deliberately NOT handled
    here — same memory would just OOM again — it escalates to the caller,
    which re-runs this task with a bigger container.
    """
    from flyte._initialize import get_init_config

    scopes = await load_scopes(project=project, domain=domain)
    print(f"scopes: {scopes}", flush=True)
    listing = await load_runs(scopes=scopes, start=start, end=end)
    runs, user_names = listing["runs"], listing["user_names"]
    print(f"runs to sweep: {len(runs)}", flush=True)

    batches = [runs[i : i + BATCH_RUNS] for i in range(0, len(runs), BATCH_RUNS)]
    # traces gather safely (see examples/stress/trace_fanout.py), so batches
    # run BATCH_CONCURRENCY at a time — one slow batch (a 200k-action fanout
    # run) no longer stalls the pipeline. Results merge in batch order.
    batch_sem = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def run_batch(i: int, b: list[list[str]]) -> list[dict]:
        async with batch_sem:
            return await sweep_batch(index=i, total=len(batches), batch=b)

    with flyte.group("run-sweep"):
        results = await asyncio.gather(*(run_batch(i, b) for i, b in enumerate(batches)))
    rows: list[dict] = [r for batch_rows in results for r in batch_rows]
    print(f"action rows: {len(rows)} (fanout-aggregated)", flush=True)
    return {
        "rows": rows,
        "users": user_names,
        "org": get_init_config().org,
        "window": [start or "", end or "now"],
        "scope": f"{project or 'all projects'}/{domain or 'all domains'}",
    }


@env.task(report=True)
async def usage_report(
    start: str = "",
    end: str = "",
    project: str = "",
    domain: str = "",
    all_scopes: bool = False,
    console_url: str = "",
) -> str:
    """start/end: YYYY-MM-DD (default: last 12 full months + current month).
    Default scope is the project/domain this task runs in; pass project/domain
    to target another, or all_scopes=True to sweep the whole org."""
    if all_scopes:
        project = domain = ""
    elif not project:
        project = os.environ.get("FLYTE_INTERNAL_EXECUTION_PROJECT", "")
        domain = domain or os.environ.get("FLYTE_INTERNAL_EXECUTION_DOMAIN", "")
    print(f"scope: {project or 'all projects'}/{domain or 'all domains'}", flush=True)
    start_dt, end_dt = resolve_window(start, end)
    kwargs = {
        "start": start_dt.date().isoformat() if start_dt else "",
        "end": end_dt.date().isoformat() if end_dt else "",
        "project": project,
        "domain": domain,
    }
    try:
        data = await collect_actions(**kwargs)
    except flyte.errors.OOMError as e:
        # deterministic OOM won't be fixed by a same-size retry — re-run the
        # sweep in a bigger container (traces don't carry into the new action)
        print(f"collect OOMed ({e.code}); retrying with {OOM_RETRY_RESOURCES}", flush=True)
        data = await collect_actions.override(resources=OOM_RETRY_RESOURCES)(**kwargs)
    # for run links in the report; falls back to client-side origin detection
    data["console"] = console_url.rstrip("/")
    html = build_html(data)
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()
    n_actions = sum(r.get("cnt", 1) for r in data["rows"])
    return f"metered {n_actions} actions ({len(data['rows'])} rows) across org {data['org']}"


def build_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return _TEMPLATE.replace("__PAYLOAD__", payload.replace("</", "<\\/"))


_TEMPLATE = r"""
<div id="fup-root">
<style>
#fup-root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --other:#898781;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: var(--page);
  display: block; padding: 20px; box-sizing: border-box;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) #fup-root {
    color-scheme: dark;
    --surface:#1a1a19; --page:#0d0d0d;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border: rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"] #fup-root {
  color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border: rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
#fup-root * { box-sizing: border-box; }
#fup-root h1 { font-size: 20px; font-weight: 600; margin: 0 0 2px; }
#fup-root .sub { color: var(--ink-2); font-size: 13px; margin-bottom: 16px; }
#fup-root .card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
}
#fup-root .rowflex { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: flex-end; }
#fup-root label.ctl { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--ink-2); }
#fup-root select, #fup-root input[type=number] {
  font: inherit; font-size: 13px; color: var(--ink);
  background: var(--surface); border: 1px solid var(--axis);
  border-radius: 6px; padding: 4px 8px; min-width: 90px;
}
#fup-root input[type=number] { width: 90px; }
#fup-root .chk { display: flex; align-items: center; gap: 6px; font-size: 12px;
  color: var(--ink-2); padding-bottom: 6px; }
#fup-root .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 14px; }
#fup-root .tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; }
#fup-root .tile.click { cursor: pointer; }
#fup-root .tile.click:hover:not(.sel) { border-color: var(--axis); }
#fup-root .tile.sel { border-color: var(--s1); }
#fup-root .tile .lbl { font-size: 12px; color: var(--ink-2); margin-bottom: 4px; }
#fup-root .tile .val { font-size: 26px; font-weight: 600; }
#fup-root .tile .note { font-size: 11px; color: var(--muted); margin-top: 3px; }
#fup-root .cardhead { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
#fup-root .cardhead h2 { font-size: 14px; font-weight: 600; margin: 0; }
#fup-root .toggle { font-size: 12px; color: var(--ink-2); background: none;
  border: 1px solid var(--axis); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
#fup-root .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px;
  font-size: 12px; color: var(--ink-2); }
#fup-root .legend .it { display: flex; align-items: center; gap: 5px; }
#fup-root .legend .sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
#fup-root table { border-collapse: collapse; width: 100%; font-size: 13px; }
#fup-root th { text-align: right; color: var(--ink-2); font-weight: 500;
  padding: 6px 10px; border-bottom: 1px solid var(--grid); }
#fup-root th:first-child, #fup-root td:first-child { text-align: left; }
#fup-root td { padding: 5px 10px; border-bottom: 1px solid var(--grid);
  text-align: right; font-variant-numeric: tabular-nums; }
#fup-root tr.grp td:first-child { cursor: pointer; }
#fup-root tr.sub td { color: var(--ink-2); font-size: 12.5px; }
#fup-root .caret { display: inline-block; width: 14px; color: var(--muted); }
#fup-root .gname { cursor: pointer; }
#fup-root .gname:hover { text-decoration: underline; }
#fup-root .indent { color: var(--muted); }
#fup-root .badge { font-size: 10.5px; border: 1px solid var(--axis); border-radius: 4px;
  padding: 0 4px; margin-left: 6px; color: var(--muted); }
#fup-root #tooltip {
  position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.18); max-width: 320px;
}
#fup-root #tooltip .tt-title { color: var(--ink-2); margin-bottom: 4px; }
#fup-root #tooltip .tt-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
#fup-root #tooltip .tt-key { width: 12px; height: 0; border-top: 3px solid; border-radius: 2px; }
#fup-root #tooltip .tt-val { font-weight: 600; }
#fup-root #tooltip .tt-name { color: var(--ink-2); }
#fup-root svg text { font-family: inherit; }
#fup-root .warn { font-size: 12px; color: var(--ink-2); margin-top: 8px; }
</style>

<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
  <div>
    <h1>Flyte usage metering report</h1>
    <div class="sub" id="meta"></div>
  </div>
  <div style="display:flex; gap:8px; flex-shrink:0;">
    <button class="toggle" id="csv-view"
      title="Export the chart and breakdown tables as currently filtered and grouped">Export view CSV</button>
    <button class="toggle" id="csv-full"
      title="Export every collected action row (ignores filters)">Export full CSV</button>
  </div>
</div>

<div class="card">
  <div class="rowflex" id="filters">
    <label class="ctl">Date range <select id="f-date">
      <option value="all">All time</option><option value="7">Last 7 days</option>
      <option value="30">Last 30 days</option><option value="90">Last 90 days</option>
      <option value="mtd">Month to date</option><option value="ytd">Year to date</option>
    </select></label>
    <label class="ctl">From <input type="date" id="f-from"></label>
    <label class="ctl">To <input type="date" id="f-to"></label>
    <label class="ctl">Project <select id="f-project"></select></label>
    <label class="ctl">Domain <select id="f-domain"></select></label>
    <label class="ctl">User <select id="f-user"></select></label>
    <label class="ctl">Action type <select id="f-tt"></select></label>
    <label class="ctl">Group by <select id="v-group">
      <option value="pj">Project</option><option value="dm">Domain</option>
      <option value="us">User</option><option value="rn">Run</option>
      <option value="tn">Task</option><option value="tt">Action type</option>
    </select></label>
    <label class="ctl">Time period <select id="v-bucket">
      <option value="day">Day</option><option value="month" selected>Month</option>
      <option value="year">Year</option>
    </select></label>
    <label class="ctl">Break down by <select id="v-stack">
      <option value="none">Aggregate</option>
      <option value="pj">Project</option><option value="dm">Domain</option>
      <option value="us">User</option><option value="rn">Run</option>
      <option value="tn">Task</option><option value="tt" selected>Action type</option>
      <option value="kind">Execution kind</option><option value="gd">GPU type</option>
    </select></label>
    <label class="ctl">Chart metric <select id="v-metric">
      <option value="cpu">CPU core-hours</option>
      <option value="mem">Memory GiB-hours</option><option value="gpu">GPU-hours</option>
      <option value="act" selected>Actions</option><option value="sec">Container run time</option>
    </select></label>
  </div>
</div>

<div class="card">
  <div class="cardhead"><h2>Assumptions (editable — recompute live)</h2></div>
  <div class="rowflex">
    <label class="ctl">Default CPU (cores)* <input type="number" id="a-cpu" value="1" min="0" step="0.1"></label>
    <label class="ctl">Default memory (GiB)* <input type="number" id="a-mem" value="2" min="0" step="0.25"></label>
  </div>
  <div class="warn">* Applied to container actions whose task declared no request — Kubernetes
  namespace defaults apply at admission but are not visible through the Flyte API.
  Every action counts toward the Actions metric; compute metrics accrue only for
  time a container actually ran. Cached results, traces, and orchestration
  primitives run by the engine itself (e.g. <span id="np-types"></span>)
  carry no cpu/mem/gpu.
  Plugin task types (ray, spark, …) are metered by their container request only;
  worker-group resources declared in plugin config are not included.</div>
</div>

<div class="tiles" id="tiles"></div>

<div class="card">
  <div class="cardhead"><h2 id="chart-title">Usage over time</h2>
    <span style="flex:1; margin-left:10px; font-size:12px; color:var(--muted); align-self:center;">
      click a column to zoom in · click a legend entry to filter or open</span>
    <button class="toggle" id="chart-table-toggle">Table view</button></div>
  <div id="chart"></div>
  <div id="chart-table" style="display:none; overflow-x:auto;"></div>
  <div class="legend" id="legend"></div>
</div>

<div class="card">
  <div class="cardhead"><h2 id="table-title">Breakdown</h2></div>
  <div style="overflow-x:auto"><table id="breakdown"></table></div>
  <div class="warn">Click a row to expand runs; click a run to expand its full
  action tree (sub-actions and traces).</div>
</div>

<div id="tooltip"></div>

<script>
(function(){
const DATA = __PAYLOAD__;
const rows = DATA.rows;
const userNames = DATA.users || {};
const $ = (id) => document.getElementById(id);
const SERIES = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const css = (v) => getComputedStyle($('fup-root')).getPropertyValue(v).trim();

const NONPOD = {};
rows.forEach(r => { if (!r.hc) NONPOD[r.tt || r.at.toLowerCase()] = 1; });
$('np-types').textContent = Object.keys(NONPOD).slice(0,4).join(', ') || 'none seen';

const uname = (s) => userNames[s] ? userNames[s] : (s.length > 14 ? s.slice(0,12) + '…' : s);
const GROUP_LABELS = {pj:'Project', dm:'Domain', us:'User', rn:'Run', tn:'Task', tt:'Action type'};
const FILTER_FOR = {pj:'f-project', dm:'f-domain', us:'f-user', tt:'f-tt'};
// last rendered chart/breakdown data, for the "Export view CSV" button
const viewState = {chart: null, breakdown: null};
const gval = (r, dim) => {
  const v = r[dim] || '(unknown)';
  return dim === 'us' ? uname(v) : v;
};

function consoleBase() {
  // the report iframe may be served from a storage origin (e.g. S3), so never
  // use the page's own URL — prefer the embedded console URL, then the
  // embedding page's origin
  if (DATA.console) return DATA.console.replace(/\/$/, '');
  try {
    const ao = window.location.ancestorOrigins;
    if (ao && ao.length) return ao[ao.length - 1];
  } catch (e) {}
  try { if (document.referrer) return new URL(document.referrer).origin; } catch (e) {}
  return '';
}
function runUrl(pj, dm, rn) {
  const path = '/v2/domain/' + encodeURIComponent(dm) + '/project/' +
    encodeURIComponent(pj) + '/runs/' + encodeURIComponent(rn);
  const base = consoleBase();
  return base ? base + path : path;
}

function fillSelect(id, values, all) {
  const sel = $(id); sel.textContent = '';
  const o = document.createElement('option'); o.value = ''; o.textContent = all; sel.appendChild(o);
  values.forEach(v => {
    const op = document.createElement('option'); op.value = v.v; op.textContent = v.t; sel.appendChild(op);
  });
}
const uniq = (dim) => [...new Set(rows.map(r => r[dim] || ''))].filter(Boolean).sort();
fillSelect('f-project', uniq('pj').map(v => ({v, t:v})), 'All projects');
fillSelect('f-domain', uniq('dm').map(v => ({v, t:v})), 'All domains');
fillSelect('f-user', uniq('us').map(v => ({v, t: uname(v)})), 'All users');
fillSelect('f-tt', uniq('tt').map(v => ({v, t:v})), 'All action types');

function assumptions() {
  return {defCpu: +$('a-cpu').value || 0, defMem: +$('a-mem').value || 0};
}

function metrics(r, a) {
  // every action counts toward usage; compute accrues only for container run
  // time. Aggregated fanout rows carry cnt actions and pre-summed durations,
  // so requested-resources x summed seconds stays exact.
  const cnt = r.cnt || 1;
  if (!r.hc || r.rs <= 0) return {sec:0, cpu:0, mem:0, gpu:0, act:cnt, assumed:false};
  const sec = r.rs;
  const cpu = (r.cpu == null ? a.defCpu : r.cpu) * sec / 3600;
  const mem = (r.mem == null ? a.defMem : r.mem) * sec / 3600;
  const gpu = (r.gpu || 0) * sec / 3600;
  return {sec, cpu, mem, gpu, act:cnt, assumed: r.cpu == null || r.mem == null};
}

// action kind, for the Actions chart breakdown
const kindOf = (r) =>
  r.cs ? 'cache hit'
  : r.at === 'TRACE' ? 'trace'
  : r.at === 'CONDITION' ? 'condition'
  : r.tt && r.tt.startsWith('core-') ? 'orchestration'
  : r.rs > 0 ? 'ran container'
  : 'no runtime';

function filtered() {
  const p = $('f-project').value, d = $('f-domain').value,
        u = $('f-user').value, t = $('f-tt').value, dr = $('f-date').value;
  // explicit From/To take precedence over the preset
  let lo = $('f-from').value ? new Date($('f-from').value + 'T00:00:00Z') : null;
  const hi = $('f-to').value ? new Date($('f-to').value + 'T23:59:59.999Z') : null;
  if (!lo && !hi) {
    const now = new Date();
    if (dr === 'mtd') lo = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
    else if (dr === 'ytd') lo = new Date(Date.UTC(now.getUTCFullYear(), 0, 1));
    else if (dr !== 'all') lo = new Date(now.getTime() - (+dr) * 86400e3);
  }
  return rows.filter(r =>
    (!p || r.pj === p) && (!d || r.dm === d) && (!u || r.us === u) &&
    (!t || r.tt === t) &&
    (!lo || (r.st && new Date(r.st) >= lo)) &&
    (!hi || (r.st && new Date(r.st) <= hi)));
}

// click-a-column drill-down: zoom the window to that bucket, one period finer
function zoomTo(x, bucket) {
  let from, to, nb;
  if (bucket === 'year') { from = x + '-01-01'; to = x + '-12-31'; nb = 'month'; }
  else if (bucket === 'month') {
    const [y, m] = x.split('-').map(Number);
    const last = new Date(Date.UTC(y, m, 0)).getUTCDate();
    from = x + '-01'; to = x + '-' + String(last).padStart(2, '0'); nb = 'day';
  } else { from = x; to = x; nb = 'day'; }
  $('f-from').value = from; $('f-to').value = to; $('v-bucket').value = nb;
  render();
}

const fmtH = (v) => v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
const fmtDur = (s) => s >= 3600 ? (s/3600).toFixed(1) + ' h' : s >= 60 ? (s/60).toFixed(1) + ' m' : s.toFixed(1) + ' s';

function bucketOf(st, mode) {
  if (!st) return '(no start)';
  if (mode === 'day') return st.slice(0, 10);
  if (mode === 'month') return st.slice(0, 7);
  return st.slice(0, 4);
}

function render() {
  const a = assumptions();
  const rs = filtered();
  const dim = $('v-group').value, bucket = $('v-bucket').value, metric = $('v-metric').value;

  // totals + tiles
  let tot = {cpu:0, mem:0, gpu:0, sec:0, act:0, compute:0, assumed:0};
  rs.forEach(r => {
    const m = metrics(r, a);
    tot.cpu += m.cpu; tot.mem += m.mem; tot.gpu += m.gpu; tot.sec += m.sec; tot.act += m.act;
    if (m.sec > 0) { tot.compute += m.act; if (m.assumed) tot.assumed += m.act; }
  });
  const tiles = [
    {l:'Actions', v: tot.act.toLocaleString(), m:'act',
     n: tot.compute + ' with compute · ' + (tot.act - tot.compute) + ' without compute'},
    {l:'CPU core-hours', v: fmtH(tot.cpu), m:'cpu', n: 'requested cores \u00d7 run time'},
    {l:'Memory GiB-hours', v: fmtH(tot.mem), m:'mem', n: 'requested GiB \u00d7 run time'},
    {l:'GPU-hours', v: fmtH(tot.gpu), m:'gpu', n: 'requested GPUs \u00d7 run time'},
    {l:'Container run time', v: fmtDur(tot.sec), m:'sec', n: tot.assumed + ' actions using assumed defaults'},
  ];
  const tl = $('tiles'); tl.textContent = '';
  tiles.forEach(t => {
    const d = document.createElement('div');
    d.className = 'tile' + (t.m ? ' click' : '') + (t.m === metric ? ' sel' : '');
    const l = document.createElement('div'); l.className = 'lbl'; l.textContent = t.l;
    const v = document.createElement('div'); v.className = 'val'; v.textContent = t.v;
    d.appendChild(l); d.appendChild(v);
    if (t.n) { const n = document.createElement('div'); n.className = 'note'; n.textContent = t.n; d.appendChild(n); }
    if (t.m) d.addEventListener('click', () => {
      $('v-metric').value = t.m; render();
    });
    tl.appendChild(d);
  });

  // grouped aggregates
  const groups = new Map();
  rs.forEach(r => {
    const g = gval(r, dim);
    if (!groups.has(g)) groups.set(g, {cpu:0, mem:0, gpu:0, act:0, n:0, sec:0, rows:[], raw: r[dim] || ''});
    const o = groups.get(g), m = metrics(r, a);
    o.cpu += m.cpu; o.mem += m.mem; o.gpu += m.gpu; o.act += m.act;
    o.sec += m.sec; o.n += m.act; o.rows.push(r);
  });
  const sorted = [...groups.entries()].sort((x, y) => y[1][metric] - x[1][metric]);

  renderChart(rs, sorted, dim, bucket, metric, a);
  renderTable(sorted, dim, metric, a, tot);
}

// ---------- chart ----------
let chartTableMode = false;
$('chart-table-toggle').addEventListener('click', () => {
  chartTableMode = !chartTableMode;
  $('chart').style.display = chartTableMode ? 'none' : '';
  $('chart-table').style.display = chartTableMode ? '' : 'none';
  $('chart-table-toggle').textContent = chartTableMode ? 'Chart view' : 'Table view';
});

const METRIC_LABEL = {cpu:'CPU core-hours', mem:'Memory GiB-hours', gpu:'GPU-hours',
  act:'Actions', sec:'Container run-time hours'};
const STACK_LABELS = {none:'', pj:'project', dm:'domain', us:'user', rn:'run',
  tn:'task', tt:'action type', kind:'execution kind', gd:'GPU type'};

function renderChart(rs, sorted, dim, bucket, metric, a) {
  const stack = $('v-stack').value;
  const stackKey =
    stack === 'none' ? (() => METRIC_LABEL[metric])
    : stack === 'kind' ? kindOf
    : stack === 'gd' ? ((r) => r.gd || (r.gpu > 0 ? 'unspecified' : 'no gpu'))
    : ((r) => gval(r, stack));
  const fmtV = metric === 'act' ? ((v) => Math.round(v).toLocaleString()) : fmtH;
  const valOf = (m) => metric === 'sec' ? m.sec / 3600 : m[metric];

  // series = top stack keys by metric total across the filtered rows
  const keyTotals = new Map();
  rs.forEach(r => {
    const m = metrics(r, a), v = valOf(m);
    if (metric !== 'act' && v <= 0) return;
    const k = stackKey(r);
    keyTotals.set(k, (keyTotals.get(k) || 0) + v);
  });
  const ranked = [...keyTotals.entries()].sort((x, y) => y[1] - x[1]).map(e => e[0]);
  const top = ranked.slice(0, 7);
  const useOther = ranked.length > 7;
  const names = useOther ? top.concat(['Other']) : top;
  const colors = names.map((n, i) => n === 'Other' && useOther ? css('--other') : css(SERIES[i]));

  const buckets = new Map();
  rs.forEach(r => {
    const m = metrics(r, a), v = valOf(m);
    if (metric !== 'act' && v <= 0) return;
    const b = bucketOf(r.st, bucket);
    if (!buckets.has(b)) buckets.set(b, {});
    const g0 = stackKey(r);
    const g = (useOther && !top.includes(g0)) ? 'Other' : g0;
    buckets.get(b)[g] = (buckets.get(b)[g] || 0) + v;
  });
  const xs = [...buckets.keys()].sort();

  $('chart-title').textContent = METRIC_LABEL[metric] +
    (stack === 'none' ? '' : ' by ' + STACK_LABELS[stack]) + ' per ' + bucket;
  viewState.chart = {
    title: $('chart-title').textContent,
    bucketLabel: bucket === 'day' ? 'Day' : bucket === 'month' ? 'Month' : 'Year',
    names, xs, buckets,
  };

  // legend — entries are clickable: runs open in the console, filterable
  // dimensions apply that filter
  const lg = $('legend'); lg.textContent = '';
  if (names.length > 1) names.forEach((n, i) => {
    const it = document.createElement('span'); it.className = 'it';
    const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = colors[i];
    it.appendChild(sw); it.appendChild(document.createTextNode(n));
    if (n !== 'Other') {
      if (stack === 'rn') {
        const r0 = rs.find(r => r.rn === n);
        if (r0) {
          it.classList.add('gname'); it.title = 'open run in the console';
          it.addEventListener('click', () => window.open(runUrl(r0.pj, r0.dm, r0.rn), '_blank'));
        }
      } else if (FILTER_FOR[stack]) {
        const r0 = rs.find(r => gval(r, stack) === n);
        if (r0 && r0[stack]) {
          it.classList.add('gname'); it.title = 'filter on ' + n;
          it.addEventListener('click', () => { $(FILTER_FOR[stack]).value = r0[stack]; render(); });
        }
      }
    }
    lg.appendChild(it);
  });

  // chart table twin
  const ct = $('chart-table'); ct.textContent = '';
  const t = document.createElement('table');
  const hr = document.createElement('tr');
  [bucket === 'day' ? 'Day' : bucket === 'month' ? 'Month' : 'Year', ...names].forEach(h => {
    const th = document.createElement('th'); th.textContent = h; hr.appendChild(th);
  });
  t.appendChild(hr);
  xs.forEach(x => {
    const tr = document.createElement('tr');
    const td0 = document.createElement('td'); td0.textContent = x; tr.appendChild(td0);
    names.forEach(n => {
      const td = document.createElement('td');
      td.textContent = fmtV(buckets.get(x)[n] || 0);
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });
  ct.appendChild(t);

  // svg
  const host = $('chart'); host.textContent = '';
  const W = Math.max(host.clientWidth || 800, 320), H = 260;
  const M = {t: 10, r: 10, b: 26, l: 56};
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', W); svg.setAttribute('height', H);

  const totals = xs.map(x => names.reduce((s, n) => s + (buckets.get(x)[n] || 0), 0));
  const maxV = Math.max(...totals, 1e-9);
  // nice ticks
  const step0 = maxV / 4, pow = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map(m => m * pow).find(s => s >= step0);
  const yMax = Math.ceil(maxV / step) * step;
  const y = (v) => M.t + ph - (v / yMax) * ph;

  for (let v = 0; v <= yMax + 1e-9; v += step) {
    const ln = document.createElementNS(svg.namespaceURI, 'line');
    ln.setAttribute('x1', M.l); ln.setAttribute('x2', W - M.r);
    ln.setAttribute('y1', y(v)); ln.setAttribute('y2', y(v));
    ln.setAttribute('stroke', v === 0 ? css('--axis') : css('--grid'));
    ln.setAttribute('stroke-width', '1'); svg.appendChild(ln);
    const tx = document.createElementNS(svg.namespaceURI, 'text');
    tx.setAttribute('x', M.l - 6); tx.setAttribute('y', y(v) + 4);
    tx.setAttribute('text-anchor', 'end'); tx.setAttribute('font-size', '11');
    tx.setAttribute('fill', css('--muted'));
    tx.textContent = v >= 1 || v === 0 ? (+v.toFixed(2)).toLocaleString() : v.toFixed(3);
    svg.appendChild(tx);
  }

  const band = pw / Math.max(xs.length, 1);
  const bw = Math.min(24, band * 0.6);
  const GAP = 2;

  xs.forEach((x, xi) => {
    const cx = M.l + band * xi + band / 2;
    let acc = 0;
    const segs = [];
    names.forEach((n, si) => {
      const v = buckets.get(x)[n] || 0;
      if (v <= 0) return;
      segs.push({n, si, v, y0: acc, y1: acc + v});
      acc += v;
    });
    segs.forEach((s, i) => {
      const topSeg = i === segs.length - 1;
      let yTop = y(s.y1), yBot = y(s.y0);
      if (i > 0) yBot -= GAP / 2;
      if (!topSeg) yTop += GAP / 2;
      const h = Math.max(yBot - yTop, 1);
      const xL = cx - bw / 2;
      let el;
      if (topSeg && h > 4) {
        el = document.createElementNS(svg.namespaceURI, 'path');
        const r = 4;
        el.setAttribute('d',
          'M' + xL + ',' + (yTop + h) + ' V' + (yTop + r) +
          ' Q' + xL + ',' + yTop + ' ' + (xL + r) + ',' + yTop +
          ' H' + (xL + bw - r) +
          ' Q' + (xL + bw) + ',' + yTop + ' ' + (xL + bw) + ',' + (yTop + r) +
          ' V' + (yTop + h) + ' Z');
      } else {
        el = document.createElementNS(svg.namespaceURI, 'rect');
        el.setAttribute('x', xL); el.setAttribute('y', yTop);
        el.setAttribute('width', bw); el.setAttribute('height', h);
      }
      el.setAttribute('fill', colors[s.si]);
      el.setAttribute('data-col', xi);
      svg.appendChild(el);
    });

    // x labels (thin out when crowded)
    const every = Math.ceil(xs.length / Math.max(Math.floor(pw / 70), 1));
    if (xi % every === 0) {
      const tx = document.createElementNS(svg.namespaceURI, 'text');
      tx.setAttribute('x', cx); tx.setAttribute('y', H - 8);
      tx.setAttribute('text-anchor', 'middle'); tx.setAttribute('font-size', '11');
      tx.setAttribute('fill', css('--muted')); tx.textContent = x;
      svg.appendChild(tx);
    }

    // column hit target (bigger than the marks)
    const hit = document.createElementNS(svg.namespaceURI, 'rect');
    hit.setAttribute('x', M.l + band * xi); hit.setAttribute('y', M.t);
    hit.setAttribute('width', band); hit.setAttribute('height', ph);
    hit.setAttribute('fill', 'transparent');
    hit.addEventListener('pointermove', (ev) => {
      svg.querySelectorAll('[data-col]').forEach(e =>
        e.setAttribute('opacity', e.getAttribute('data-col') == xi ? '1' : '0.55'));
      showTip(ev, x, segs.slice().reverse().map(s =>
        ({name: s.n, color: colors[s.si], val: fmtV(s.v)})));
    });
    hit.addEventListener('pointerleave', () => {
      svg.querySelectorAll('[data-col]').forEach(e => e.setAttribute('opacity', '1'));
      hideTip();
    });
    hit.style.cursor = 'pointer';
    hit.addEventListener('click', () => { hideTip(); zoomTo(x, bucket); });
    svg.appendChild(hit);
  });
  host.appendChild(svg);
}

function showTip(ev, title, items) {
  const tip = $('tooltip'); tip.textContent = '';
  const t = document.createElement('div'); t.className = 'tt-title'; t.textContent = title;
  tip.appendChild(t);
  items.forEach(it => {
    const r = document.createElement('div'); r.className = 'tt-row';
    const k = document.createElement('span'); k.className = 'tt-key'; k.style.borderTopColor = it.color;
    const v = document.createElement('span'); v.className = 'tt-val'; v.textContent = it.val;
    const n = document.createElement('span'); n.className = 'tt-name'; n.textContent = it.name;
    r.appendChild(k); r.appendChild(v); r.appendChild(n); tip.appendChild(r);
  });
  tip.style.display = 'block';
  const pad = 14;
  let x = ev.clientX + pad, ypx = ev.clientY + pad;
  if (x + tip.offsetWidth > window.innerWidth - 8) x = ev.clientX - tip.offsetWidth - pad;
  if (ypx + tip.offsetHeight > window.innerHeight - 8) ypx = ev.clientY - tip.offsetHeight - pad;
  tip.style.left = x + 'px'; tip.style.top = ypx + 'px';
}
function hideTip() { $('tooltip').style.display = 'none'; }

// ---------- breakdown table with drill-down ----------
function cellRow(cells, cls) {
  const tr = document.createElement('tr'); if (cls) tr.className = cls;
  cells.forEach((c, i) => {
    const td = document.createElement('td');
    if (c instanceof Node) td.appendChild(c); else td.textContent = c;
    tr.appendChild(td);
  });
  return tr;
}

function renderTable(sorted, dim, metric, a, tot) {
  viewState.breakdown = {dim, metric, sorted, tot};
  $('table-title').textContent = 'Breakdown by ' + GROUP_LABELS[dim].toLowerCase();
  const tbl = $('breakdown'); tbl.textContent = '';
  const hr = document.createElement('tr');
  [GROUP_LABELS[dim], 'Actions', 'Metered time', 'CPU core-h', 'Mem GiB-h', 'GPU-h',
   '% of ' + METRIC_LABEL[metric]].forEach(h => {
    const th = document.createElement('th'); th.textContent = h; hr.appendChild(th);
  });
  tbl.appendChild(hr);

  sorted.forEach(([g, o]) => {
    const nameCell = document.createElement('span');
    const caret = document.createElement('span'); caret.className = 'caret'; caret.textContent = '▸';
    nameCell.appendChild(caret);
    const nm = document.createElement('span'); nm.textContent = g;
    // clicking the name filters on it (or opens the run in the console UI);
    // clicking anywhere else on the row expands it
    if (dim === 'rn' && o.rows.length) {
      nm.className = 'gname';
      nm.title = 'open run in the console';
      nm.addEventListener('click', (ev) => {
        ev.stopPropagation();
        window.open(runUrl(o.rows[0].pj, o.rows[0].dm, o.rows[0].rn), '_blank');
      });
    } else if (FILTER_FOR[dim] && o.raw) {
      nm.className = 'gname';
      nm.title = 'filter on ' + g;
      nm.addEventListener('click', (ev) => {
        ev.stopPropagation();
        $(FILTER_FOR[dim]).value = o.raw; render();
      });
    }
    nameCell.appendChild(nm);
    const tr = cellRow([nameCell, o.n, fmtDur(o.sec), fmtH(o.cpu), fmtH(o.mem), fmtH(o.gpu),
      (tot[metric] > 0 ? (100 * o[metric] / tot[metric]).toFixed(1) : '0.0') + '%'], 'grp');
    tbl.appendChild(tr);
    let open = false, subRows = [];
    tr.addEventListener('click', () => {
      open = !open; caret.textContent = open ? '▾' : '▸';
      if (!open) { subRows.forEach(r => r.remove()); subRows = []; return; }
      subRows = renderRuns(tbl, tr, o.rows, a, metric);
    });
  });

  const totTr = cellRow(['Total', sorted.reduce((s, e) => s + e[1].n, 0), fmtDur(tot.sec),
    fmtH(tot.cpu), fmtH(tot.mem), fmtH(tot.gpu), '100%']);
  totTr.style.fontWeight = '600';
  tbl.appendChild(totTr);
}

function renderRuns(tbl, afterTr, groupRows, a, metric) {
  const byRun = new Map();
  groupRows.forEach(r => {
    const key = r.pj + '/' + r.dm + '/' + r.rn;
    if (!byRun.has(key)) byRun.set(key, {cpu:0, mem:0, gpu:0, act:0, sec:0, n:0, rows:[], rn:r.rn});
    const o = byRun.get(key), m = metrics(r, a);
    o.cpu += m.cpu; o.mem += m.mem; o.gpu += m.gpu; o.act += m.act; o.sec += m.sec; o.n += m.act; o.rows.push(r);
  });
  const runs = [...byRun.entries()].sort((x, y) => y[1][metric] - x[1][metric]).slice(0, 50);
  const made = [];
  let anchor = afterTr;
  runs.forEach(([key, o]) => {
    const nameCell = document.createElement('span');
    const caret = document.createElement('span'); caret.className = 'caret'; caret.textContent = '▸';
    const ind = document.createElement('span'); ind.className = 'indent'; ind.textContent = '\u00a0\u00a0\u00a0';
    nameCell.appendChild(ind); nameCell.appendChild(caret);
    nameCell.appendChild(document.createTextNode(key + ' '));
    const lk = document.createElement('span'); lk.className = 'gname'; lk.textContent = '↗';
    lk.title = 'open run in the console';
    lk.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const r0 = o.rows[0];
      window.open(runUrl(r0.pj, r0.dm, r0.rn), '_blank');
    });
    nameCell.appendChild(lk);
    const tr = cellRow([nameCell, o.n, fmtDur(o.sec), fmtH(o.cpu), fmtH(o.mem), fmtH(o.gpu), ''], 'sub grp');
    anchor.after(tr); anchor = tr; made.push(tr);
    let open = false, treeRows = [];
    tr.addEventListener('click', (ev) => {
      ev.stopPropagation();
      open = !open; caret.textContent = open ? '▾' : '▸';
      if (!open) { treeRows.forEach(r => r.remove()); treeRows = []; return; }
      treeRows = renderTree(tr, o.rows, a);
      made.push(...treeRows);
    });
  });
  if (byRun.size > 50) {
    const more = '\u00a0\u00a0\u00a0… ' + (byRun.size - 50) + ' more runs (filter to narrow)';
    const tr = cellRow([more, '', '', '', '', '', ''], 'sub');
    anchor.after(tr); made.push(tr);
  }
  return made;
}

function renderTree(afterTr, runRows, a) {
  // parent/child tree: roots have no parent or parent outside this run
  const byName = new Map(runRows.map(r => [r.an, r]));
  const children = new Map();
  const roots = [];
  runRows.forEach(r => {
    if (r.pa && byName.has(r.pa)) {
      if (!children.has(r.pa)) children.set(r.pa, []);
      children.get(r.pa).push(r);
    } else roots.push(r);
  });
  const made = [];
  let anchor = afterTr;
  const emit = (r, depth) => {
    const m = metrics(r, a);
    const nameCell = document.createElement('span');
    const ind = document.createElement('span'); ind.className = 'indent';
    ind.textContent = '\u00a0'.repeat(6 + depth * 3) + (depth >= 0 ? '└ ' : '');
    nameCell.appendChild(ind);
    nameCell.appendChild(document.createTextNode(r.an + (r.tn ? ' · ' + r.tn : '')));
    const badge = document.createElement('span'); badge.className = 'badge';
    badge.textContent = r.at === 'TRACE' ? 'trace' : (r.tt || r.at.toLowerCase());
    nameCell.appendChild(badge);
    if (r.cnt > 1) {
      const bc = document.createElement('span'); bc.className = 'badge';
      bc.textContent = '\u00d7' + r.cnt.toLocaleString();
      bc.title = r.cnt.toLocaleString() + ' identical leaf actions aggregated';
      nameCell.appendChild(bc);
    }
    if (r.hc && (r.cpu == null || r.mem == null)) {
      const b2 = document.createElement('span'); b2.className = 'badge'; b2.textContent = 'assumed req';
      nameCell.appendChild(b2);
    }
    const tr = cellRow([nameCell, r.att > 1 ? r.att + ' tries' : '', fmtDur(m.sec || r.ts),
      r.hc ? fmtH(m.cpu) : '—', r.hc ? fmtH(m.mem) : '—', r.gpu ? fmtH(m.gpu) : '—',
      r.ph.toLowerCase()], 'sub');
    anchor.after(tr); anchor = tr; made.push(tr);
    (children.get(r.an) || []).sort((x, y) => (x.st || '').localeCompare(y.st || ''))
      .forEach(c => emit(c, depth + 1));
  };
  roots.sort((x, y) => (x.st || '').localeCompare(y.st || '')).forEach(r => emit(r, 0));
  return made;
}

// ---------- CSV export ----------
const csvCell = (v) => {
  const s = String(v == null ? '' : v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};
const toCsv = (table) => table.map(r => r.map(csvCell).join(',')).join('\n') + '\n';
function downloadCsv(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'text/csv'}));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// every collected action row, unfiltered; computed hours use the current
// assumption inputs
$('csv-full').addEventListener('click', () => {
  const a = assumptions();
  const out = [[
    'project','domain','run','action','count','parent','action_type','task_type',
    'task_name','phase','attempts','start_time','cache_hit','execution_kind',
    'user_id','user_name','queued_s','initializing_s','running_s','total_s',
    'requested_cpu','requested_mem_gib','requested_gpu','gpu_device',
    'ran_container','assumed_defaults','cpu_core_hours','mem_gib_hours','gpu_hours',
  ]];
  rows.forEach(r => {
    const m = metrics(r, a);
    out.push([
      r.pj, r.dm, r.rn, r.an, r.cnt || 1, r.pa || '', r.at, r.tt || '', r.tn || '',
      r.ph, r.att, r.st, r.cs ? 1 : 0, kindOf(r),
      r.us, userNames[r.us] || '', r.qs, r.ins, r.rs, r.ts,
      r.cpu == null ? '' : r.cpu, r.mem == null ? '' : r.mem, r.gpu || 0, r.gd || '',
      r.hc ? 1 : 0, m.assumed && m.sec > 0 ? 1 : 0,
      m.cpu.toFixed(6), m.mem.toFixed(6), m.gpu.toFixed(6),
    ]);
  });
  downloadCsv('flyte-usage-' + (DATA.org || 'org') + '-full.csv', toCsv(out));
});

// the chart pivot and breakdown table exactly as currently rendered
// (filters, grouping, stacking, and assumptions applied)
$('csv-view').addEventListener('click', () => {
  const c = viewState.chart, b = viewState.breakdown;
  if (!c || !b) return;
  const out = [['# ' + c.title]];
  out.push([c.bucketLabel, ...c.names]);
  c.xs.forEach(x => out.push([x, ...c.names.map(n => c.buckets.get(x)[n] || 0)]));
  out.push([]);
  out.push(['# Breakdown by ' + GROUP_LABELS[b.dim].toLowerCase()]);
  out.push([GROUP_LABELS[b.dim], 'actions', 'metered_time_s', 'cpu_core_hours',
    'mem_gib_hours', 'gpu_hours', 'pct_of_' + b.metric]);
  b.sorted.forEach(([g, o]) => out.push([
    g, o.n, o.sec.toFixed(3), o.cpu.toFixed(6), o.mem.toFixed(6), o.gpu.toFixed(6),
    b.tot[b.metric] > 0 ? (100 * o[b.metric] / b.tot[b.metric]).toFixed(1) : '0.0',
  ]));
  out.push(['Total', b.sorted.reduce((s, e) => s + e[1].n, 0), b.tot.sec.toFixed(3),
    b.tot.cpu.toFixed(6), b.tot.mem.toFixed(6), b.tot.gpu.toFixed(6), '100.0']);
  downloadCsv('flyte-usage-' + (DATA.org || 'org') + '-view.csv', toCsv(out));
});

// meta line
(function(){
  const n = rows.reduce((s, r) => s + (r.cnt || 1), 0),
        runs = new Set(rows.map(r => r.pj + '/' + r.dm + '/' + r.rn)).size;
  const w = DATA.window || ['', ''];
  $('meta').textContent = 'Org "' + DATA.org + '" · scope ' + (DATA.scope || 'all') +
    ' · window ' + (w[0] || 'beginning') + ' → ' + (w[1] || 'now') +
    ' — ' + n + ' actions across ' + runs + ' runs';
})();

['f-date','f-from','f-to','f-project','f-domain','f-user','f-tt',
 'v-group','v-bucket','v-metric','v-stack',
 'a-cpu','a-mem'].forEach(id =>
  $(id).addEventListener(id.startsWith('a-') ? 'input' : 'change', render));
// choosing a preset clears any explicit From/To
$('f-date').addEventListener('change', () => {
  $('f-from').value = ''; $('f-to').value = ''; render();
});

// The console may lay this report out after the script runs (hidden tab,
// late-attached iframe) — render defensively and re-render when the root
// first gets real dimensions or changes width.
function safeRender() { try { render(); } catch (e) { console.error('render:', e); } }
let lastW = -1;
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => {
    const w = $('fup-root').clientWidth;
    if (w !== lastW) { lastW = w; safeRender(); }
  }).observe($('fup-root'));
} else {
  window.addEventListener('resize', safeRender);
}
window.addEventListener('load', safeRender);
setTimeout(safeRender, 400);
setTimeout(safeRender, 1500);
safeRender();
})();
</script>
</div>
"""


def main() -> None:
    """Thin submit shim — all logic lives in the task, so this is equivalent to:
    flyte run utilization_report.py usage_report [--start ...] [--project ...] [--all-scopes]
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/.union/config.yaml"))
    ap.add_argument("--start", default="", help="YYYY-MM-DD (default: 12 full months back)")
    ap.add_argument("--end", default="", help="YYYY-MM-DD (default: now)")
    ap.add_argument("--project", default="", help="sweep this project (default: where the task runs)")
    ap.add_argument("--domain", default="", help="sweep this domain (default: where the task runs)")
    ap.add_argument("--all", action="store_true", help="sweep every project/domain in the org")
    args = ap.parse_args()

    flyte.init_from_config(args.config)
    from flyte._initialize import get_init_config

    client = get_init_config().client
    console_url = (client.endpoint or "") if client else ""
    run = flyte.run(
        usage_report,
        start=args.start,
        end=args.end,
        project=args.project,
        domain=args.domain,
        all_scopes=args.all,
        console_url=console_url,
    )
    print(run.name, run.url)
    run.wait()
    print("phase:", run.phase)


if __name__ == "__main__":
    main()
