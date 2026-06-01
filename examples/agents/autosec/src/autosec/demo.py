"""AutoSec demo MVP — see SPEC.md §11.

A deliberately minimal, 5-10 minute slice of the "auto security ops" researcher.
It fans out a 4-stage Flyte pipeline across every bundled C file in `targets/`
(each with a planted memory-corruption bug) **in parallel**, then delegates the
high-security PoC validation to a **Daytona** VM sandbox (the SPEC §2.6
"external VM SaaS provider").

The point of the demo is the *orchestration story*, not finding a real 0-day.
Each Flyte feature below maps to a failure mode a naive `while True:` agent loop
gets wrong (see SPEC §11.4):

  A. LLM API timeouts        -> @env.task(retries=, timeout=)
  B. Hallucination/bad calls -> checkpoint & resume (@flyte.trace + @env.task)
  C. Infra failures (OOM)    -> user try/except + .override(resources=...)
  D. Cost runaway            -> resource limits + execution timeouts + teardown

Optional "demo beats" (toggle one between runs to show recovery deterministically):

    AUTOSEC_FORCE_LLM_TIMEOUT=1   # first attempt of hypothesize hangs -> timeout+retry
    AUTOSEC_FORCE_BAD_TOOL_CALL=1 # first attempt returns garbage JSON -> raise+resume
    AUTOSEC_FORCE_OOM=1           # first whole-program scan OOMs -> bigger box + fallback
    AUTOSEC_FORCE_ALL=1           # activate all three at once (beats are staggered by
                                  #   attempt so each one is actually demonstrated)

Run (from ``examples/agents/autosec``)::

    export ANTHROPIC_API_KEY=sk-...
    export DAYTONA_API_KEY=dtn-...      # Daytona sandbox provisioned separately
    uv run autosec                       # console script (see pyproject.toml)
    # or: uv run python -m autosec.demo
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import pathlib
import re
from typing import Any

import flyte
import flyte.report

HERE = pathlib.Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent  # .../examples/agents/autosec (holds pyproject.toml)
TARGETS_DIR = HERE / "targets"
MODEL = os.getenv("AUTOSEC_MODEL", "claude-haiku-4-5")

# `include=[targets/]` bundles the whole `targets/` directory into the code bundle
# next to this module, so `TARGETS_DIR` resolves at runtime (the task loads from
# the code bundle, not from the installed site-packages copy). Dependencies
# (flyte, litellm, daytona) come from pyproject.toml via with_uv_project; the
# .dockerignore keeps secrets like .env out of the image build context.
env = flyte.TaskEnvironment(
    name="autosec-demo",
    image=(
        flyte.Image.from_debian_base()
        .with_dockerignore(PROJECT_ROOT / ".dockerignore")
        .with_uv_project(PROJECT_ROOT / "pyproject.toml", project_install_mode="install_project")
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    # Absolute path: relative includes are anchored at the env's "declaring file",
    # which is mis-detected when a project-local .venv lives under examples/. An
    # absolute path bypasses that anchoring; the directory is bundled next to this module.
    include=[str(TARGETS_DIR)],
    secrets=[
        # Create these on your Flyte backend (or run locally with the env vars set).
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="internal-daytona-api-key", as_env_var="DAYTONA_API_KEY"),
    ],
)


def _attempt() -> int:
    """0 on the first try, 1+ on retries. Used to make demo beats deterministic."""
    tc = flyte.ctx()
    return tc.attempt_number if tc is not None else 0


def _force(flag: str) -> bool:
    """True if a specific demo-beat flag is set, or the master AUTOSEC_FORCE_ALL is set."""
    return bool(os.getenv(flag) or os.getenv("AUTOSEC_FORCE_ALL"))


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply; raise if there is none."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    blob = match.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # LLMs frequently emit invalid JSON escapes (e.g. '\0' NUL terminators,
        # '\x..' in C snippets). Escape any backslash that isn't a valid JSON
        # escape, then retry.
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", blob)
        return json.loads(fixed)


# --- LLM helper -------------------------------------------------------------
# @flyte.trace makes this a checkpoint boundary: once a call succeeds its output
# is memoized, so a later task retry (beat B) does NOT re-run or re-bill it.
@flyte.trace
async def call_llm(prompt: str, *, force_timeout: bool = False) -> str:
    if force_timeout:
        # Beat A: hang longer than the task timeout so Flyte kills + retries us.
        await asyncio.sleep(600)

    from litellm import acompletion

    resp = await acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        timeout=25,  # per-request ceiling; the task timeout is the outer guard
    )
    return resp["choices"][0]["message"]["content"]


# --- Stage 1: static analysis (CPU, OOM-prone) ------------------------------
@env.task(retries=2, timeout=30)
async def scan_static(source: str, scope: str = "whole") -> str:
    """Cheap stand-in for whole-program analysis (Joern/CodeQL in the real system)."""
    try:
        # Beat C: simulate an OOM on the first whole-program attempt only.
        if scope == "whole" and _force("AUTOSEC_FORCE_OOM") and _attempt() == 0:
            raise MemoryError("whole-program graph exceeded memory limit")
        findings = _grep_dangerous_calls(source)
        return findings or "(no dangerous-call sites found)"
    except MemoryError as exc:
        # Infra-level recovery: re-dispatch the SAME task with a bigger box and a
        # narrower (file-scoped) analysis. Flyte lets user code change the infra
        # profile of a retry via .override(resources=...). scope="file" avoids
        # re-triggering the simulated OOM, so this terminates.
        print(f"[scan_static] {exc}; escalating resources + narrowing scope")
        return await scan_static.override(
            short_name="scan_static_more_resources", resources=flyte.Resources(cpu=2, memory="4Gi")
        )(source, scope="file")


def _grep_dangerous_calls(source: str) -> str:
    hits = []
    for i, line in enumerate(source.splitlines(), start=1):
        for fn in ("strcpy", "strcat", "sprintf", "gets", "memcpy"):
            if fn in line:
                hits.append(f"L{i}: {fn} -> {line.strip()}")
    return "\n".join(hits)


# --- Stage 2: hypothesize the vulnerability (LLM) ---------------------------
@env.task(retries=3, timeout=20)
async def hypothesize(source: str, static_findings: str) -> dict:
    prompt = (
        "You are a vulnerability researcher. Decide whether this C source contains "
        "an exploitable memory-corruption bug reachable from argv. Reply with ONLY "
        "a JSON object.\n"
        'If vulnerable: {"vulnerable": true, "function": str, '
        '"buffer_size": int (bytes of the overflowable buffer), "vuln_class": str, '
        '"reasoning": str}.\n'
        "If the code looks safe (bounded copies, length checks, snprintf/strlcpy, "
        'etc.): {"vulnerable": false, "reasoning": str}.\n\n'
        f"SOURCE:\n{source}\n\nDANGEROUS CALLS:\n{static_findings}\n"
    )
    # Beat A: hang on the first attempt -> task timeout -> retry.
    timeout_on = _force("AUTOSEC_FORCE_LLM_TIMEOUT")
    bad_on = _force("AUTOSEC_FORCE_BAD_TOOL_CALL")
    raw = await call_llm(prompt, force_timeout=timeout_on and _attempt() == 0)

    # Beat B: simulate a hallucinated/malformed tool call. When the timeout beat is
    # also active it consumes attempt 0, so defer this to attempt 1 — that way both
    # beats are actually demonstrated in a single run (e.g. with AUTOSEC_FORCE_ALL).
    bad_attempt = 1 if timeout_on else 0
    if bad_on and _attempt() == bad_attempt:
        raw = "Sure! The bug is somewhere around here, trust me."

    hyp = _extract_json(raw)  # malformed -> raises -> task retries -> resumes
    if "vulnerable" not in hyp:
        # Back-compat: a bare hypothesis with a buffer_size implies a finding.
        hyp["vulnerable"] = "buffer_size" in hyp
    if hyp.get("vulnerable") and "buffer_size" not in hyp:
        raise ValueError(f"vulnerable hypothesis missing buffer_size: {hyp}")
    return hyp


# --- Stage 3: build a proof-of-concept --------------------------------------
@env.task(retries=2, timeout=90)
async def build_poc(hypothesis: dict) -> dict:
    """Construct a trigger input that overflows the identified buffer."""
    buffer_size = int(hypothesis.get("buffer_size", 64))
    payload_len = buffer_size + 64  # comfortably past the saved return address
    return {
        "payload_len": payload_len,
        "payload_repr": f'"A" * {payload_len}',
        "target_function": hypothesis.get("function", "greet"),
    }


# --- Stage 4: validate in a Daytona VM (delegated high-security step) --------
@env.task(retries=2, timeout=300)
async def validate_in_daytona(source: str, poc: dict) -> dict:
    """Compile + run the target with the PoC input inside an isolated Daytona VM.

    The exploit code executes in Daytona's sandbox, never on the Flyte node
    (SPEC §2.6 / §7). The VM is torn down in `finally` regardless of outcome
    (SPEC VD-5) so a stuck or failed run cannot leak a billable VM.
    """
    from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig

    api_key = os.environ["DAYTONA_API_KEY"]
    daytona = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = daytona.create(CreateSandboxFromSnapshotParams(ephemeral=True, auto_stop_internal=1))
    try:
        driver = _build_sandbox_driver(source, int(poc["payload_len"]))
        resp = sandbox.process.code_run(driver)
        triggered = "SIGSEGV" in (resp.result or "") or resp.exit_code not in (0, None)
        return {
            "triggered": bool(triggered),
            "sandbox_exit_code": resp.exit_code,
            "log": resp.result,
        }
    finally:
        sandbox.delete()  # guaranteed teardown (VD-5)


def _build_sandbox_driver(source: str, payload_len: int) -> str:
    """Python program that runs INSIDE the Daytona VM: compile + fire the PoC."""
    b64 = base64.b64encode(source.encode()).decode()
    return (
        "import base64, subprocess\n"
        f"open('target.c','w').write(base64.b64decode('{b64}').decode())\n"
        "c = subprocess.run(['gcc','-fno-stack-protector','-w','-o','target','target.c'],"
        " capture_output=True, text=True)\n"
        "if c.returncode != 0:\n"
        "    print('COMPILE_FAILED'); print(c.stderr)\n"
        "else:\n"
        f"    r = subprocess.run(['./target', 'A'*{payload_len}], capture_output=True, text=True)\n"
        "    rc = r.returncode\n"
        "    print('EXIT', rc)\n"
        "    if rc and rc < 0:\n"
        "        import signal\n"
        "        print('SIGNAL', signal.Signals(-rc).name)\n"
        "        if -rc == signal.SIGSEGV: print('SIGSEGV')\n"
        "    print(r.stdout); print(r.stderr)\n"
    )


# --- Orchestration ----------------------------------------------------------
def _load_targets() -> dict[str, str]:
    """Read every ``targets/*.c`` file into a {name: source} mapping."""
    return {p.name: p.read_text() for p in sorted(TARGETS_DIR.glob("*.c"))}


@env.task
async def analyze_target(name: str, source: str) -> dict:
    """Run the full pipeline against a single target file.

    Secure targets short-circuit after `hypothesize`: no PoC is built and no
    Daytona VM is spun up (saving cost), and the report marks them as clean.
    """
    findings = await scan_static(source)
    hypothesis = await hypothesize(source, findings)

    if not hypothesis.get("vulnerable"):
        poc: dict = {}
        verdict = {"triggered": False, "skipped": True}
    else:
        poc = await build_poc(hypothesis)
        verdict = await validate_in_daytona(source, poc)

    return {
        "target": name,
        "static_findings": findings,
        "hypothesis": hypothesis,
        "poc": poc,
        "verdict": verdict,
    }


_REPORT_CSS = """
<style>
  .autosec { --bg:#ffffff; --card:#f7f8fa; --line:#e3e7ec; --muted:#5b6675;
    --text:#1b2330; --red:#c0392b; --amber:#b6791f; --green:#1e7e34; --accent:#1f6feb;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--text); padding:24px; border-radius:12px;
    border:1px solid var(--line); }
  .autosec h2 { margin:0 0 4px; font-size:20px; letter-spacing:.2px; }
  .autosec .sub { color:var(--muted); font-size:13px; margin:0 0 20px; }
  .autosec .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:22px; }
  .autosec .card { background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:14px 18px; min-width:120px; }
  .autosec .card .n { font-size:26px; font-weight:700; line-height:1; }
  .autosec .card .l { color:var(--muted); font-size:12px; margin-top:6px;
    text-transform:uppercase; letter-spacing:.6px; }
  .autosec table { width:100%; table-layout:fixed; border-collapse:collapse; font-size:13px;
    background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .autosec th, .autosec td { overflow-wrap:anywhere; }
  .autosec thead th { background:#eef1f5; color:var(--muted); text-align:left;
    font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.6px;
    padding:11px 14px; border-bottom:1px solid var(--line); }
  .autosec tbody td { padding:11px 14px; border-bottom:1px solid var(--line);
    vertical-align:top; }
  .autosec tbody tr:last-child td { border-bottom:none; }
  .autosec tbody tr:nth-child(even) { background:#fafbfc; }
  .autosec tbody tr:hover { background:#eef4ff; }
  .autosec code { background:#eef1f5; border:1px solid var(--line); border-radius:5px;
    padding:1px 6px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
  .autosec .num { text-align:right; font-variant-numeric:tabular-nums; }
  .autosec .reason { color:var(--muted); line-height:1.45; }
  .autosec .badge { display:inline-block; padding:3px 10px; border-radius:999px;
    font-size:11px; font-weight:700; letter-spacing:.4px; white-space:nowrap; }
  .autosec .b-exploited { background:rgba(192,57,43,.10); color:var(--red);
    border:1px solid rgba(192,57,43,.35); }
  .autosec .b-vuln { background:rgba(182,121,31,.12); color:var(--amber);
    border:1px solid rgba(182,121,31,.35); }
  .autosec .b-secure { background:rgba(30,126,52,.10); color:var(--green);
    border:1px solid rgba(30,126,52,.35); }
  /* Column widths: REASONING is the widest. */
  .autosec col.c-target  { width:15%; }
  .autosec col.c-status  { width:11%; }
  .autosec col.c-class   { width:11%; }
  .autosec col.c-fn      { width:11%; }
  .autosec col.c-buf     { width:8%; }
  .autosec col.c-payload { width:8%; }
  .autosec col.c-exit    { width:6%; }
  .autosec col.c-reason  { width:30%; }
  /* Per-target tab elements */
  .autosec .kv { display:flex; flex-wrap:wrap; gap:12px 28px; margin:16px 0 4px; }
  .autosec .kv .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .autosec .kv .v { font-weight:600; font-size:14px; margin-top:3px; }
  .autosec .section-label { font-size:11px; text-transform:uppercase; letter-spacing:.6px;
    color:var(--muted); margin:22px 0 7px; }
  .autosec .reason-block { background:var(--card); border:1px solid var(--line);
    border-radius:8px; padding:13px 15px; line-height:1.5; font-size:13px; }
  .autosec pre.code { background:#f6f8fa; border:1px solid var(--line); border-radius:8px;
    padding:14px 16px; overflow:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:12.5px; line-height:1.5; color:#1b2330; margin:0; }
  /* CSS-only sub-tabs (one panel per target file) */
  .autosec .subtabs > input[type=radio] { position:absolute; opacity:0; pointer-events:none; }
  .autosec .subnav { display:flex; flex-wrap:wrap; gap:4px; border-bottom:1px solid var(--line);
    margin:8px 0 18px; }
  .autosec .subnav label { display:inline-flex; align-items:center; gap:8px; padding:8px 14px;
    font-size:12.5px; cursor:pointer; border:1px solid transparent; border-bottom:none;
    border-radius:8px 8px 0 0; color:var(--muted); margin-bottom:-1px; }
  .autosec .subnav label:hover { background:var(--card); color:var(--text); }
  .autosec .subnav .dot { width:8px; height:8px; border-radius:50%; }
  .autosec .dot.b-exploited { background:var(--red); }
  .autosec .dot.b-vuln { background:var(--amber); }
  .autosec .dot.b-secure { background:var(--green); }
  .autosec .panel { display:none; }
</style>
"""


def _status(finding: dict) -> tuple[str, str]:
    """Return (css_class, label) for a finding's three-state status badge."""
    hyp = finding.get("hypothesis") or {}
    verdict = finding.get("verdict") or {}
    if not hyp.get("vulnerable"):
        return "b-secure", "SECURE"
    if verdict.get("triggered"):
        return "b-exploited", "EXPLOITED"
    return "b-vuln", "VULNERABLE"


def _render_report_html(findings: list[dict]) -> str:
    """Render the per-target findings into a styled HTML security report."""
    exploited = sum(1 for f in findings if (f.get("verdict") or {}).get("triggered"))
    vulnerable = sum(1 for f in findings if (f.get("hypothesis") or {}).get("vulnerable"))
    secure = len(findings) - vulnerable

    rows = []
    for f in sorted(findings, key=lambda x: x["target"]):
        hyp = f.get("hypothesis") or {}
        verdict = f.get("verdict") or {}
        cls, label = _status(f)
        is_vuln = bool(hyp.get("vulnerable"))
        vuln_class = hyp.get("vuln_class", "—") if is_vuln else "—"
        fn = hyp.get("function", "—") if is_vuln else "—"
        buf = hyp.get("buffer_size", "—") if is_vuln else "—"
        payload = (f.get("poc") or {}).get("payload_len", "—") if is_vuln else "—"
        exit_code = verdict.get("sandbox_exit_code", "—")
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(f['target']))}</code></td>"
            f'<td><span class="badge {cls}">{label}</span></td>'
            f"<td>{html.escape(str(vuln_class))}</td>"
            f"<td><code>{html.escape(str(fn))}</code></td>"
            f'<td class="num">{html.escape(str(buf))}</td>'
            f'<td class="num">{html.escape(str(payload))}</td>'
            f'<td class="num">{html.escape(str(exit_code))}</td>'
            f'<td class="reason">{html.escape(str(hyp.get("reasoning", "")))}</td>'
            "</tr>"
        )

    return f"""{_REPORT_CSS}
    <div class="autosec">
      <h2>AutoSec &middot; security findings report</h2>
      <p class="sub">{len(findings)} target(s) analyzed in parallel &middot; PoCs validated in isolated Daytona VMs.</p>
      <div class="cards">
        <div class="card"><div class="n">{len(findings)}</div><div class="l">Targets</div></div>
        <div class="card"><div class="n" style="color:#ff6b6b">{exploited}</div><div class="l">Exploited</div></div>
        <div class="card">
          <div class="n" style="color:#ffb454">{vulnerable - exploited}</div>
          <div class="l">Vuln, PoC failed</div>
        </div>
        <div class="card"><div class="n" style="color:#3fb950">{secure}</div><div class="l">Secure</div></div>
      </div>
      <table>
        <colgroup>
          <col class="c-target"><col class="c-status"><col class="c-class"><col class="c-fn">
          <col class="c-buf"><col class="c-payload"><col class="c-exit"><col class="c-reason">
        </colgroup>
        <thead>
          <tr>
            <th>Target</th><th>Status</th><th>Vuln class</th><th>Function</th>
            <th class="num">Buffer&nbsp;(B)</th><th class="num">Payload&nbsp;(B)</th>
            <th class="num">Exit</th><th>Reasoning</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """


def _target_detail_html(finding: dict, source: str) -> str:
    """Inner detail markup for one target: title, status, stats, reasoning, code."""
    hyp = finding.get("hypothesis") or {}
    verdict = finding.get("verdict") or {}
    cls, label = _status(finding)
    is_vuln = bool(hyp.get("vulnerable"))

    def cell(k: str, v: Any) -> str:
        return f'<div><div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div></div>'

    triggered = verdict.get("triggered")
    verdict_txt = "skipped (secure)" if verdict.get("skipped") else ("triggered" if triggered else "not triggered")
    stats = "".join(
        [
            cell("Vuln class", hyp.get("vuln_class", "—") if is_vuln else "—"),
            cell("Function", hyp.get("function", "—") if is_vuln else "—"),
            cell("Buffer (B)", hyp.get("buffer_size", "—") if is_vuln else "—"),
            cell("Payload (B)", (finding.get("poc") or {}).get("payload_len", "—") if is_vuln else "—"),
            cell("Daytona exit", verdict.get("sandbox_exit_code", "—")),
            cell("PoC", verdict_txt),
        ]
    )
    reasoning = html.escape(str(hyp.get("reasoning", "")) or "—")
    code = html.escape(source or "(source unavailable)")
    return f"""
      <h3 style="margin:0 0 4px"><code>{html.escape(str(finding["target"]))}</code>
          &nbsp;<span class="badge {cls}">{label}</span></h3>
      <p class="sub">Per-target detail &middot; PoCs validated in an isolated Daytona VM.</p>
      <div class="kv">{stats}</div>
      <div class="section-label">Reasoning</div>
      <div class="reason-block">{reasoning}</div>
      <div class="section-label">Source</div>
      <pre class="code">{code}</pre>
    """


def _render_targets_tab_html(findings: list[dict], sources: dict[str, str]) -> str:
    """Render ONE 'targets' tab whose HTML contains CSS-only sub-tabs per target file."""
    ordered = sorted(findings, key=lambda x: x["target"])

    radios, nav, panels, rules = [], [], [], []
    for i, f in enumerate(ordered):
        name = f["target"]
        cls, _ = _status(f)
        checked = " checked" if i == 0 else ""
        radios.append(f'<input type="radio" name="as-targets" id="as-t{i}"{checked}>')
        nav.append(f'<label for="as-t{i}"><span class="dot {cls}"></span><code>{html.escape(str(name))}</code></label>')
        panels.append(f'<div class="panel" id="as-p{i}">{_target_detail_html(f, sources.get(name, ""))}</div>')
        # Show the active label + its panel when this radio is checked.
        rules.append(
            f'.autosec #as-t{i}:checked ~ .subnav label[for="as-t{i}"]'
            "{background:#fff;color:var(--text);border-color:var(--line);font-weight:600;}"
            f".autosec #as-t{i}:checked ~ .panels #as-p{i}{{display:block;}}"
        )

    return f"""{_REPORT_CSS}
    <style>{"".join(rules)}</style>
    <div class="autosec">
      <h2>AutoSec &middot; target detail</h2>
      <p class="sub">{len(ordered)} target(s) &middot; select a file to see its status, reasoning, and source.</p>
      <div class="subtabs">
        {"".join(radios)}
        <div class="subnav">{"".join(nav)}</div>
        <div class="panels">{"".join(panels)}</div>
      </div>
    </div>
    """


@env.task(retries=1)
async def random_error() -> str:
    if _attempt() == 0:
        raise Exception("Random error")
    return "Passed!"


@env.task(report=True)
async def run_autosec_agent() -> dict:
    targets = _load_targets()
    if not targets:
        raise FileNotFoundError(f"no targets found under {TARGETS_DIR}")

    # Fan out: analyze every target concurrently. Each `analyze_target` is its own
    # Flyte action, so the targets are researched in parallel; per-target stages
    # still run sequentially with their own retries/timeouts/resource overrides.
    findings = list(await asyncio.gather(*(analyze_target(name, src) for name, src in targets.items())))

    # Main tab: the aggregated summary table (unchanged).
    await flyte.report.replace.aio(_render_report_html(findings))
    # A single "targets" tab whose HTML carries CSS-only sub-tabs, one per file.
    flyte.report.get_tab("targets").replace(_render_targets_tab_html(findings, targets))
    await flyte.report.flush.aio()

    # Sprinkle in a random error on the first attempt of this task.
    await random_error()

    return {
        "targets_analyzed": len(findings),
        "triggered": sum(1 for f in findings if f["verdict"].get("triggered")),
        "findings": findings,
    }


def cli() -> None:
    """Console-script entry point (see pyproject.toml ``[project.scripts]``)."""
    flyte.init_from_config(root_dir=HERE)
    run = flyte.with_runcontext(env_vars={"AUTOSEC_FORCE_ALL": "1"}).run(run_autosec_agent)
    print(f"Run URL: {run.url}")


if __name__ == "__main__":
    cli()
