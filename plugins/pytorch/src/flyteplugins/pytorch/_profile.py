"""torch.profiler → Flyte report.

Wrap a region of a task body in `torch_profile()` and the collected profile is rendered into a
Flyte report tab: summary tiles, top-ops tables, and an interactive Perfetto timeline (the
chrome trace is gzipped, base64-embedded in the report, and pushed into an `ui.perfetto.dev`
iframe via its postMessage deep-link API — so the full flamegraph is explorable in the Flyte UI).

    from flyteplugins.pytorch import torch_profile

    @env.task(report=True)                     # required: the report tab only shows with report=True
    def train():
        with torch_profile(schedule=torch.profiler.schedule(wait=1, warmup=2, active=4, repeat=1)) as prof:
            for step in range(8):
                with torch.profiler.record_function("matmul_add"):
                    ...
                prof.step()

    @env.task(report=True)
    async def train_async():                   # async task -> `async with`
        async with torch_profile() as prof:
            ...

All keyword arguments besides `tab` and `max_embed_mb` are forwarded verbatim to
`torch.profiler.profile` (activities, schedule, record_shapes, profile_memory, with_stack, ...).
Rendering is best-effort: a profiling/rendering failure never fails the task, and exceptions
raised by the body are never suppressed. Targets single-process tasks; for distributed tasks,
guard the block on rank 0 yourself.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import logging
import os
import tempfile
import uuid
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Same theme-neutral styling approach as the nsight plugin deck: text rides on the console's own
# color via currentColor + opacity, surfaces are grey-alpha, the accent lands only on the bars.
_ACCENT = "#7c4dff"
_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
_MUTED = "opacity:.58"
_FAINT = "opacity:.42"
_LINE = "rgba(128,128,128,.22)"
_SOFT = "rgba(128,128,128,.09)"
_TRACK = "rgba(128,128,128,.15)"
_ZEBRA = "rgba(128,128,128,.05)"
_NUM = "font-variant-numeric:tabular-nums"
_ELLIP = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
_CODE = (
    f"font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;"
    f"background:{_SOFT};padding:1px 5px;border-radius:4px"
)


def _esc(v: object) -> str:
    return html.escape(str(v))


def _fmt_us(us: float) -> str:
    """Human-format a microsecond duration (torch profiler times are µs)."""
    if us <= 0:
        return "0"
    for unit, scale in (("s", 1e6), ("ms", 1e3)):
        if us >= scale:
            return f"{us / scale:.2f} {unit}"
    return f"{us:.0f} µs"


def _fmt_bytes(n: float) -> str:
    n = abs(n)
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n:.0f} B"


def _attr(evt: Any, *names: str) -> float:
    """First present-and-truthy numeric attribute; names drift across torch versions
    (self_cuda_time_total -> self_device_time_total in torch 2.x)."""
    for name in names:
        v = getattr(evt, name, None)
        if v:
            return float(v)
    return 0.0


def _summary_rows(key_averages: Any) -> List[dict]:
    """Normalize prof.key_averages() into plain dicts (all times µs, memory bytes)."""
    rows = []
    for evt in key_averages:
        rows.append(
            {
                "name": str(getattr(evt, "key", evt)),
                "count": int(getattr(evt, "count", 0)),
                "self_cpu_us": _attr(evt, "self_cpu_time_total"),
                "cpu_us": _attr(evt, "cpu_time_total"),
                "self_device_us": _attr(evt, "self_device_time_total", "self_cuda_time_total"),
                "device_us": _attr(evt, "device_time_total", "cuda_time_total"),
                "self_device_mem": _attr(evt, "self_device_memory_usage", "self_cuda_memory_usage"),
                "cpu_mem": _attr(evt, "cpu_memory_usage"),
            }
        )
    return rows


def _tiles(tiles: List[Tuple[str, str]]) -> str:
    cells = "".join(
        f'<div style="flex:1 1 140px;min-width:140px;border:1px solid {_LINE};background:{_SOFT};'
        f'border-radius:10px;padding:11px 14px">'
        f'<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:600;'
        f'{_MUTED}">{_esc(label)}</div>'
        f'<div style="font-size:22px;font-weight:640;margin-top:5px;letter-spacing:-.02em;'
        f'{_NUM}">{_esc(value)}</div></div>'
        for label, value in tiles
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin:6px 0">{cells}</div>'


def _bars(rows: List[dict], key: str, title: str, n: int = 10) -> str:
    ranked = sorted((r for r in rows if r[key] > 0), key=lambda r: r[key], reverse=True)[:n]
    if not ranked:
        return ""
    top = ranked[0][key]
    bars = []
    for i, r in enumerate(ranked, 1):
        width = max(1.0, 100.0 * r[key] / top)
        meta = f"{_fmt_us(r[key])} · {r['count']} calls"
        bars.append(
            f'<div style="margin:7px 0">'
            f'<div style="display:flex;justify-content:space-between;gap:12px;font-size:12px;margin-bottom:4px">'
            f'<span style="{_ELLIP};max-width:72%"><span style="{_FAINT};{_NUM}">{i}.</span> {_esc(r["name"])}</span>'
            f'<span style="{_MUTED};white-space:nowrap;{_NUM}">{meta}</span></div>'
            f'<div style="height:8px;border-radius:5px;background:{_TRACK};overflow:hidden">'
            f'<div style="height:100%;border-radius:5px;background:{_ACCENT};width:{width:.1f}%;min-width:3px"></div>'
            "</div></div>"
        )
    return (
        f'<h3 style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        f'{_MUTED};margin:20px 0 9px">{_esc(title)}</h3>' + "".join(bars)
    )


def _table(rows: List[dict], title: str) -> str:
    if not rows:
        return ""
    cols = [
        ("name", "Name"),
        ("count", "Count"),
        ("self_cpu_us", "Self CPU"),
        ("cpu_us", "CPU total"),
        ("self_device_us", "Self device"),
        ("device_us", "Device total"),
        ("self_device_mem", "Device mem"),
        ("cpu_mem", "CPU mem"),
    ]
    head = "".join(
        f'<th style="text-align:left;padding:7px 12px;font-size:10.5px;text-transform:uppercase;'
        f'letter-spacing:.05em;font-weight:600;{_MUTED};border-bottom:1px solid {_LINE}">{_esc(label)}</th>'
        for _, label in cols
    )
    ranked = sorted(rows, key=lambda r: max(r["self_device_us"], r["self_cpu_us"]), reverse=True)
    body = []
    for ri, r in enumerate(ranked):
        zebra = f";background:{_ZEBRA}" if ri % 2 else ""
        tds = [f'<td style="padding:5px 12px{zebra}" title="{_esc(r["name"])}">{_esc(r["name"])}</td>']
        for k, _ in cols[1:]:
            v = r[k]
            if k == "count":
                text = f"{v:,}"
            elif k.endswith("_mem"):
                text = _fmt_bytes(v) if v else "—"
            else:
                text = _fmt_us(v)
            tds.append(f'<td style="padding:5px 12px;white-space:nowrap;{_NUM}{zebra}">{text}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f'<details style="margin:10px 0;border:1px solid {_LINE};border-radius:10px;overflow:hidden">'
        f'<summary style="cursor:pointer;padding:9px 13px;font-size:12px;font-weight:600;background:{_SOFT}">'
        f'{_esc(title)} <span style="{_FAINT};font-weight:400">({len(rows)} rows)</span></summary>'
        f'<div style="overflow:auto;max-height:420px">'
        f'<table style="border-collapse:collapse;font-size:12px;width:100%">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></details>"
    )


def _timeline_section(b64: str, title: str, sfx: str) -> str:
    """The interactive part: a lazy Perfetto iframe fed the embedded gzipped chrome trace.

    Lazy (button-gated) on purpose: report tabs are display:none until clicked, and Perfetto is a
    heavy app — loading it into a hidden 0-height tab both wastes bandwidth and mis-sizes its
    canvas. The handshake is Perfetto's documented deep-link API: poll PING until the UI answers
    PONG, then post {perfetto: {buffer, title, fileName}}. Perfetto reads gzipped traces natively.
    """
    btn = (
        f"border:1px solid {_LINE};background:{_SOFT};color:inherit;border-radius:8px;"
        f"padding:7px 14px;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit"
    )
    return f"""
<h3 style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;{_MUTED};margin:22px 0 9px">
Timeline</h3>
<div style="display:flex;gap:10px;align-items:center">
  <button id="pt-open-{sfx}" style="{btn}" onclick="ptOpen_{sfx}(this)">▶ Open timeline (Perfetto)</button>
  <button style="{btn}" onclick="ptDownload_{sfx}()">⬇ Download trace (.json.gz)</button>
  <span style="font-size:11.5px;{_MUTED}">timeline loads ui.perfetto.dev; the download opens there or in chrome://tracing</span>
</div>
<iframe id="pt-frame-{sfx}"
 style="display:none;width:100%;height:720px;border:1px solid {_LINE};border-radius:10px;margin-top:10px"></iframe>
<script>
var PT_B64_{sfx} = "{b64}";
function ptBuf_{sfx}() {{
  var bin = atob(PT_B64_{sfx}), buf = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}}
function ptDownload_{sfx}() {{
  var a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([ptBuf_{sfx}()], {{type: "application/gzip"}}));
  a.download = "torch_trace.json.gz"; a.click(); URL.revokeObjectURL(a.href);
}}
function ptOpen_{sfx}(btn) {{
  var ORIGIN = "https://ui.perfetto.dev";
  var frame = document.getElementById("pt-frame-{sfx}");
  btn.style.display = "none"; frame.style.display = "block"; frame.src = ORIGIN;
  var buffer = ptBuf_{sfx}();
  var timer = setInterval(function() {{ frame.contentWindow.postMessage("PING", ORIGIN); }}, 250);
  window.addEventListener("message", function onMsg(evt) {{
    if (evt.origin !== ORIGIN || evt.data !== "PONG") return;
    clearInterval(timer); window.removeEventListener("message", onMsg);
    frame.contentWindow.postMessage(
      {{perfetto: {{buffer: buffer, title: {json.dumps(title)}, fileName: "torch_trace.json.gz"}}}}, ORIGIN);
  }}, false);
}}
</script>
"""


def _render_html(
    title: str,
    rows: List[dict],
    b64: Optional[str],
    trace_bytes: int,
    trace_note: Optional[str] = None,
) -> str:
    """Assemble the tab HTML. Pure — no I/O, no task context."""
    sfx = uuid.uuid4().hex[:8]
    total_device = sum(r["self_device_us"] for r in rows)
    total_cpu = sum(r["self_cpu_us"] for r in rows)
    calls = sum(r["count"] for r in rows)
    peak_mem = max((r["self_device_mem"] for r in rows), default=0)

    tiles = [
        ("Self device time", _fmt_us(total_device)),
        ("Self CPU time", _fmt_us(total_cpu)),
        ("Distinct ops", f"{len(rows):,}"),
        ("Op calls", f"{calls:,}"),
    ]
    if peak_mem:
        tiles.append(("Top op device mem", _fmt_bytes(peak_mem)))

    ctx = f"{len(rows)} ops · {calls:,} calls"
    if trace_bytes:
        ctx += f" · trace {_fmt_bytes(trace_bytes)}"
        ctx += f" ({_fmt_bytes(len(b64) * 3 / 4)} gz)" if b64 else " (not embedded)"

    parts = [
        f'<h2 style="font-size:16px;font-weight:650;letter-spacing:-.01em;margin:2px 0 3px">{_esc(title)}</h2>',
        f'<div style="font-size:12px;{_MUTED};margin:0 0 14px">{ctx}</div>',
        _tiles(tiles),
        _bars(rows, "self_device_us", "Top ops by self device (GPU) time"),
        _bars(rows, "self_cpu_us", "Top ops by self CPU time"),
        _timeline_section(b64, title, sfx) if b64 else "",
        f'<p style="font-size:12px;{_MUTED};margin:18px 2px 4px;line-height:1.55">{trace_note}</p>'
        if trace_note
        else "",
        _table(rows, "All ops"),
    ]
    body = "\n".join(p for p in parts if p)
    return f'<div style="font-family:{_FONT};font-size:13px;line-height:1.5">{body}</div>'


class _TorchProfile:
    """Context manager usable with `with` (sync task body) or `async with` (async task body).

    Both protocols do the same thing — run torch.profiler over the block, then render the result
    into a Flyte report tab — and differ only in which flush they call (sync vs awaited).
    """

    def __init__(self, tab: str, max_embed_mb: float, profile_kwargs: dict):
        self._tab = tab
        self._max_embed_mb = max_embed_mb
        self._profile_kwargs = profile_kwargs
        self._prof: Any = None

    def _start(self) -> Any:
        import torch.profiler

        self._prof = torch.profiler.profile(**self._profile_kwargs)
        return self._prof.__enter__()

    # -- sync protocol --------------------------------------------------------------------------
    def __enter__(self) -> Any:
        return self._start()

    def __exit__(self, *exc: object) -> bool:
        self._prof.__exit__(*exc)
        try:
            html_out = self._build_html()
            if html_out is not None:
                import flyte.report

                flyte.report.get_tab(self._tab).log(html_out)
                flyte.report.flush()
        except Exception:
            logger.warning("torch_profile: failed to render profile report", exc_info=True)
        return False

    # -- async protocol -------------------------------------------------------------------------
    async def __aenter__(self) -> Any:
        return self._start()

    async def __aexit__(self, *exc: object) -> bool:
        self._prof.__exit__(*exc)
        try:
            html_out = self._build_html()
            if html_out is not None:
                import flyte.report

                flyte.report.get_tab(self._tab).log(html_out)
                await flyte.report.flush.aio()
        except Exception:
            logger.warning("torch_profile: failed to render profile report", exc_info=True)
        return False

    # -- shared finalize ------------------------------------------------------------------------
    def _build_html(self) -> Optional[str]:
        from flyte._context import internal_ctx

        if internal_ctx().get_report() is None:
            logger.warning(
                "torch_profile: no active Flyte report — declare the task with @env.task(report=True); "
                "profile not rendered"
            )
            return None

        rows = _summary_rows(self._prof.key_averages())

        b64: Optional[str] = None
        trace_note: Optional[str] = None
        trace_bytes = 0
        trace_path = os.path.join(tempfile.mkdtemp(prefix="torch_profile_"), "trace.json")
        try:
            # Raises if a schedule never reached an active window; summary still renders.
            self._prof.export_chrome_trace(trace_path)
            trace_bytes = os.path.getsize(trace_path)
            with open(trace_path, "rb") as f:
                gz = gzip.compress(f.read(), mtime=0)
            if len(gz) <= self._max_embed_mb * 2**20:
                b64 = base64.b64encode(gz).decode("ascii")
            else:
                trace_note = self._persist_trace(trace_path, len(gz))
        except Exception:
            logger.warning("torch_profile: could not export chrome trace; rendering summary only", exc_info=True)

        title = "PyTorch profiler — GPU profile"
        return _render_html(title, rows, b64, trace_bytes, trace_note)

    def _persist_trace(self, trace_path: str, gz_size: int) -> str:
        """Trace too large to inline: upload it to the run's raw-data path and link it instead."""
        code = f' style="{_CODE}"'
        note = f"Trace is {gz_size / 2**20:.0f} MB gzipped — too large to embed in the report."
        try:
            from flyte.io import File

            # from_local_sync dispatches to the syncify background loop, so it is safe from both
            # the sync-executor thread and (briefly blocking) the async exit path.
            f = File.from_local_sync(trace_path)
            note += (
                f" It is saved to <code{code}>{_esc(f.path)}</code> — download it "
                f"(e.g. <code{code}>flyte storage cp</code>) and open it at "
                f"<code{code}>ui.perfetto.dev</code>."
            )
        except Exception:
            logger.warning("torch_profile: could not upload oversized trace", exc_info=True)
            note += " Upload failed; re-run with with_stack/record_shapes off for a smaller trace."
        return note


def torch_profile(tab: str = "Torch Profile", *, max_embed_mb: float = 50.0, **profile_kwargs) -> _TorchProfile:
    """Profile a region of a Flyte task with torch.profiler and render the result to the report.

    Use as `with torch_profile(...) as prof:` in a sync task body or `async with` in an async one.
    Yields the underlying `torch.profiler.profile` object (call `prof.step()` when using a
    schedule). The task must be declared with `@env.task(report=True)`.

    Args:
        tab: Report tab name to render into.
        max_embed_mb: Traces whose gzipped size exceeds this are not embedded in the report; they
            are uploaded to blob storage and linked instead (summary tables still render).
        **profile_kwargs: Forwarded verbatim to `torch.profiler.profile` — activities, schedule,
            record_shapes, profile_memory, with_stack, etc.
    """
    return _TorchProfile(tab, max_embed_mb, profile_kwargs)
