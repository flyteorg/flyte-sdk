"""Turn a .nsys-rep into a Flyte report tab.

Runs a handful of `nsys stats` reports over the trace, parses their CSV, and renders a
self-contained HTML deck: a row of summary tiles, a top-kernels bar chart, an NVTX range
breakdown, and collapsible full tables. Everything is inline HTML with neutral styling so it
reads on any console background and needs no runtime plotting dependency.

The report is best-effort. Any single `nsys stats` report can be unsupported on a given nsys
version or simply empty for a given trace; those sections are skipped rather than failed.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import re
from typing import Optional

from . import _control

logger = logging.getLogger(__name__)

# The stats reports we render, in display order. Names track nsys 2023+; unknown ones drop out.
DEFAULT_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_api_sum",
    "nvtx_pushpop_sum",
)

# A single-accent, theme-neutral palette: text rides on the console's own color via currentColor +
# opacity, surfaces/borders are grey-alpha (legible on light or dark), and the accent lands only on
# the magnitude bars — never on text.
_ACCENT = "#7c4dff"
_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
_MUTED = "opacity:.58"  # secondary text (labels, meta)
_FAINT = "opacity:.42"  # tertiary text (rank numbers, row counts)
_LINE = "rgba(128,128,128,.22)"  # hairline borders
_SOFT = "rgba(128,128,128,.09)"  # subtle surfaces: cards, table header, summary bar
_TRACK = "rgba(128,128,128,.15)"  # the empty portion of a bar
_ZEBRA = "rgba(128,128,128,.05)"  # alternating table rows
_NUM = "font-variant-numeric:tabular-nums"  # line up digits in columns and stat values
_ELLIP = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
_CODE = (
    f"font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;"
    f"background:{_SOFT};padding:1px 5px;border-radius:4px"
)


def _strip_preamble(text: str) -> str:
    """Drop nsys's non-CSV status lines that precede the table.

    `nsys stats --format csv` prints progress to stdout ahead of the CSV: "Generating SQLite file
    ...", "Processing [...] with [...]", a blank line, and a " ** Section (id):" banner. None of
    those contain a comma, whereas every nsys CSV header does — so the table starts at the first line
    with a comma. Returns "" when there is no CSV (an empty or "SKIPPED" report).
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "," in line:
            return "\n".join(lines[i:])
    return ""


def _rows(csv_text: Optional[str]) -> list[dict[str, str]]:
    if not csv_text:
        return []
    csv_text = _strip_preamble(csv_text)
    if not csv_text:
        return []
    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
    except Exception:  # pragma: no cover - malformed CSV
        logger.debug("could not parse nsys stats CSV", exc_info=True)
        return []
    # csv.DictReader files overflow fields (a data row with more columns than the header — e.g. an
    # unquoted comma in a demangled kernel name) under a None restkey. Drop it so downstream lookups
    # that iterate row keys never hit a non-string key.
    return [{k: v for k, v in row.items() if k is not None} for row in rows]


def _col(row: dict[str, str], *needles: str) -> str:
    """Fetch a value by fuzzy (case-insensitive substring) column match; column names drift by version."""
    for key, value in row.items():
        if not isinstance(key, str):
            continue  # defensive: a csv.DictReader overflow key is None, not matchable
        low = key.lower()
        if all(n.lower() in low for n in needles):
            return value
    return ""


def _unit(row: dict[str, str], *needles: str) -> str:
    """The parenthetical unit of the first column matching all needles, e.g. 'Total (MB)' -> 'MB'.

    Lets the report label a value with the unit nsys actually reported instead of assuming one.
    """
    for key in row:
        if not isinstance(key, str):
            continue
        low = key.lower()
        if all(n.lower() in low for n in needles):
            m = re.search(r"\(([^)]+)\)", key)
            return m.group(1) if m else ""
    return ""


_TEMPLATE_HEAD = re.compile(r"<([A-Za-z0-9_]+)")


def _short_name(name: str) -> str:
    """Condense a demangled CUDA kernel signature (or NVTX range) to a readable identifier.

    nsys reports full C++ signatures — often hundreds of chars of template args. Show just the kernel
    name: the identifier before the template/parameter lists, or, for cutlass (which hides the useful
    variant inside the angle brackets, e.g. cutlass_80_tensorop_s1688gemm_...), the first name within
    them. NVTX ranges lose their leading ':'. Returns the input unchanged if it isn't a signature.
    """
    s = name.strip().lstrip(":").strip()
    if not s:
        return name.strip()
    if "cutlass" in s:
        m = _TEMPLATE_HEAD.search(s)
        if m:
            return m.group(1)
    # Anonymous-namespace markers carry angle brackets/parens of their own; drop them first so the
    # split below lands on the real template/parameter list rather than inside "<unnamed>".
    s = s.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    head = re.split(r"[<(]", s, maxsplit=1)[0].strip()
    tokens = head.split()  # drop a leading return type, e.g. "void "
    head = tokens[-1] if tokens else head
    parts = [p for p in head.split("::") if p]
    return parts[-1] if parts else s


# Generic wrappers whose real operation is nested one level down in their template args.
_HINT_DESCEND = {"ReduceOp", "func_wrapper_t"}
_HINT_SKIP = {"int", "float", "double", "bool", "unsigned", "long", "char", "void", "size_t"}


def _template_args(name: str) -> list[str]:
    """Top-level comma-separated arguments of the first ``<...>`` group, respecting < ( [ nesting."""
    start = name.find("<")
    if start < 0:
        return []
    depth = 0
    args: list[str] = []
    cur: list[str] = []
    for ch in name[start:]:
        if ch in "<([":
            depth += 1
            if depth == 1:
                continue  # skip the outermost opening bracket
        elif ch in ">)]":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and ch == ",":
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _functor_hint(name: str, _depth: int = 0) -> str:
    """The operation a generic dispatch kernel runs, pulled from its template args (best effort).

    Sibling kernels like vectorized_elementwise_kernel share a base name and differ only in a functor
    template argument; surface it so they can be told apart. Skips the leading (int) tile params and
    bare types, and descends through a couple of generic wrappers (ReduceOp, func_wrapper_t) whose
    real op is nested a level down. Returns "" when nothing informative is found.
    """
    cleaned = name.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    for arg in _template_args(cleaned):
        short = _short_name(arg)
        if not short or short[0] in "(0123456789" or short in _HINT_SKIP:
            continue
        if short in _HINT_DESCEND and _depth < 3:
            deeper = _functor_hint(arg, _depth + 1)
            if deeper:
                return deeper
        return short
    return ""


def _num(value: str) -> float:
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _fmt_ns(ns: float) -> str:
    if ns <= 0:
        return "0"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.2f} {unit}"
    return f"{ns:.0f} ns"


def _fmt_int(n: float) -> str:
    return f"{int(n):,}"


def _esc(v: object) -> str:
    return html.escape(str(v))


def _tiles(tiles: list[tuple[str, str]]) -> str:
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


def _bars(rows: list[dict], label_cols: tuple[str, ...], title: str, n: int = 10) -> str:
    """Horizontal bar chart of the top-n rows by total time, as width-scaled divs."""
    ranked = sorted(rows, key=lambda r: _num(_col(r, "Total Time")), reverse=True)[:n]
    if not ranked:
        return ""
    top = _num(_col(ranked[0], "Total Time")) or 1.0
    raw = [_col(r, *label_cols) for r in ranked]
    labels = [_short_name(x) or "(unnamed)" for x in raw]
    # A generic kernel name (vectorized_elementwise_kernel, reduce_kernel) can appear several times
    # with different functors; append a hint to the collisions so the bars aren't a stack of
    # identical labels.
    counts: dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    for idx, lbl in enumerate(labels):
        if counts[lbl] > 1:
            hint = _functor_hint(raw[idx])
            if hint:
                labels[idx] = f"{lbl} · {hint}"
    bars = []
    for i, (r, label) in enumerate(zip(ranked, labels), 1):
        total = _num(_col(r, "Total Time"))
        pct = _col(r, "Time", "%")  # the "Time (%)" column; needs both needles so "Total Time (ns)" won't match
        width = max(1.0, 100.0 * total / top)
        meta = _fmt_ns(total) + (f" · {_esc(pct)}%" if pct else "")
        bars.append(
            f'<div style="margin:7px 0">'
            f'<div style="display:flex;justify-content:space-between;gap:12px;font-size:12px;margin-bottom:4px">'
            f'<span style="{_ELLIP};max-width:72%"><span style="{_FAINT};{_NUM}">{i}.</span> {_esc(label)}</span>'
            f'<span style="{_MUTED};white-space:nowrap;{_NUM}">{meta}</span></div>'
            f'<div style="height:8px;border-radius:5px;background:{_TRACK};overflow:hidden">'
            f'<div style="height:100%;border-radius:5px;background:{_ACCENT};width:{width:.1f}%;min-width:3px"></div>'
            "</div></div>"
        )
    return (
        f'<h3 style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;'
        f'{_MUTED};margin:20px 0 9px">{_esc(title)}</h3>' + "".join(bars)
    )


def _table(rows: list[dict], title: str) -> str:
    """Full detail table, collapsed by default. Expanding it shows every row; a tall table scrolls
    within its own box (max-height) so it never runs the whole report off the page."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(
        f'<th style="text-align:left;padding:7px 12px;font-size:10.5px;text-transform:uppercase;'
        f'letter-spacing:.05em;font-weight:600;{_MUTED};border-bottom:1px solid {_LINE}">{_esc(c)}</th>'
        for c in cols
    )
    body = []
    for ri, r in enumerate(rows):
        zebra = f";background:{_ZEBRA}" if ri % 2 else ""
        tds = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(c, str) and c.strip().lower() in ("name", "range") and isinstance(v, str):
                # Condense the signature for reading; the full name stays available on hover.
                tds.append(f'<td style="padding:5px 12px{zebra}" title="{_esc(v)}">{_esc(_short_name(v))}</td>')
            else:
                tds.append(f'<td style="padding:5px 12px;white-space:nowrap;{_NUM}{zebra}">{_esc(v)}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f'<details style="margin:10px 0;border:1px solid {_LINE};border-radius:10px;overflow:hidden">'
        f'<summary style="cursor:pointer;padding:9px 13px;font-size:12px;font-weight:600;background:{_SOFT}">'
        f'{_esc(title)} <span style="{_FAINT};font-weight:400">({len(rows)} rows)</span></summary>'
        f'<div style="overflow:auto;max-height:420px">'
        f'<table style="border-collapse:collapse;font-size:12px;width:100%">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></details>"
    )


def _summary_tiles(parsed: dict[str, list[dict]]) -> tuple[str, dict]:
    kern = parsed.get("cuda_gpu_kern_sum", [])
    mem = parsed.get("cuda_gpu_mem_size_sum", [])
    nvtx = parsed.get("nvtx_pushpop_sum", [])

    gpu_time = sum(_num(_col(r, "Total Time")) for r in kern)
    kern_count = sum(_num(_col(r, "Instances")) for r in kern)
    top = max(kern, key=lambda r: _num(_col(r, "Total Time")), default=None)
    top_name = (_col(top, "Name") if top else "") or "—"
    top_pct = (_col(top, "Time", "%") if top else "") or "0"

    def _mem(*aliases: str) -> str:
        # Operation labels vary by nsys version ("[CUDA memcpy HtoD]" vs "Host-to-Device"). The size
        # unit is read from the column header ("Total (MB)"), not assumed.
        for r in mem:
            op = _col(r, "Operation").lower()
            if any(a in op for a in aliases):
                return f"{_num(_col(r, 'Total')):.1f} {_unit(r, 'Total') or 'MB'}"
        return "—"

    tiles = [
        ("GPU kernel time", _fmt_ns(gpu_time)),
        ("Kernel launches", _fmt_int(kern_count)),
        ("Distinct kernels", _fmt_int(len(kern))),
        ("HtoD copied", _mem("htod", "host-to-device")),
        ("DtoH copied", _mem("dtoh", "device-to-host")),
        ("NVTX ranges", _fmt_int(len(nvtx))),
    ]
    summary = {
        "gpu_kernel_time_ns": gpu_time,
        "kernel_launches": int(kern_count),
        "distinct_kernels": len(kern),
        "top_kernel": top_name,
        "top_kernel_pct": top_pct,
        "nvtx_ranges": len(nvtx),
    }
    return _tiles(tiles), summary


def _render_body(parsed: dict[str, list[dict]], heading: str, trace_url: Optional[str] = None) -> tuple[str, dict]:
    """Assemble the report HTML body from parsed stats. Pure (no I/O); returns (html, summary).

    trace_url, when given, is a durable storage path to the .nsys-rep, linked from the deck. It is
    set for clustered/jobset workers, which cannot attach the trace as a File output (no controller).
    """
    tiles_html, summary = _summary_tiles(parsed)

    ctx = (
        f"{summary['distinct_kernels']} kernels · {summary['kernel_launches']:,} launches · "
        f"{summary['nvtx_ranges']} NVTX ranges"
    )
    if summary["top_kernel"] not in ("—", ""):
        ctx += f" · busiest: {_esc(_short_name(summary['top_kernel']))} ({_esc(summary['top_kernel_pct'])}%)"

    code = f' style="{_CODE}"'
    if trace_url:
        note = (
            f'<p style="font-size:12px;{_MUTED};margin:18px 2px 4px;line-height:1.55">'
            f"The <code{code}>.nsys-rep</code> trace is saved to <code{code}>{_esc(trace_url)}</code> — "
            f"download it (e.g. <code{code}>flyte storage cp</code>) and open it in the Nsight Systems GUI "
            f"(<code{code}>nsys-ui</code>) for the full timeline.</p>"
        )
    else:
        note = (
            f'<p style="font-size:12px;{_MUTED};margin:18px 2px 4px;line-height:1.55">'
            f"Download the .nsys-rep trace output and open it in the Nsight Systems GUI "
            f"(<code{code}>nsys-ui</code>) for the full timeline.</p>"
        )

    parts = [
        f'<h2 style="font-size:16px;font-weight:650;letter-spacing:-.01em;margin:2px 0 3px">{_esc(heading)}</h2>',
        f'<div style="font-size:12px;{_MUTED};margin:0 0 14px">{ctx}</div>',
        tiles_html,
        _bars(parsed.get("cuda_gpu_kern_sum", []), ("Name",), "Top CUDA kernels by GPU time"),
        _bars(parsed.get("nvtx_pushpop_sum", []), ("Range",), "NVTX ranges by time"),
        note,
        _table(parsed.get("cuda_gpu_kern_sum", []), "CUDA kernel summary"),
        _table(parsed.get("cuda_gpu_mem_time_sum", []), "GPU memory-op summary"),
        _table(parsed.get("cuda_api_sum", []), "CUDA API summary"),
        _table(parsed.get("nvtx_pushpop_sum", []), "NVTX range summary"),
    ]
    body_html = "\n".join(p for p in parts if p)
    return f'<div style="font-family:{_FONT};font-size:13px;line-height:1.5">{body_html}</div>', summary


async def render(
    report_path: str,
    *,
    reports=DEFAULT_REPORTS,
    tab: str = "GPU Profile",
    title: Optional[str] = None,
    trace_url: Optional[str] = None,
) -> dict:
    """Render metrics from a .nsys-rep into a Flyte report tab. Returns a small summary dict."""
    import flyte.report

    parsed: dict[str, list[dict]] = {}
    for name in reports:
        parsed[name] = _rows(await _control.run_stats(report_path, name))

    html_body, summary = _render_body(parsed, title or "Nsight Systems — GPU profile", trace_url)
    try:
        flyte.report.get_tab(tab).log(html_body)
        await flyte.report.flush.aio()
    except Exception:  # pragma: no cover - no active report (task without report=True)
        logger.debug("no active Flyte report to render nsys metrics into", exc_info=True)

    return summary


def render_sync(
    report_path: str,
    *,
    reports=DEFAULT_REPORTS,
    tab: str = "GPU Profile",
    title: Optional[str] = None,
    trace_url: Optional[str] = None,
) -> dict:
    """Blocking twin of render, for a `with nsys.range(...)` block in a non-async task body.

    `flyte.report.get_tab().log()` is already sync and `flyte.report.flush()` is the sync form of the
    same flush the async path awaits, so this renders the identical deck without an event loop.
    """
    import flyte.report

    parsed: dict[str, list[dict]] = {}
    for name in reports:
        parsed[name] = _rows(_control.run_stats_sync(report_path, name))

    html_body, summary = _render_body(parsed, title or "Nsight Systems — GPU profile", trace_url)
    try:
        flyte.report.get_tab(tab).log(html_body)
        flyte.report.flush()
    except Exception:  # pragma: no cover - no active report (task without report=True)
        logger.debug("no active Flyte report to render nsys metrics into", exc_info=True)

    return summary
