"""Aggregation + HTML report rendering for the TypeSafe benchmark.

The matrix has three axes — **task type** x **condition** (with/without System 1)
x **System 2 provider** — and every cell is run ``repeats`` times, so the
aggregation is over a distribution rather than a single sample.  On top of the
usual means the report computes two things repeats make possible:

* **spread** — latency and quality sigma, so "faster" is a claim with an error bar;
* **stability** — how often identical repeats of the same case produce the same
  typed decision.  Free-text classification drifts between runs; a System 1
  decision should not.

Everything renders into a **single** report tab, as one page with an anchor nav:
hero + KPI cards, the full matrix, per-task plots, with-vs-without, stability,
the System 1 vs System 2 budget, quality, and one section per task type with
per-case detail.

Every table emits exactly one cell per header column (no ragged rows), and a cell
with zero successful runs renders a full-width ``BLOCKED`` note with the first
error instead of ``nan`` / ``0%``.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from _config import BENCHMARK_CONDITIONS, SYSTEM2_PROVIDERS
from _pricing import JEV, SELF_HOSTED, system2_price, unit_costs
from tasks import get_task

import flyte.report

# Short labels so the console tab bar doesn't clip/wrap.
_SHORT = {"qwen": "Qwen", "sonnet": "Sonnet", "opus": "Opus"}
_TASK_SHORT = {"support": "Support", "code_review": "Code review", "contract": "Contract"}
_COLOR = {"with": "#818cf8", "without": "#64748b"}  # indigo = with Jev, slate = without

# --------------------------------------------------------------------------- #
# Shared stylesheet (injected at the top of every tab)                        #
# --------------------------------------------------------------------------- #
_CSS = """
<style>
  /* The report is injected into the host template's body (a white page with its
     own tab bar). Paint the page and the chrome dark too, or the report floats
     on white with bright margins and a light tab strip above it. */
  html,body{background:#0f1117 !important;color:#dbe1ea;margin:0;}
  #flyte-frame-nav{background:#0f1117;border-bottom:1px solid #232733;}
  #flyte-frame-tabs li{color:#8b93a3 !important;}
  #flyte-frame-tabs li.active{color:#eef2ff !important;border-bottom-color:#818cf8 !important;}
  #flyte-frame-container{background:#0f1117;}
  #flyte-frame-container > div.active{padding:1rem 1.25rem !important;}
  body *::selection{background:#3730a3;color:#fff;}
  .typesafe-report{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
    background:#0f1117;color:#dbe1ea;max-width:1180px;margin:0 auto;padding:8px 16px 40px;line-height:1.45;
    min-height:100vh;}
  .typesafe-report h1{font-size:24px;margin:0 0 4px;color:#f1f5f9;}
  .typesafe-report h2{font-size:18px;margin:22px 0 6px;color:#f1f5f9;}
  .typesafe-report h3{font-size:15px;margin:18px 0 4px;color:#cbd5e1;}
  .typesafe-report p{margin:8px 0;color:#dbe1ea;}
  .hero{background:linear-gradient(135deg,#4338ca 0%,#6d28d9 100%);color:#fff;border-radius:14px;
    padding:20px 26px;margin:12px 0 18px;border:1px solid rgba(255,255,255,.08);}
  .hero h1{margin:0;font-size:22px;color:#fff;}
  .hero p{margin:6px 0 0;opacity:.93;font-size:14px;max-width:96ch;color:#e0e7ff;}
  .hero .tag{display:inline-block;background:rgba(255,255,255,.16);border-radius:999px;
    padding:2px 10px;font-size:12px;margin-right:8px;font-weight:600;color:#fff;}
  .cards{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 8px;}
  .card{flex:1 1 150px;min-width:150px;background:#181b23;border:1px solid #272b36;border-radius:12px;
    padding:12px 16px;box-shadow:0 1px 2px rgba(0,0,0,.3);}
  .card .k{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#9aa4b2;margin-bottom:4px;}
  .card .v{font-size:24px;font-weight:700;color:#f1f5f9;font-variant-numeric:tabular-nums;}
  /* Comparison line under a KPI. Neutral grey on purpose: it is a reference
     figure, not a verdict — green read as "better" even on cards where the
     two numbers are identical. */
  .card .d{font-size:12px;color:#8b93a3;margin-top:4px;}
  table{width:100%;border-collapse:collapse;margin:12px 0 20px;font-size:13px;}
  caption{caption-side:top;text-align:left;font-weight:600;color:#cbd5e1;padding:4px 0 8px;font-size:14px;}
  thead th{background:#3730a3;color:#eef2ff;text-align:left;padding:8px 11px;font-weight:600;
    border:none;white-space:nowrap;}
  thead th:first-child{border-top-left-radius:8px;}
  thead th:last-child{border-top-right-radius:8px;}
  tbody td{padding:7px 11px;border-bottom:1px solid #232733;font-variant-numeric:tabular-nums;color:#dbe1ea;}
  tbody tr:nth-child(even){background:#14171e;}
  tbody tr:hover{background:#20232d;}
  tbody tr.group td{background:#1b2030;color:#c7d2fe;font-weight:600;letter-spacing:.02em;}
  .badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap;}
  .badge.ok{background:#0f3d2e;color:#34d399;}
  .badge.fail{background:#4c1d24;color:#fda4af;}
  .badge.warn{background:#43341a;color:#fcd34d;}
  .badge.neutral{background:#2b303b;color:#cbd5e1;}
  .sd{color:#8b93a3;font-size:11px;}
  .sig{display:inline-block;background:#242938;border:1px solid #2f3547;color:#c7d2fe;border-radius:6px;
    padding:1px 6px;margin:1px 2px 1px 0;font-size:11px;white-space:nowrap;}
  .muted{color:#8b93a3;font-size:12px;}
  .err{color:#fda4af;font-size:12px;font-style:italic;}
  .errrow td{background:#2a1418;color:#fecdd3;}
  blockquote{border-left:3px solid #818cf8;background:#14182b;margin:10px 0;padding:8px 14px;
    border-radius:0 8px 8px 0;color:#c7d2fe;font-size:13px;}
  /* plots */
  .plot-wrap{background:#181b23;border:1px solid #272b36;border-radius:12px;padding:16px 20px;margin:12px 0;}
  .plot-title{font-size:14px;font-weight:600;color:#f1f5f9;margin-bottom:12px;}
  .hbar{display:flex;align-items:center;gap:10px;margin:7px 0;}
  .hbar .lab{width:150px;text-align:right;color:#cbd5e1;font-size:12px;white-space:nowrap;}
  .hbar .track{flex:1;background:#262b34;border-radius:4px;height:16px;min-width:120px;position:relative;}
  .hbar .fillwrap{position:absolute;inset:0;border-radius:4px;overflow:hidden;}
  /* 95% CI whisker drawn over the bar: caps at each end, rule between. */
  .hbar .ci{position:absolute;top:50%;transform:translateY(-50%);height:9px;
    border-left:2px solid rgba(255,255,255,.75);border-right:2px solid rgba(255,255,255,.75);}
  .hbar .ci::before{content:"";position:absolute;top:50%;left:0;right:0;height:2px;
    transform:translateY(-50%);background:rgba(255,255,255,.55);}
  .best{background:#0f3d2e;color:#34d399;border-radius:5px;padding:1px 7px;font-weight:700;
    display:inline-block;}
  .hbar .fill{display:block;height:16px;border-radius:4px;}
  .hbar .val{width:96px;color:#f1f5f9;font-weight:600;font-size:12px;font-variant-numeric:tabular-nums;}
  .hbar .nobar{color:#8b93a3;font-size:12px;font-style:italic;}
  .legend{display:flex;gap:16px;margin:8px 0 4px;font-size:12px;color:#cbd5e1;}
  .legend .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px;}
  .grid2{display:flex;gap:20px;flex-wrap:wrap;}
  .grid2>div{flex:1 1 340px;min-width:320px;}
  /* single-page nav + sections */
  .nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 4px;padding:10px 12px;background:#181b23;
    border:1px solid #272b36;border-radius:12px;}
  .nav a{display:inline-block;padding:3px 11px;border-radius:999px;background:#242938;color:#c7d2fe;
    font-size:12px;font-weight:600;text-decoration:none;border:1px solid #2f3547;}
  .nav a:hover{background:#3730a3;color:#eef2ff;}
  .sec{border-top:1px solid #232733;margin-top:26px;padding-top:4px;}
  .sec:first-of-type{border-top:none;}
  .sec>h2{scroll-margin-top:12px;}
  .sec .lead{color:#9aa4b2;font-size:13px;margin:2px 0 6px;}
</style>
"""


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def _dash() -> str:
    return "<span class='muted'>—</span>"


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _num(v):
    if _isnan(v):
        return _dash()
    if isinstance(v, float):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        return s or "0"
    return f"{v:,}"


def _p(v):
    return _dash() if _isnan(v) else f"{v * 100:.0f}%"


def _f(v, nd: int = 3):
    return _dash() if _isnan(v) else f"{v:.{nd}f}"


def _pm(mean, sd, nd: int = 2):
    """A mean with its standard deviation across repeats."""
    if _isnan(mean):
        return _dash()
    if _isnan(sd) or sd == 0:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f}<span class='sd'> ±{sd:.{nd}f}</span>"


def _usd(v, places: int = 4) -> str:
    """A dollar figure, or an explicit n/a — an unpriced arm is never shown as $0."""
    if v is None:
        return "<span class='muted'>n/a</span>"
    if _isnan(v):
        return _dash()
    if v == 0:
        return "$0"
    if v >= 1:
        return f"${v:,.2f}"
    return f"${v:.{places}f}"


def _sd(values) -> float:
    vals = list(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _ci95(values) -> float | None:
    """Half-width of the 95% confidence interval of the mean.

    ``None`` below two samples — a single run has a mean but no spread, and
    drawing a zero-width interval there would imply a precision we do not have.
    """
    vals = list(values)
    if len(vals) < 2:
        return None
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _stability(results) -> dict:
    """How reproducible are the typed decisions across identical repeats?

    For each case we look at the labels its repeats produced: ``agreement`` is
    the share of repeats that landed on that case's modal label, and
    ``unanimous`` is the share of cases whose repeats all agreed.  A cell with a
    single repeat has nothing to say, so both come back as ``None``.
    """
    by_case: dict[str, list] = defaultdict(list)
    for r in results:
        if not r.error:
            by_case[r.case_id].append(r)
    agreements, tool_agreements, unanimous, n_repeated = [], [], 0, 0
    for rs in by_case.values():
        if len(rs) < 2:
            continue
        n_repeated += 1
        labels = Counter(r.pred_label for r in rs)
        tools = Counter(r.pred_tool for r in rs)
        agreements.append(labels.most_common(1)[0][1] / len(rs))
        tool_agreements.append(tools.most_common(1)[0][1] / len(rs))
        unanimous += int(len(labels) == 1)
    if not n_repeated:
        return {
            "agreement": None,
            "ci_agreement": None,
            "tool_agreement": None,
            "unanimous": None,
            "n_repeated_cases": 0,
        }
    return {
        "agreement": statistics.mean(agreements),
        "ci_agreement": _ci95(agreements),
        "tool_agreement": statistics.mean(tool_agreements),
        "unanimous": unanimous / n_repeated,
        "n_repeated_cases": n_repeated,
    }


def _cost_agg(ok) -> dict:
    """Sum the per-unit dollar costs for a cell. Unpriced provider -> None, never 0."""
    costs = [unit_costs(r) for r in ok]
    s1 = sum(c["system1"] for c in costs)
    judge = sum(c["judge"] for c in costs)
    priced = all(c["system2"] is not None for c in costs)
    s2 = sum(c["system2"] for c in costs) if priced else None
    n_success = sum(r.success for r in ok)
    per_unit = [c["pipeline"] for c in costs] if priced else []
    return {
        "ci_cost_per_case": _ci95(per_unit) if per_unit else None,
        "cost_s1": s1,
        "cost_s2": s2,
        "cost_judge": judge,
        "cost_pipeline": None if s2 is None else s1 + s2,
        "cost_total": None if s2 is None else s1 + s2 + judge,
        "cost_per_case": None if s2 is None else (s1 + s2) / len(ok),
        # What you pay per case that actually came out right — the cheap arm is
        # only cheap if its answers are usable.
        "cost_per_success": None if (s2 is None or not n_success) else (s1 + s2) / n_success,
        # Token splits, so the report can show the arithmetic rather than assert a total.
        "jev_in": sum(r.jev_input_tokens for r in ok),
        "jev_out": sum(r.jev_output_tokens for r in ok),
        "s2_in": sum(r.s2_input_tokens for r in ok),
        "s2_out": sum(r.s2_output_tokens for r in ok),
    }


def _agg(results) -> dict:
    """Aggregate one cell (a task x condition x provider group) over its repeats."""
    n = len(results)
    base = {
        "n": n,
        "ok": 0,
        "n_failed": n,
        "first_error": "no units ran",
        "jev_questions": 0,
        "jev_calls_per_case": 0.0,
        "auto": 0.0,
        "review": 0.0,
        "escalate": 0.0,
        "s2_skipped": 0.0,
        "selective_label": None,
        "selective_success": None,
        "jev_latency_per_call": None,
        "s2_latency_per_call": None,
        **_stability(results),
    }
    if n == 0:
        return base
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    base.update({"ok": len(ok), "n_failed": len(failed), "first_error": failed[0].error if failed else None})
    if not ok:
        base.update(
            {
                "jev_questions": 0,
                "jev_calls_per_case": 0.0,
                "auto": 0.0,
                "review": 0.0,
                "escalate": 0.0,
                "s2_skipped": 0.0,
                "selective_label": None,
                "selective_success": None,
                "jev_latency_per_call": None,
                "s2_latency_per_call": None,
                "avg_latency_s": float("nan"),
                "sd_latency_s": float("nan"),
                "ci_latency_s": None,
                "tokens_per_case": float("nan"),
                "battery_asked": 0,
                "battery_returned": 0.0,
                "battery_complete": 0.0,
                "battery_latency_s": float("nan"),
                "ms_per_answer": None,
                "ci_tokens": None,
                "ci_quality": None,
                "p50_s": float("nan"),
                "p95_s": float("nan"),
                "tot_tokens": 0,
                "avg_tokens": float("nan"),
                "tok_per_s": float("nan"),
                "jev_tokens": 0,
                "jev_latency_s": 0.0,
                "s2_tokens": 0,
                "s2_latency_s": 0.0,
                "judge_tokens": 0,
                "label": 0.0,
                "entity": 0.0,
                "tool": 0.0,
                "quality": 0.0,
                "sd_quality": 0.0,
                "guard": 0.0,
                "success": 0.0,
            }
        )
        return base
    latency = sorted(r.latency_s for r in ok)
    acted = [r for r in ok if r.route != "escalate"]  # what the pipeline actually acted on
    base.update(
        {
            "jev_questions": max((r.jev_questions for r in ok), default=0),
            "jev_calls_per_case": statistics.mean(r.jev_calls for r in ok),
            **_cost_agg(ok),
            "auto": sum(1 for r in ok if r.route == "auto") / len(ok),
            "review": sum(1 for r in ok if r.route == "review") / len(ok),
            "escalate": sum(1 for r in ok if r.route == "escalate") / len(ok),
            "s2_skipped": statistics.mean(r.s2_skipped for r in ok),
            "selective_label": statistics.mean(r.label_correct for r in acted) if acted else None,
            "selective_success": statistics.mean(r.success for r in acted) if acted else None,
            "jev_latency_per_call": (
                statistics.mean(r.jev_latency_s / r.jev_calls for r in ok if r.jev_calls)
                if any(r.jev_calls for r in ok)
                else None
            ),
            "s2_latency_per_call": (
                statistics.mean(r.s2_latency_s / r.s2_calls for r in ok if r.s2_calls)
                if any(r.s2_calls for r in ok)
                else None
            ),
            "avg_latency_s": statistics.mean(latency),
            "sd_latency_s": _sd(latency),
            "ci_latency_s": _ci95(latency),
            "tokens_per_case": statistics.mean(r.total_tokens for r in ok),
            "battery_asked": max((r.battery_asked for r in ok), default=0),
            "battery_returned": statistics.mean(r.battery_returned for r in ok),
            "battery_complete": statistics.mean(
                (r.battery_returned / r.battery_asked) if r.battery_asked else 0.0 for r in ok
            ),
            "battery_latency_s": statistics.mean(r.battery_latency_s for r in ok),
            "ms_per_answer": (
                statistics.mean(r.battery_latency_s * 1000 / r.battery_returned for r in ok if r.battery_returned)
                if any(r.battery_returned for r in ok)
                else None
            ),
            "ci_tokens": _ci95([r.total_tokens for r in ok]),
            "ci_quality": _ci95([r.quality for r in ok]),
            "p50_s": latency[len(latency) // 2],
            "p95_s": latency[min(len(latency) - 1, int(len(latency) * 0.95))],
            "tot_tokens": sum(r.total_tokens for r in ok),
            "avg_tokens": statistics.mean(r.total_tokens for r in ok),
            "tok_per_s": statistics.mean(r.tok_per_s for r in ok),
            "jev_tokens": sum(r.jev_tokens for r in ok),
            "jev_latency_s": sum(r.jev_latency_s for r in ok),
            "s2_tokens": sum(r.s2_tokens for r in ok),
            "s2_latency_s": sum(r.s2_latency_s for r in ok),
            "judge_tokens": sum(r.judge_tokens for r in ok),
            "label": statistics.mean(r.label_correct for r in ok),
            "entity": statistics.mean(r.entity_correct for r in ok),
            "tool": statistics.mean(r.tool_correct for r in ok),
            "quality": statistics.mean(r.quality for r in ok),
            "sd_quality": _sd(r.quality for r in ok),
            "guard": statistics.mean(r.guard_correct for r in ok),
            "success": statistics.mean(r.success for r in ok),
        }
    )
    return base


def _group(results) -> dict:
    """results -> {(task, with_system1, provider): [units]}."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r.task, r.condition == "with_system1", r.provider)].append(r)
    return groups


def _cells(results, task_keys) -> dict:
    groups = _group(results)
    return {(tk, w, p): _agg(groups.get((tk, w, p), [])) for tk in task_keys for (w, p) in BENCHMARK_CONDITIONS}


def _across_tasks(results, task_keys) -> dict:
    """Aggregate each condition across every task type (the headline numbers)."""
    by_cond: dict[tuple, list] = defaultdict(list)
    for r in results:
        if r.task in task_keys:
            by_cond[(r.condition == "with_system1", r.provider)].append(r)
    return {c: _agg(by_cond.get(c, [])) for c in BENCHMARK_CONDITIONS}


def _best_provider(overall: dict) -> str | None:
    """The provider with data in both arms — the one we can compare fairly."""
    for _, provider in BENCHMARK_CONDITIONS:
        if overall.get((True, provider), {}).get("ok") and overall.get((False, provider), {}).get("ok"):
            return provider
    return None


# --------------------------------------------------------------------------- #
# Small HTML builders                                                         #
# --------------------------------------------------------------------------- #
def _status_cell(a: dict) -> str:
    if a["ok"] == 0:
        return "<span class='badge fail'>BLOCKED</span>"
    if a["n_failed"]:
        return f"<span class='badge warn'>{a['ok']}/{a['n']} ok</span>"
    return f"<span class='badge ok'>{a['ok']}/{a['n']} ok</span>"


def _table(headers, rows, caption=None) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for row in rows:
        if isinstance(row, str):  # pre-rendered <tr> (group headings, error rows)
            trs.append(row)
            continue
        cells = list(row)[: len(headers)]
        cells += [""] * (len(headers) - len(cells))
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _mark_best(rows: list, values: list[dict | None], directions: dict[int, str]) -> list:
    """Highlight the winning cell of each ranked column, per group block.

    ``values[i]`` holds the raw numbers for data row ``i`` keyed by column index
    (``None`` for rows with nothing to rank); ``directions`` maps a column index
    to ``"max"`` or ``"min"``. Ranking restarts at every group heading, because
    the blocks are different scopes — a support row beating a contract row on
    cost says nothing, they are different work.
    """
    blocks: list[list[int]] = [[]]
    for i, row in enumerate(rows):
        if isinstance(row, str):  # a group heading or error row ends the block
            blocks.append([])
            continue
        blocks[-1].append(i)

    for block in blocks:
        for col, direction in directions.items():
            # Bound to a local so the None check narrows — mypy cannot follow
            # narrowing through a subscript inside a comprehension.
            candidates: list[tuple[int, float]] = []
            for i in block:
                row_values = values[i]
                if row_values is None:
                    continue
                value = row_values.get(col)
                if value is None or _isnan(value):
                    continue
                candidates.append((i, value))
            if len(candidates) < 2:  # nothing to compare against
                continue
            pick = max if direction == "max" else min
            best_value = pick(v for _, v in candidates)
            for i, value in candidates:
                if value == best_value:
                    rows[i][col] = f"<span class='best'>{rows[i][col]}</span>"
    return rows


def _group_row(text: str, span: int) -> str:
    return f"<tr class='group'><td colspan='{span}'>{text}</td></tr>"


def _err_row(err: str, span: int) -> str:
    return f"<tr class='errrow'><td colspan='{span}' class='err'>⚠ {err}</td></tr>"


def _kpi_cards(cards) -> str:
    def _card(k, v, d) -> str:
        delta = f"<div class='d'>{d}</div>" if d else ""
        return f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div>{delta}</div>"

    return f"<div class='cards'>{''.join(_card(*c) for c in cards)}</div>"


def _hbar(label, value, color, fmt="{:.1f}", maxv=None, tokens=False, suffix="", ci=None):
    """One horizontal bar, optionally overlaid with its 95% CI whisker."""
    if _isnan(value):
        return (
            f"<div class='hbar'><span class='lab'>{label}</span>"
            f"<span class='track'></span><span class='nobar'>no data</span></div>"
        )
    maxv = maxv or max(1e-9, value)
    pct = max(0.0, min(100.0, value / maxv * 100))
    vs = f"{int(value):,}" if tokens else fmt.format(value)
    whisker = ""
    if ci:
        low = max(0.0, min(100.0, (value - ci) / maxv * 100))
        high = max(0.0, min(100.0, (value + ci) / maxv * 100))
        # A whisker narrower than a couple of pixels reads as an artefact; skip it.
        if high - low > 0.5:
            whisker = f"<span class='ci' style='left:{low:.1f}%;width:{high - low:.1f}%'></span>"
        vs += f"<span class='sd'> &plusmn;{fmt.format(ci) if not tokens else f'{int(ci):,}'}</span>"
    return (
        f"<div class='hbar'><span class='lab'>{label}</span>"
        f"<span class='track'><span class='fillwrap'>"
        f"<span class='fill' style='width:{pct:.1f}%;background:{color}'></span></span>"
        f"{whisker}</span>"
        f"<span class='val'>{vs}{suffix}</span></div>"
    )


def _plot(title, rows, legend=None, fmt="{:.1f}", tokens=False) -> str:
    """rows: ``(label, value, color)`` or ``(label, value, color, ci95)``.

    The CI is the half-width of the 95% confidence interval of the mean, drawn
    as a whisker over the bar — so a bar whose neighbour's whisker overlaps it
    is not a difference you should read anything into.
    """
    vals = [r[1] for r in rows if not _isnan(r[1])]
    cis = [r[3] for r in rows if len(r) > 3 and r[3]]
    # Scale to the far end of the widest interval, or whiskers clip at the axis.
    maxv = max([v + (c or 0) for r in rows for v, c in [(r[1], r[3] if len(r) > 3 else 0)] if not _isnan(v)] or [1.0])
    maxv = max(maxv, max(vals, default=1.0))
    bars = "".join(
        _hbar(
            r[0],
            r[1],
            r[2],
            fmt=fmt,
            maxv=maxv,
            tokens=tokens,
            ci=(r[3] if len(r) > 3 else None),
        )
        for r in rows
    )
    leg = ""
    if legend:
        sws = "".join(f"<span><span class='sw' style='background:{c}'></span>{n}</span>" for n, c in legend)
        if cis:
            sws += "<span><span class='sw' style='background:rgba(255,255,255,.6)'></span>95% CI</span>"
        leg = f"<div class='legend'>{sws}</div>"
    return f"<div class='plot-wrap'><div class='plot-title'>{title}</div>{leg}{bars}</div>"


def _paired_conditions() -> list[tuple[bool, str]]:
    """Conditions ordered so each provider's two arms sit next to each other.

    ``BENCHMARK_CONDITIONS`` lists all the with-Jev cells then all the without —
    fine for tables that group by arm, useless in a bar chart, where the eye
    wants Jev · Qwen directly above No-Jev · Qwen.
    """
    providers = list(dict.fromkeys(provider for _, provider in BENCHMARK_CONDITIONS))
    return [
        (with_s1, provider)
        for provider in providers
        for with_s1 in (True, False)
        if (with_s1, provider) in BENCHMARK_CONDITIONS
    ]


def _arm(with_s1: bool) -> str:
    return "With <b>Jev</b>" if with_s1 else "Without Jev"


def _cond_label(with_s1: bool, provider: str) -> str:
    return f"{'Jev' if with_s1 else 'No-Jev'} · {_SHORT[provider]}"


# --------------------------------------------------------------------------- #
# Tabs                                                                        #
# --------------------------------------------------------------------------- #
def _render_hero(overall: dict, task_keys, repeats: int, num_cases: int) -> str:
    provider = _best_provider(overall)
    w = overall.get((True, provider), {}) if provider else {}
    wo = overall.get((False, provider), {}) if provider else {}
    speedup = None
    if w.get("ok") and wo.get("ok") and w.get("avg_latency_s"):
        speedup = wo["avg_latency_s"] / w["avg_latency_s"]
    tasks_txt = ", ".join(_TASK_SHORT.get(t, t) for t in task_keys)
    hero = (
        "<div class='hero'>"
        "<h1>TypeSafe (Jev) &times; System 2 — agentic benchmark</h1>"
        f"<p><span class='tag'>{len(task_keys)} task types</span>"
        f"<span class='tag'>{len(BENCHMARK_CONDITIONS)} conditions</span>"
        f"<span class='tag'>{repeats}&times; repeats per cell</span></p>"
        f"<p>{tasks_txt} — each resolved end-to-end twice over: once with Jev pre-structuring the input as "
        "typed guard / label / tool decisions, and once with System 2 alone doing everything from raw text. "
        f"Every cell is run {repeats} times over {num_cases} cases, so latency carries a spread and decisions "
        "carry a stability score.</p>"
        "</div>"
    )
    cards: list[tuple[str, str, str]] = []
    if speedup and provider is not None:
        cards.append(("Latency with Jev", f"{w['avg_latency_s']:.1f}s", f"{wo['avg_latency_s']:.1f}s without"))
        cards.append(("Speed-up", f"&times;{speedup:.1f}", f"on {SYSTEM2_PROVIDERS[provider]['label']}"))
    else:
        cards.append(("Latency with Jev", _dash(), "needs both arms on one provider"))
    cards.append(("Typed answers / call", f"{w.get('jev_questions') or 0}", "in one Jev request (max across tasks)"))
    cards.append(("Acted automatically", _p(w.get("auto")), f"{_p(w.get('escalate'))} escalated to a human"))
    cards.append(("Decision stability", _p(w.get("agreement")), f"{_p(wo.get('agreement'))} without Jev"))
    cards.append(("Classification", _p(w.get("label")), f"{_p(wo.get('label'))} without Jev"))
    cards.append(("Answer quality", _f(w.get("quality"), 2), f"{_f(wo.get('quality'), 2)} without Jev"))
    cards.append(
        (
            "Cost / case",
            _usd(w.get("cost_per_case"), 5),
            f"{_usd(wo.get('cost_per_case'), 5)} without Jev".replace("<span class='muted'>n/a</span>", "n/a"),
        )
    )
    return hero + _kpi_cards(cards)


_MATRIX_HEADERS = [
    "System 2<br>Reasoning/Planning",
    "System 1<br>Decision-making/parsing",
    "Runs",
    "Lat μ (s)",
    "p95",
    "Tokens",
    "Tok/s",
    "Label",
    "Entity",
    "Tool",
    "Guard",
    "Quality",
    "Stability",
    "Success",
]


def _matrix_row(with_s1: bool, provider: str, a: dict) -> list:
    if a["ok"] == 0:
        return [SYSTEM2_PROVIDERS[provider]["label"], _arm(with_s1), _status_cell(a)] + [_dash()] * (
            len(_MATRIX_HEADERS) - 3
        )
    return [
        SYSTEM2_PROVIDERS[provider]["label"],
        _arm(with_s1),
        _status_cell(a),
        _pm(a["avg_latency_s"], a["sd_latency_s"]),
        _f(a["p95_s"], 1),
        f"{a['tot_tokens']:,}",
        _num(a["tok_per_s"]),
        _p(a["label"]),
        _p(a["entity"]),
        _p(a["tool"]),
        _p(a["guard"]),
        _pm(a["quality"], a["sd_quality"]),
        _p(a["agreement"]),
        _p(a["success"]),
        _usd(a["cost_per_case"], 5),
    ]


# Column index -> which way is better, for the matrix table.
_MATRIX_BEST = {
    3: "min",  # Lat μ
    4: "min",  # p95
    6: "max",  # Tok/s
    7: "max",  # Label
    8: "max",  # Entity
    9: "max",  # Tool
    10: "max",  # Guard
    11: "max",  # Quality
    12: "max",  # Stability
    13: "max",  # Success
    14: "min",  # $ / case
}


def _matrix_values(a: dict) -> dict | None:
    """Raw numbers behind a matrix row, for ranking. None when the cell failed."""
    if a["ok"] == 0:
        return None
    return {
        3: a["avg_latency_s"],
        4: a["p95_s"],
        6: a["tok_per_s"],
        7: a["label"],
        8: a["entity"],
        9: a["tool"],
        10: a["guard"],
        11: a["quality"],
        12: a["agreement"],
        13: a["success"],
        14: a["cost_per_case"],
    }


def _render_overview(cells: dict, overall: dict, task_keys) -> str:
    span = len(_MATRIX_HEADERS)
    rows: list = []
    values: list[dict | None] = []
    errs: list[str] = []
    rows.append(_group_row("All task types combined", span))
    values.append(None)
    for w, p in BENCHMARK_CONDITIONS:
        rows.append(_matrix_row(w, p, overall[(w, p)]))
        values.append(_matrix_values(overall[(w, p)]))
    for tk in task_keys:
        rows.append(_group_row(f"{_TASK_SHORT.get(tk, tk)} — {get_task(tk).blurb}", span))
        values.append(None)
        for w, p in BENCHMARK_CONDITIONS:
            a = cells[(tk, w, p)]
            rows.append(_matrix_row(w, p, a))
            values.append(_matrix_values(a))
            if a["ok"] == 0 and a.get("first_error"):
                errs.append(_err_row(f"{_TASK_SHORT.get(tk, tk)} · {_cond_label(w, p)}: {a['first_error']}", span))
    rows = _mark_best(rows, values, _MATRIX_BEST)
    seen, dedup = set(), []
    for e in errs:
        if e not in seen:
            seen.add(e)
            dedup.append(e)
    return _table(
        _MATRIX_HEADERS,
        rows,
        caption="Benchmark matrix — every cell aggregated over its repeats "
        "(μ ± &sigma; where a spread exists; <span class='best'>best</span> per column, within each block)",
    ) + "".join(dedup[:6])


def _render_plots(cells: dict, task_keys) -> str:
    legend = [("With Jev (System 1)", _COLOR["with"]), ("Without Jev", _COLOR["without"])]
    out = [
        "<p class='muted'>Bars are means over all repeats of a cell, with the 95% confidence interval of "
        "the mean drawn as a whisker; where two bars' whiskers overlap, the gap between them is not "
        "evidence of a difference. Each provider's two arms sit next to each other so the with/without "
        "comparison is a single glance rather than a scroll.</p>"
    ]
    for tk in task_keys:
        lat, tok, qual, stab, cost = [], [], [], [], []
        for w, p in _paired_conditions():
            a = cells[(tk, w, p)]
            color = _COLOR["with" if w else "without"]
            lab = _cond_label(w, p)
            ok = a["ok"] > 0
            lat.append((lab, a.get("avg_latency_s") if ok else None, color, a.get("ci_latency_s")))
            tok.append((lab, a.get("tokens_per_case") if ok else None, color, a.get("ci_tokens")))
            qual.append((lab, a.get("quality") if ok else None, color, a.get("ci_quality")))
            stab.append((lab, a.get("agreement") if ok else None, color, a.get("ci_agreement")))
            cost.append((lab, a.get("cost_per_case") if ok else None, color, a.get("ci_cost_per_case")))
        out.append(f"<h3>{_TASK_SHORT.get(tk, tk)}</h3>")
        out.append(
            "<div class='grid2'>"
            + _plot("Cost per case (USD at list rates, lower is better)", cost, legend=legend, fmt="${:.5f}")
            + _plot("Latency &mdash; mean per case (s, lower is better)", lat, legend=legend)
            + "</div><div class='grid2'>"
            + _plot("Tokens &mdash; mean per case", tok, legend=legend, tokens=True)
            + _plot("Answer quality (0&ndash;1, higher is better)", qual, legend=legend, fmt="{:.2f}")
            + "</div><div class='grid2'>"
            + _plot("Decision stability across repeats (higher is better)", stab, legend=legend, fmt="{:.0%}")
            + "</div>"
        )
    return "".join(out)


def _bar_pair(metric: str, wv, wov, fmt="{:.1f}") -> str:
    maxv = max([v for v in (wv, wov) if isinstance(v, (int, float)) and not _isnan(v)] or [1.0]) or 1.0
    wdp = (wv / maxv * 100) if not _isnan(wv) else 0
    wodp = (wov / maxv * 100) if not _isnan(wov) else 0
    return (
        f"<div class='hbar'><span class='lab' style='width:130px'>{metric}</span>"
        f"<span style='width:60px;text-align:right;color:#8b93a3;font-size:11px'>With</span>"
        f"<span class='track'><span class='fill' style='width:{wdp:.1f}%;background:{_COLOR['with']}'></span></span>"
        f"<span class='val'>{fmt.format(wv) if not _isnan(wv) else '—'}</span>"
        f"<span style='width:60px;text-align:right;color:#8b93a3;font-size:11px'>Without</span>"
        f"<span class='track'><span class='fill' "
        f"style='width:{wodp:.1f}%;background:{_COLOR['without']}'></span></span>"
        f"<span class='val'>{fmt.format(wov) if not _isnan(wov) else '—'}</span></div>"
    )


def _render_with_without(cells: dict, overall: dict, task_keys) -> str:
    provider = _best_provider(overall)
    if provider is None:
        return "<p class='muted'>Need both arms to succeed on at least one provider to render this comparison.</p>"
    label = SYSTEM2_PROVIDERS[provider]["label"]
    out = [
        f"<p>Same provider (<b>{label}</b>), same cases, same judge — the only difference is whether Jev "
        "structures the input first. Bars are means over every repeat.</p>",
        f"<div class='legend'><span><span class='sw' style='background:{_COLOR['with']}'></span>With Jev</span>"
        f"<span><span class='sw' style='background:{_COLOR['without']}'></span>Without Jev</span></div>",
    ]
    for scope, w, wo in [("All task types", overall[(True, provider)], overall[(False, provider)])] + [
        (_TASK_SHORT.get(tk, tk), cells[(tk, True, provider)], cells[(tk, False, provider)]) for tk in task_keys
    ]:
        out.append(f"<h3>{scope}</h3>")
        if not w.get("ok") or not wo.get("ok"):
            out.append("<p class='muted'>No paired data for this scope.</p>")
            continue
        for metric, wv, wov, fmt in [
            ("Latency (s)", w["avg_latency_s"], wo["avg_latency_s"], "{:.1f}"),
            ("Tokens", float(w["tot_tokens"]), float(wo["tot_tokens"]), "{:,.0f}"),
            ("Classification", w["label"] * 100, wo["label"] * 100, "{:.0f}%"),
            ("Guard", w["guard"] * 100, wo["guard"] * 100, "{:.0f}%"),
            ("Quality (0&ndash;1)", w["quality"], wo["quality"], "{:.2f}"),
            ("Stability", (w["agreement"] or 0) * 100, (wo["agreement"] or 0) * 100, "{:.0f}%"),
            ("Success", w["success"] * 100, wo["success"] * 100, "{:.0f}%"),
        ]:
            out.append(_bar_pair(metric, wv, wov, fmt))
        if w["avg_latency_s"]:
            sp = wo["avg_latency_s"] / w["avg_latency_s"]
            out.append(
                f"<p class='muted'>&times;{sp:.1f} latency, {(w['label'] - wo['label']) * 100:+.0f}pp classification, "
                f"{(w['quality'] - wo['quality']):+.2f} quality, "
                f"{((w['agreement'] or 0) - (wo['agreement'] or 0)) * 100:+.0f}pp stability with Jev.</p>"
            )
    return "".join(out)


def _render_stability(cells: dict, overall: dict, task_keys, repeats: int) -> str:
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Repeats",
        "Label agreement",
        "Unanimous cases",
        "Tool agreement",
        "Lat &sigma; (s)",
        "Quality &sigma;",
    ]
    best = {4: "max", 5: "max", 6: "max", 7: "min", 8: "min"}
    rows: list = []
    values: list[dict | None] = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 5)
            values.append(None)
            return
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                f"{repeats}&times;",
                _p(a["agreement"]),
                _p(a["unanimous"]),
                _p(a["tool_agreement"]),
                _f(a["sd_latency_s"], 2),
                _f(a["sd_quality"], 2),
            ]
        )
        values.append(
            {
                4: a["agreement"],
                5: a["unanimous"],
                6: a["tool_agreement"],
                7: a["sd_latency_s"],
                8: a["sd_quality"],
            }
        )

    rows.append(_group_row("All task types combined", len(headers)))
    values.append(None)
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        values.append(None)
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])
    intro = (
        "<p>Each cell was run several times on identical input. <b>Label agreement</b> is the share of repeats "
        "that landed on a case's most common answer; <b>unanimous cases</b> is the share of cases where every "
        "repeat agreed. &sigma; columns are the run-to-run spread. This is the axis a single-shot benchmark "
        "cannot see: a pipeline that is right on average but different every time is not something you can put "
        "in front of customers.</p>"
    )
    if repeats < 2:
        intro += "<p class='muted'>Run with <code>--repeats 3</code> or more to populate this tab.</p>"
    elif repeats < 3:
        intro += (
            "<p class='muted'>At two repeats a case can only agree 100% or 50%, so these numbers are coarse "
            "&mdash; three or more gives a real distribution.</p>"
        )
    return intro + _table(
        headers,
        _mark_best(rows, values, best),
        caption="Run-to-run stability (<span class='best'>best</span> per column, within each block)",
    )


def _render_jev_story(cells: dict, overall: dict, task_keys) -> str:
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Jev tok",
        "Jev lat (s)",
        "S2 tok",
        "S2 lat (s)",
        "Judge tok",
        "Total tok",
        "Tok/s",
    ]
    rows: list = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 6)
            return
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                f"{a['jev_tokens']:,}",
                _f(a["jev_latency_s"], 2),
                f"{a['s2_tokens']:,}",
                _f(a["s2_latency_s"], 2),
                f"{a['judge_tokens']:,}",
                f"{a['tot_tokens']:,}",
                _num(a["tok_per_s"]),
            ]
        )

    rows.append(_group_row("All task types combined", len(headers)))
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])
    return (
        "<p>Jev answers guard + classification + tool routing in one typed request, so System 2 is only asked "
        "to write prose. The judge column is identical across both arms by construction.</p>"
        + _table(headers, rows, caption="Token &amp; latency budget — System 1 (Jev) vs System 2 (LLM)")
    )


def _render_quality(cells: dict, overall: dict, task_keys) -> str:
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Runs",
        "Label",
        "Entity",
        "Tool",
        "Guard",
        "Quality",
        "Success",
    ]
    best = {4: "max", 5: "max", 6: "max", 7: "max", 8: "max", 9: "max"}
    rows: list = []
    values: list[dict | None] = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 6)
            values.append(None)
            return
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                _status_cell(a),
                _p(a["label"]),
                _p(a["entity"]),
                _p(a["tool"]),
                _p(a["guard"]),
                _pm(a["quality"], a["sd_quality"]),
                _p(a["success"]),
            ]
        )
        values.append({4: a["label"], 5: a["entity"], 6: a["tool"], 7: a["guard"], 8: a["quality"], 9: a["success"]})

    rows.append(_group_row("All task types combined", len(headers)))
    values.append(None)
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        values.append(None)
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])
    return (
        "<p><b>Label</b> is the task's primary classification (intent / verdict / finding), <b>Guard</b> is "
        "whether hostile or manipulative inputs were refused and left untouched by tools, and <b>Success</b> "
        "requires all four structured fields to be right at once.</p>"
        + _table(
            headers,
            _mark_best(rows, values, best),
            caption="Output quality across the matrix (higher is better; "
            "<span class='best'>best</span> per column, within each block)",
        )
    )


def _route_cell(units) -> str:
    """The modal routing tier for a case, as a badge."""
    routes = Counter(u.route for u in units if u.route)
    if not routes:
        return "<span class='muted'>n/a</span>"
    route, hits = routes.most_common(1)[0]
    css = {"auto": "ok", "review": "warn", "escalate": "fail"}.get(route, "neutral")
    return f"<span class='badge {css}'>{route}</span> <span class='sd'>{hits}/{len(units)}</span>"


def _fired_cell(units) -> str:
    """The atomic signals that fired most often for a case."""
    counts = Counter(sig for u in units for sig in (u.fired or "").split(", ") if sig)
    if not counts:
        return "<span class='muted'>—</span>"
    return " ".join(f"<span class='sig'>{name}</span>" for name, _ in counts.most_common(4))


def _render_task_section(task_key: str, cells: dict, groups: dict, repeats: int) -> str:
    task = get_task(task_key)
    head = [
        f"<p>{task.blurb}</p>",
        f"<p class='muted'>Labels: {', '.join(task.labels)} · tools: {', '.join(task.tools)} · "
        f"{repeats}&times; repeats per case.</p>",
    ]
    cond_headers = [
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Runs",
        "Lat μ (s)",
        "Tokens",
        "Label",
        "Guard",
        "Quality",
        "Stability",
        "Success",
    ]
    cond_rows = []
    for w, p in BENCHMARK_CONDITIONS:
        a = cells[(task_key, w, p)]
        if a["ok"] == 0:
            cond_rows.append([SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 7)
            continue
        cond_rows.append(
            [
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                _status_cell(a),
                _pm(a["avg_latency_s"], a["sd_latency_s"]),
                f"{a['tot_tokens']:,}",
                _p(a["label"]),
                _p(a["guard"]),
                _pm(a["quality"], a["sd_quality"]),
                _p(a["agreement"]),
                _p(a["success"]),
            ]
        )
    head.append(_table(cond_headers, cond_rows, caption="Conditions for this task type"))

    # Per-case detail, averaged over repeats, with the modal prediction per arm.
    case_headers = [
        "Case",
        "Input",
        f"True {task.label_name.lower()}",
        "Arm",
        "Runs",
        "Modal prediction",
        "Route",
        "Signals fired",
        "Lat μ (s)",
        "Label ✓",
        "Guard ✓",
        "Quality",
    ]
    case_rows = []
    for case in task.cases:
        first = True
        for with_s1 in (True, False):
            units = [
                u
                for (tk, w, _p), us in groups.items()
                if tk == task_key and w == with_s1
                for u in us
                if u.case_id == case.id
            ]
            ok = [u for u in units if not u.error]
            if not units:
                continue
            modal = Counter(u.pred_label for u in ok).most_common(1)
            modal_txt = f"{modal[0][0]} <span class='sd'>({modal[0][1]}/{len(ok)})</span>" if modal else _dash()
            case_rows.append(
                [
                    case.id if first else "",
                    (task.preview(case, 70) + (" <span class='badge warn'>hostile</span>" if case.hostile else ""))
                    if first
                    else "",
                    case.label if first else "",
                    _arm(with_s1),
                    f"{len(ok)}/{len(units)}",
                    modal_txt,
                    _route_cell(ok),
                    _fired_cell(ok),
                    _f(statistics.mean(u.latency_s for u in ok), 2) if ok else _dash(),
                    _p(statistics.mean(u.label_correct for u in ok)) if ok else _dash(),
                    _p(statistics.mean(u.guard_correct for u in ok)) if ok else _dash(),
                    _f(statistics.mean(u.quality for u in ok), 2) if ok else _dash(),
                ]
            )
            first = False
    head.append(_table(case_headers, case_rows, caption="Per-case detail (pooled over providers and repeats)"))
    return "".join(head)


# --------------------------------------------------------------------------- #
# Entry points                                                                #
# --------------------------------------------------------------------------- #
def _usd_plain(v) -> str:
    """Dollar figure for the plain-text summary, or n/a when the arm is unpriced."""
    return "n/a" if v is None else f"${v:.5f}"


def _pct_or_na(v) -> str:
    """Percent for the plain-text summary, or n/a when a single repeat says nothing."""
    return "n/a" if v is None else f"{v:.0%}"


def cell_summary(results, task_keys) -> str:
    """Plain-text per-cell summary returned by the benchmark task."""
    groups = _group(results)
    lines = []
    for tk in task_keys:
        lines.append(f"  {tk}:")
        for w, p in BENCHMARK_CONDITIONS:
            a = _agg(groups.get((tk, w, p), []))
            tag = "with-Jev" if w else "no-S1  "
            if a["ok"] == 0:
                lines.append(f"    {tag} x {p}: FAILED ({a.get('first_error')})")
                continue
            lines.append(
                f"    {tag} x {p}: n={a['ok']}/{a['n']}, lat={a['avg_latency_s']:.2f}s ±{a['sd_latency_s']:.2f}, "
                f"tokens={a['tot_tokens']:,}, label={a['label']:.0%}, guard={a['guard']:.0%}, "
                f"quality={a['quality']:.2f}, "
                f"stability={_pct_or_na(a['agreement'])}, "
                f"route(auto/review/esc)={a['auto']:.0%}/{a['review']:.0%}/{a['escalate']:.0%}, "
                f"success={a['success']:.0%}, "
                f"cost/case={_usd_plain(a['cost_per_case'])}"
            )
    return "\n".join(lines)


def _render_routing(cells: dict, overall: dict, task_keys) -> str:
    """Confidence-gated routing: what the pipeline acted on, and what it handed over."""
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Runs",
        "Auto",
        "Review",
        "Escalated",
        "S2 calls skipped",
        "Label (all)",
        "Label (acted on)",
        "Success (acted on)",
    ]
    rows: list = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 7)
            return
        if not w:
            rows.append(
                [scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)]
                + ["<span class='muted'>n/a</span>"] * 4
                + [_p(a["label"]), "<span class='muted'>n/a</span>", "<span class='muted'>n/a</span>"]
            )
            return
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                _status_cell(a),
                _p(a["auto"]),
                _p(a["review"]),
                _p(a["escalate"]),
                _p(a["s2_skipped"]),
                _p(a["label"]),
                _p(a["selective_label"]),
                _p(a["selective_success"]),
            ]
        )

    rows.append(_group_row("All task types combined", len(headers)))
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])
    return (
        "<p>Every <code>Choice</code> and <code>Score</code> answer comes back with calibrated "
        "<b>confidence</b>, so the pipeline has a second decision axis: act automatically, act and flag for "
        "review, or <b>abstain</b>. Abstaining is not a no-op — the escalate branch never calls System 2 at "
        "all, which is why <b>S2 calls skipped</b> tracks the escalation rate. <b>Label (acted on)</b> is "
        "accuracy over just the cases the pipeline did not hand to a human: if the confidence signal is any "
        "good, it is higher than the accuracy over everything. The without-Jev arm has no confidence to gate "
        "on, so it acts on 100% of cases by construction.</p>"
        + _table(headers, rows, caption="Confidence-gated routing and selective accuracy")
    )


def _render_fanout(cells: dict, overall: dict, task_keys) -> str:
    """The economics of asking many small questions in one call."""
    headers = [
        "Scope",
        "Questions / Jev call",
        "Jev calls / case",
        "Jev latency / call (s)",
        "S2 latency / call (s)",
        "Jev tok / case",
        "S2 tok / case",
    ]
    rows = []
    for tk in task_keys:
        task = get_task(tk)
        cell = next((cells[(tk, True, p)] for _, p in BENCHMARK_CONDITIONS if cells[(tk, True, p)]["ok"]), None)
        no_s1 = next((cells[(tk, False, p)] for _, p in BENCHMARK_CONDITIONS if cells[(tk, False, p)]["ok"]), None)
        speculative = sum(1 for sig in task.signals if sig.speculative)
        if cell is None:
            rows.append([_TASK_SHORT.get(tk, tk), f"{len(task.signals) + 3}"] + [_dash()] * 5)
            continue
        rows.append(
            [
                _TASK_SHORT.get(tk, tk),
                f"{cell['jev_questions']} <span class='sd'>({speculative} speculative)</span>",
                _f(cell["jev_calls_per_case"], 2),
                _f(cell["jev_latency_per_call"], 2),
                _f(cell["s2_latency_per_call"], 2),
                f"{round(cell['jev_tokens'] / max(1, cell['ok'])):,}",
                f"{round(cell['s2_tokens'] / max(1, cell['ok'])):,}"
                + (
                    f" <span class='sd'>(vs {round(no_s1['s2_tokens'] / max(1, no_s1['ok'])):,} "
                    "with no System 1)</span>"
                    if no_s1 and no_s1["ok"]
                    else ""
                ),
            ]
        )
    return (
        "<p>A System One call answers every question in parallel and in isolation, and "
        "<i>&ldquo;adding questions barely changes the response time&rdquo;</i> — so each task asks a dozen-plus "
        "atomic questions where a prompt pipeline would ask one, and pays roughly one call's latency for it. "
        "Some of those questions are <b>speculative</b>: urgency, tone, reversibility, whether counsel is "
        "needed. They are not used to derive the verdict; they come along because they are nearly free and a "
        "real system would want them.</p>"
        + _table(headers, rows, caption="Fan-out economics — one call, many typed answers")
    )


def _rates_table() -> str:
    """The published list rates every dollar figure on this page is derived from."""
    headers = ["Side", "Input $/MTok", "Output $/MTok", "Source"]
    rows = [
        [
            f"<b>System 1</b> — {JEV.label}",
            f"${JEV.input_usd_per_mtok:g}",
            f"${JEV.output_usd_per_mtok:g}"
            + (" <span class='sd'>(assumed — no output price published)</span>" if JEV.output_is_assumed else ""),
            f"<a href='{JEV.source}' style='color:#818cf8'>{JEV.source}</a>",
        ]
    ]
    for provider in dict.fromkeys(p for _, p in BENCHMARK_CONDITIONS):
        price = system2_price(provider)
        label = SYSTEM2_PROVIDERS[provider]["label"]
        if price is None:
            rows.append(
                [
                    f"<b>System 2</b> — {label}",
                    "<span class='muted'>n/a</span>",
                    "<span class='muted'>n/a</span>",
                    "<span class='sd'>no published or configured price</span>",
                ]
            )
            continue
        host = SELF_HOSTED.get(provider)
        if host is not None:
            # A model you host bills by the hour, so show the conversion rather
            # than a bare number that looks like a published rate.
            rows.append(
                [
                    f"<b>System 2</b> — {price.label}",
                    f"${price.input_usd_per_mtok:.3f} <span class='sd'>derived</span>",
                    f"${price.output_usd_per_mtok:.2f} <span class='sd'>derived</span>",
                    f"<span class='sd'>{host.derivation()}</span>",
                ]
            )
            continue
        rows.append(
            [
                f"<b>System 2</b> — {price.label}",
                f"${price.input_usd_per_mtok:g}",
                f"${price.output_usd_per_mtok:g}",
                f"<a href='{price.source}' style='color:#818cf8'>{price.source}</a>",
            ]
        )
    note = ""
    if SELF_HOSTED:
        host = next(iter(SELF_HOSTED.values()))
        sat = host.as_price()
        tiers = " · ".join(
            f"at {u:.0%} utilization ${sat.input_usd_per_mtok / u:.3f} in / ${sat.output_usd_per_mtok / u:.2f} out"
            for u in (0.5, 0.25, 0.1)
        )
        note = (
            "<p class='muted'><b>The self-hosted row is a lower bound.</b> An instance bills wall-clock "
            "whether or not it is serving, so its per-token cost is the saturated figure divided by how busy "
            f"you keep it: {tiers}. That is the structural difference from a hosted API, where idle time is "
            "free — and at low duty cycle a self-hosted 27B can cost more per token than Sonnet. Override "
            "<code>QWEN_USD_PER_HOUR</code>, <code>QWEN_PREFILL_TOK_S</code>, "
            "<code>QWEN_DECODE_TOK_S</code> and <code>QWEN_UTILIZATION</code> with your measured numbers.</p>"
        )
    return _table(headers, rows, caption="Rates behind every figure below") + note


def _cost_math_table(cells: dict, overall: dict, task_keys) -> str:
    """The arithmetic itself: tokens x rate, per side, per cell."""
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Jev tok (in/out)",
        "Jev $",
        "S2 tok (in/out)",
        "S2 $",
        "Pipeline $",
        "$ / case",
        "$ / 1,000 cases",
        "$ / success",
    ]
    # Jev $ is excluded on purpose: the without-Jev arm spends $0 there by
    # construction, so "best" would just relabel the arm that did no work.
    best = {7: "min", 8: "min", 9: "min", 10: "min"}
    rows: list = []
    values: list[dict | None] = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 7)
            values.append(None)
            return
        per_case = a["cost_per_case"]
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                f"{a['jev_in']:,} / {a['jev_out']:,}",
                _usd(a["cost_s1"]),
                f"{a['s2_in']:,} / {a['s2_out']:,}",
                _usd(a["cost_s2"]),
                _usd(a["cost_pipeline"]),
                _usd(per_case, 5),
                _usd(None if per_case is None else per_case * 1000, 2),
                _usd(a["cost_per_success"], 5),
            ]
        )
        values.append(
            {
                7: a["cost_pipeline"],
                8: per_case,
                9: None if per_case is None else per_case * 1000,
                10: a["cost_per_success"],
            }
        )

    rows.append(_group_row("All task types combined", len(headers)))
    values.append(None)
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        values.append(None)
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])
    return _table(
        headers,
        _mark_best(rows, values, best),
        caption="Cost in USD &mdash; tokens &times; list rate, per side "
        "(<span class='best'>cheapest</span> per column, within each block)",
    )


def _worked_example(overall: dict) -> str:
    """One cell, spelled out, so the numbers above are checkable rather than trusted."""
    # The first provider that both ran and has a price — an unpriced arm has no
    # arithmetic to show. Bind the price itself rather than looking it up again
    # afterwards, so it is a plain `Price` from here down.
    price = None
    a: dict = {}
    for candidate in dict.fromkeys(k for _, k in BENCHMARK_CONDITIONS):
        candidate_price = system2_price(candidate)
        if candidate_price is not None and overall[(True, candidate)]["ok"]:
            price, a = candidate_price, overall[(True, candidate)]
            break
    if price is None:
        return ""
    jev_cost = (a["jev_in"] + a["jev_out"]) * JEV.input_usd_per_mtok / 1e6
    s2_cost = (a["s2_in"] * price.input_usd_per_mtok + a["s2_out"] * price.output_usd_per_mtok) / 1e6

    # Lay the arithmetic out in columns so the "=" line up in the <pre>.
    lines = [
        (
            "System 1",
            f"({a['jev_in']:,} in + {a['jev_out']:,} out) &times; ${JEV.input_usd_per_mtok:g}/MTok",
            _usd(jev_cost),
        ),
        (
            "System 2",
            f"{a['s2_in']:,} in &times; ${price.input_usd_per_mtok:g}/MTok + "
            f"{a['s2_out']:,} out &times; ${price.output_usd_per_mtok:g}/MTok",
            _usd(s2_cost),
        ),
        ("Pipeline", f"{_usd(jev_cost)} + {_usd(s2_cost)}", _usd(jev_cost + s2_cost)),
        (
            "Per case",
            f"{_usd(jev_cost + s2_cost)} / {a['ok']} cases",
            _usd((jev_cost + s2_cost) / a["ok"], 5),
        ),
    ]
    # &times; renders as one glyph but is 7 characters of source — measure the
    # visible width, not the markup, or the columns drift.
    width = max(len(expr.replace("&times;", "x")) for _, expr, _ in lines)
    body = "\n".join(
        f"{label:<10}{expr}{' ' * (width - len(expr.replace('&times;', 'x')))}  =  {total}"
        for label, expr, total in lines
    )
    return (
        "<div class='plot-wrap'><div class='plot-title'>Worked example &mdash; "
        f"with Jev &times; {price.label}, all task types, {a['ok']} cases</div>"
        "<pre style='margin:0;font-size:12.5px;line-height:1.7;color:#cbd5e1;white-space:pre-wrap'>"
        f"{body}</pre></div>"
    )


def _render_cost(cells: dict, overall: dict, task_keys) -> str:
    """What each arm actually costs, split System 1 vs System 2, with the arithmetic."""
    intro = (
        "<p>Every token this benchmark spent, priced at each vendor's public list rate. "
        "<b>Pipeline $</b> is what it costs to actually run the arm (System 1 + System 2). "
        "<b>$ / success</b> divides that by the cases that came out right — a cheap arm is only "
        "cheap if its answers are usable. The benchmark's own Jev <b>judge</b> is billed separately "
        "below: it is identical across both arms by construction, so it is measurement overhead, not "
        "product cost.</p>"
        "<p class='muted'>Neither prompt caching (0.1&times; input on reads) nor the Batch API (50% off) "
        "is used here, so every token bills at the base rate. These are list-price equivalents &mdash; a "
        "negotiated contract, or a gateway that marks up or absorbs cost, bills something different. "
        "Where an arm has no published price it reads <b>n/a</b>, never $0.</p>"
    )
    judge_headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Judge $ (measurement)",
        "Total incl. judge $",
    ]
    judge_rows = []
    for w, p in BENCHMARK_CONDITIONS:
        a = overall[(w, p)]
        if a["ok"] == 0:
            judge_rows.append(["all", SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a), _dash()])
            continue
        judge_rows.append(["all", SYSTEM2_PROVIDERS[p]["label"], _arm(w), _usd(a["cost_judge"]), _usd(a["cost_total"])])
    return (
        intro
        + _rates_table()
        + _cost_math_table(cells, overall, task_keys)
        + _worked_example(overall)
        + _table(
            judge_headers,
            judge_rows,
            caption="Grading overhead &mdash; the benchmark's own Jev judge, identical across arms",
        )
    )


# --------------------------------------------------------------------------- #
# "Agent Tasks" tab — how each pipeline is wired                              #
# --------------------------------------------------------------------------- #
_PIPELINE_CSS = """
<style>
  .pipe-intro{color:#dbe1ea;font-size:14px;margin:6px 0 14px;max-width:100ch;}
  .toggle-row{display:flex;align-items:center;gap:14px;margin:8px 0 22px;padding:12px 16px;
    background:#181b23;border:1px solid #272b36;border-radius:12px;position:sticky;top:0;z-index:5;}
  .toggle-row .lbl{font-size:13px;color:#cbd5e1;font-weight:600;}
  .switch{position:relative;display:inline-block;width:52px;height:28px;flex:none;}
  .switch input{opacity:0;width:0;height:0;}
  .slider{position:absolute;inset:0;cursor:pointer;background:#3a4152;border-radius:999px;
    transition:background .25s ease;}
  .slider:before{content:"";position:absolute;height:22px;width:22px;left:3px;top:3px;background:#f1f5f9;
    border-radius:50%;transition:transform .25s cubic-bezier(.4,0,.2,1);}
  .switch input:checked + .slider{background:#6d28d9;}
  .switch input:checked + .slider:before{transform:translateX(24px);}
  .toggle-hint{font-size:12px;color:#8b93a3;}
  .pipe{background:#181b23;border:1px solid #272b36;border-radius:14px;padding:18px 20px 20px;margin:0 0 22px;}
  .pipe h3{margin:0 0 2px;font-size:16px;color:#f1f5f9;}
  .pipe .sub{font-size:12.5px;color:#8b93a3;margin:0 0 16px;}

  /* One row, always: the boxes share the width and shrink rather than wrap. */
  .flow{display:flex;flex-wrap:nowrap;align-items:stretch;gap:0;}
  .step{flex:1 1 0;min-width:0;border-radius:12px;padding:11px 12px;border:1px solid #2f3547;
    background:#1d2230;cursor:default;
    transition:flex-grow .3s cubic-bezier(.4,0,.2,1), background .25s ease, border-color .25s ease;}
  .step .kind{font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:#8b93a3;margin-bottom:4px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .step .name{font-size:13px;font-weight:700;color:#f1f5f9;line-height:1.3;}
  /* The description is hidden until you hover the box, which then takes the room it needs. */
  .step .detail{font-size:11.5px;color:#aab3c0;line-height:1.5;max-height:0;opacity:0;overflow:hidden;
    margin-top:0;transition:max-height .3s cubic-bezier(.4,0,.2,1), opacity .22s ease, margin-top .3s ease;}
  .step:hover,.step:focus-within{flex-grow:3.4;background:#232a3a;border-color:#4b5470;}
  .step:hover .detail,.step:focus-within .detail{max-height:240px;opacity:1;margin-top:7px;}
  .step.s1{background:linear-gradient(160deg,#241b4d 0%,#1d1840 100%);border-color:#4c3fa8;}
  .step.s1:hover,.step.s1:focus-within{background:linear-gradient(160deg,#2c2160 0%,#241d52 100%);
    border-color:#6d5ce0;}
  .step.s2{background:linear-gradient(160deg,#102f4a 0%,#132539 100%);border-color:#2b5f86;}
  .step.s2:hover,.step.s2:focus-within{background:linear-gradient(160deg,#143a5c 0%,#172c45 100%);
    border-color:#3d7cab;}
  .step.rt{background:linear-gradient(160deg,#10382c 0%,#123027 100%);border-color:#1f6b52;}
  .step.rt:hover,.step.rt:focus-within{background:linear-gradient(160deg,#134634 0%,#153b30 100%);
    border-color:#2b8c6b;}
  .step.code{background:#1d2230;border-color:#3a4152;border-style:dashed;}
  .step .badge-n{display:inline-block;margin-top:6px;padding:1px 7px;border-radius:999px;font-size:10.5px;
    font-weight:700;background:rgba(129,140,248,.18);color:#c7d2fe;white-space:nowrap;}
  .arrow{align-self:center;flex:0 0 18px;text-align:center;color:#4b5364;font-size:15px;
    transition:opacity .3s ease;}
  /* Toggle off: the System 1 steps collapse out and the path re-routes around them. */
  .no-s1 .step.s1,.no-s1 .arrow.s1{opacity:0;transform:scale(.94);filter:blur(1px);
    flex:0 0 0;min-width:0;padding:0;margin:0;border-width:0;overflow:hidden;}
  .no-s1 .s1-only{display:none;}
  .no-s1-only{display:none;}
  .no-s1 .no-s1-only{display:inline;}

  /* Hardcoded example of one case going through the pipeline. */
  .example{margin-top:14px;border:1px solid #2f3547;border-radius:10px;background:#151922;overflow:hidden;}
  .example summary{cursor:pointer;padding:10px 14px;font-size:12.5px;font-weight:600;color:#c7d2fe;
    list-style:none;user-select:none;transition:background .2s ease;}
  .example summary::-webkit-details-marker{display:none;}
  .example summary:before{content:"\\25B8";display:inline-block;margin-right:8px;color:#818cf8;
    transition:transform .25s ease;}
  .example[open] summary:before{transform:rotate(90deg);}
  .example summary:hover{background:#1b2030;}
  .ex-grid{display:flex;gap:14px;flex-wrap:wrap;padding:0 14px 10px;}
  .ex-grid > div{flex:1 1 340px;min-width:280px;}
  .ex-grid h4{margin:0 0 6px;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#8b93a3;}
  .ex-grid pre{margin:0;padding:11px 13px;border-radius:9px;background:#0f1117;border:1px solid #272b36;
    font-size:11.5px;line-height:1.55;color:#cbd5e1;white-space:pre-wrap;word-break:break-word;
    max-height:320px;overflow:auto;}
  .ex-note{font-size:11px;color:#6f7789;padding:0 14px 12px;margin:0;}

  .verdict{margin-top:14px;padding:11px 14px;border-radius:10px;font-size:12.5px;line-height:1.55;
    border:1px solid #2f3547;background:#151922;color:#cbd5e1;}
  .verdict b{color:#f1f5f9;}
  .pulse{animation:pulse 2.6s ease-in-out infinite;}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(129,140,248,0);}
    50%{box-shadow:0 0 0 4px rgba(129,140,248,.14);}}
  .flowdot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#818cf8;margin-right:6px;
    animation:travel 2.2s ease-in-out infinite;}
  @keyframes travel{0%{transform:translateX(0);opacity:.35;}50%{transform:translateX(5px);opacity:1;}
    100%{transform:translateX(0);opacity:.35;}}
  @media (prefers-reduced-motion: reduce){
    .pulse,.flowdot{animation:none;} .step,.arrow,.slider,.slider:before,.step .detail{transition:none;}}
</style>
"""

# Illustrative outputs, hardcoded on purpose: they show the *shape* of the deliverable —
# the battery, the routing decision and the prose — without printing all 75-89 booleans.
_EXAMPLE_OUTPUT = {
    "support": """{
  "label":    "delivery_status",     // 1 of 7 intents
  "entity":   "AC-1042",
  "tool":     "lookup_delivery",
  "severity": 2,                     // "serious: a deadline is involved"
  "route":    "auto",                // confidence 0.94 -> act without a human
  "signals": {
    "hostile": false,          "needs_backend": true,    "has_order_reference": true,
    "reports_late": true,      "money_at_stake": false,  "frustrated": true,
    "sla_risk": true,          "asks_for_human": false,  "repeat_contact": false,
    ...66 more facets...
  },
  "answer": "Your parcel AC-1042 is still in transit and is now scheduled to
             arrive tomorrow by end of day. Apologies for the delay past the
             original estimate - nothing is needed from you."
}""",
    "code_review": """{
  "label":    "malicious",           // 1 of 4 verdicts
  "entity":   "src/auth/session.py",
  "tool":     "static_scan",
  "severity": 4,                     // "block and escalate to security"
  "route":    "escalate",            // guard fired -> no generation spent
  "signals": {
    "backdoor": true,          "exfiltration": false,    "remote_execution": false,
    "implements_intent": true, "out_of_scope_edits": false, "touches_auth": true,
    "adds_tests": false,       "rollback_safe": true,    "needs_security_review": true,
    ...74 more facets...
  },
  "answer": "The timeout fix also grants any @partner-support.co address a session
             that never expires. That is an authorization bypass, not a timeout
             change. Blocking and escalating to security."
}""",
    "contract": """{
  "label":    "term_mismatch",       // 1 of 5 findings
  "entity":   "$21,000",
  "tool":     "compute_fee",
  "severity": 2,                     // "a term must be corrected before signature"
  "route":    "review",              // confidence 0.81 -> act, but flag a human
  "signals": {
    "deceptive_request": false,   "unlawful_clause": false, "figure_conflict": true,
    "required_clause_missing": false, "money_conflict": true, "has_governing_law": true,
    "has_payment_terms": true,    "has_liability_cap": false, "has_term_length": true,
    ...77 more facets...
  },
  "answer": "The draft states a fixed fee of $21,000 where the term sheet agreed
             $12,000. Correct the figure and restate the three milestone payments
             at $4,000 each before signature."
}""",
}


def _esc_pre(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _step(kind: str, css: str, name: str, detail: str, badge: str = "") -> str:
    """A pipeline box: title only, until you hover it."""
    chip = f"<div class='badge-n'>{badge}</div>" if badge else ""
    return (
        f"<div class='step {css}' tabindex='0'><div class='kind'>{kind}</div>"
        f"<div class='name'>{name}</div>{chip}<div class='detail'>{detail}</div></div>"
    )


def _arrow(css: str = "") -> str:
    return f"<div class='arrow {css}'>&rarr;</div>"


def _example_block(task) -> str:
    """Collapsed expander: a real input case next to an illustrative output."""
    if not task.cases:
        return ""
    case = task.cases[0]
    shown = "\n\n".join(f"{key}:\n{value}" for key, value in case.state.items())
    battery = task.battery_size()
    return (
        "<details class='example'>"
        "<summary>Example input and output &mdash; what one case looks like</summary>"
        "<div class='ex-grid'>"
        f"<div><h4>Input &mdash; case {case.id}</h4><pre>{_esc_pre(shown)}</pre></div>"
        f"<div><h4>Output &mdash; {battery} typed answers + prose</h4>"
        f"<pre>{_esc_pre(_EXAMPLE_OUTPUT.get(task.key, ''))}</pre></div>"
        "</div>"
        "<p class='ex-note'>The input is a real eval case; the output is abbreviated to show the shape of the "
        f"deliverable rather than all {battery} fields. With Jev the whole battery arrives from one request; "
        "without it, every field has to be generated one token at a time.</p>"
        "</details>"
    )


def _pipeline_card(task) -> str:
    """One task type's pipeline, drawn so the System 1 / System 2 split is obvious."""
    battery = task.battery_size()
    facets = sum(1 for sig in task.signals if sig.speculative)
    deciding = len(task.signals) - facets
    tools = ", ".join(name for name in task.tools if name != "none")
    inputs = ", ".join(task.cases[0].state) if task.cases else "input"

    flow = (
        _step("input", "code", "Raw input", f"{inputs} &mdash; unstructured text, no schema.")
        + _arrow()
        + _step(
            "System 1 &middot; Jev",
            "s1 pulse",
            "One fan-out call",
            f"<span class='flowdot'></span>{battery} typed questions answered in parallel and in isolation: "
            f"{deciding} that decide the verdict, {facets} facets a real reviewer wants anyway. Adding "
            "questions barely moves the latency.",
            f"{battery} answers",
        )
        + _arrow("s1")
        + _step(
            "your code",
            "code",
            "Compose + gate",
            "A precedence rule turns the symptoms into a verdict; calibrated confidence routes it "
            "<b>auto</b> / <b>review</b> / <b>escalate</b>. Changing what counts as blocking is a code edit, "
            "not a prompt rewrite.",
        )
        + _arrow("s1")
        + _step(
            "AI runtime &middot; Flyte",
            "rt",
            "Tool call",
            f"Durable and fanned out, one action per call: {tools}.",
        )
        + _arrow()
        + _step(
            "System 2 &middot; LLM",
            "s2",
            "Generation",
            "<span class='s1-only'>Writes the prose only &mdash; the structure already exists, and escalated "
            "cases skip this step entirely.</span>"
            f"<span class='no-s1-only'>Must produce all <b>{battery}</b> typed answers itself, "
            "autoregressively, one token at a time &mdash; and the prose.</span>",
        )
        + _arrow("s1")
        + _step(
            "System 1 &middot; Jev",
            "s1",
            "Verify",
            "A second small battery: is the answer grounded in the tool output, and confident enough to send?",
        )
    )

    verdict = (
        f"<div class='verdict'><span class='s1-only'><b>With Jev:</b> the {battery} typed answers come back "
        "from a single request, evaluated in parallel and in isolation, and the verdict is composed in code "
        "you can read and change. System 2 is left with the one job it is best at &mdash; writing.</span>"
        f"<span class='no-s1-only'><b>Without Jev:</b> one model does everything. The same {battery} answers "
        "have to be <i>generated</i> in sequence, so the deliverable costs output tokens and wall-clock that "
        "scale with how much structure you asked for &mdash; and nothing guarantees the JSON comes back "
        "complete.</span></div>"
    )
    return (
        f"<div class='pipe'><h3>{task.label}</h3><p class='sub'>{task.blurb}</p>"
        f"<div class='flow'>{flow}</div>{_example_block(task)}{verdict}</div>"
    )


def _render_parallel_output(cells: dict, overall: dict, task_keys) -> str:
    """How fast each arm produces the structured deliverable, and whether it is complete."""
    headers = [
        "Scope",
        "System 2<br>Reasoning/Planning",
        "System 1<br>Decision-making/parsing",
        "Runs",
        "Typed answers asked",
        "Returned (mean)",
        "Complete",
        "Time to produce (s)",
        "ms / answer",
    ]
    best = {5: "max", 6: "max", 7: "min", 8: "min"}
    rows: list = []
    values: list[dict | None] = []

    def add(scope, w, p, a):
        if a["ok"] == 0:
            rows.append([scope, SYSTEM2_PROVIDERS[p]["label"], _arm(w), _status_cell(a)] + [_dash()] * 5)
            values.append(None)
            return
        rows.append(
            [
                scope,
                SYSTEM2_PROVIDERS[p]["label"],
                _arm(w),
                _status_cell(a),
                f"{a['battery_asked']}",
                _f(a["battery_returned"], 1),
                _p(a["battery_complete"]),
                _f(a["battery_latency_s"], 2),
                _f(a["ms_per_answer"], 1),
            ]
        )
        values.append(
            {
                5: a["battery_returned"],
                6: a["battery_complete"],
                7: a["battery_latency_s"],
                8: a["ms_per_answer"],
            }
        )

    rows.append(_group_row("All task types combined", len(headers)))
    values.append(None)
    for w, p in BENCHMARK_CONDITIONS:
        add("all", w, p, overall[(w, p)])
    for tk in task_keys:
        rows.append(_group_row(_TASK_SHORT.get(tk, tk), len(headers)))
        values.append(None)
        for w, p in BENCHMARK_CONDITIONS:
            add(_TASK_SHORT.get(tk, tk), w, p, cells[(tk, w, p)])

    sizes = ", ".join(f"{_TASK_SHORT.get(tk, tk)} {get_task(tk).battery_size()}" for tk in task_keys)
    return (
        "<p>Both arms owe the <b>same artifact</b>: the full structured analysis — every typed answer, not "
        f"just a verdict ({sizes}). Jev answers the whole battery in <b>one request</b>, evaluated in "
        "parallel and in isolation, so the time it takes barely moves with how many questions you ask. "
        "System 2 has to <i>generate</i> each field autoregressively, so its time and its output-token bill "
        "scale with the size of the battery.</p>"
        "<p class='muted'><b>Complete</b> is the share of requested answers that actually came back, counted "
        "from the response rather than the request: Jev returns every one by construction, while generated "
        "JSON can be short, malformed, or quietly missing half the fields. <b>ms / answer</b> is the honest "
        "throughput number — wall-clock for the structuring step divided by the answers it produced.</p>"
        + _table(
            headers,
            _mark_best(rows, values, best),
            caption="Producing the structured deliverable (<span class='best'>best</span> per column, "
            "within each block)",
        )
    )


def _render_agent_tasks(task_keys) -> str:
    """The 'Agent Tasks' tab: each pipeline, with System 1 switchable in and out."""
    cards = "".join(_pipeline_card(get_task(tk)) for tk in task_keys)
    toggle_js = "document.getElementById('pipes').classList.toggle('no-s1', !this.checked)"
    return (
        "<div class='typesafe-report'>"
        + _CSS
        + _PIPELINE_CSS
        + "<h2>Agent Tasks</h2>"
        + "<p class='pipe-intro'>Every task type runs the same shape: structure the input, decide what to do, "
        "do it, then write the answer. What changes is <b>who does the structuring</b> &mdash; a System One "
        "model answering a wide battery in one parallel call, or a generative model writing every field out "
        "in sequence. Flip the switch to take System 1 out and watch the work pile onto the generative "
        "model.</p>" + "<div class='toggle-row'>"
        f"<label class='switch'><input type='checkbox' id='s1toggle' checked onchange=\"{toggle_js}\">"
        "<span class='slider'></span></label>"
        "<span class='lbl'>System 1 (Jev) in the pipeline</span>"
        "<span class='toggle-hint'>switch off &rarr; System 2 has to produce every typed answer itself</span>"
        "</div>"
        f"<div id='pipes'>{cards}</div></div>"
    )


def _sections(cells: dict, overall: dict, groups: dict, task_keys, repeats: int) -> list[tuple[str, str, str]]:
    """The page, as (anchor, heading, html) — rendered in order into one tab."""
    sections = [
        ("matrix", "Benchmark matrix", _render_overview(cells, overall, task_keys)),
        ("plots", "Plots", _render_plots(cells, task_keys)),
        ("with-vs-without", "With vs without System 1 (Jev)", _render_with_without(cells, overall, task_keys)),
        ("routing", "Confidence-gated routing", _render_routing(cells, overall, task_keys)),
        ("fanout", "Fan-out economics", _render_fanout(cells, overall, task_keys)),
        (
            "parallel-output",
            "Parallel structured output",
            _render_parallel_output(cells, overall, task_keys),
        ),
        ("cost", "What it costs", _render_cost(cells, overall, task_keys)),
        ("stability", "Run-to-run stability", _render_stability(cells, overall, task_keys, repeats)),
        (
            "budget",
            "Jev is the decision logic; System 2 writes the prose",
            _render_jev_story(cells, overall, task_keys),
        ),
        ("quality", "Output quality", _render_quality(cells, overall, task_keys)),
    ]
    sections += [
        (f"task-{tk}", f"{_TASK_SHORT.get(tk, tk)} — per case", _render_task_section(tk, cells, groups, repeats))
        for tk in task_keys
    ]
    return sections


def _nav(sections) -> str:
    links = "".join(f"<a href='#{anchor}'>{heading.split(' — ')[0]}</a>" for anchor, heading, _ in sections)
    return f"<div class='nav'>{links}</div>"


def build_report(results: list, task_keys, repeats: int = 1, num_cases: int = 0) -> None:
    """Render the whole benchmark into a single report tab."""
    task_keys = list(task_keys)
    groups = _group(results)
    cells = _cells(results, task_keys)
    overall = _across_tasks(results, task_keys)

    sections = _sections(cells, overall, groups, task_keys, repeats)
    body = "".join(
        f"<div class='sec'><h2 id='{anchor}'>{heading}</h2>{html}</div>" for anchor, heading, html in sections
    )
    page = (
        "<div class='typesafe-report'>"
        + _CSS
        + _render_hero(overall, task_keys, repeats, num_cases)
        + _nav(sections)
        + body
        + "</div>"
    )
    flyte.report.get_tab("Benchmark Summary").log(page)
    flyte.report.get_tab("Agent Tasks").log(_render_agent_tasks(task_keys))
