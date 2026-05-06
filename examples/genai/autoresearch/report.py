from __future__ import annotations

import dataclasses
import html
import json

from autoresearch_types import HistoryEntry


def history_val_bpb_svg(points: list[tuple[int, float, str | None]]) -> str:
    """Inline SVG: experiment round (x) vs val_bpb (y), lower is better."""
    if not points:
        return ""
    xs = [float(p[0]) for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin_raw, ymax_raw = min(ys), max(ys)
    span = ymax_raw - ymin_raw
    pad = max(span * 0.1, 1e-9) if span > 0 else max(abs(ymin_raw) * 0.05, 0.01)
    ymin, ymax = ymin_raw - pad, ymax_raw + pad
    if xmax <= xmin:
        xmax = xmin + 1.0

    pad_l, pad_r, pad_t, pad_b = 56, 28, 20, 44
    w, h = 640, 320
    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / (xmax - xmin) * gw

    def sy(y: float) -> float:
        return pad_t + gh - (y - ymin) / (ymax - ymin) * gh

    # A "kept" point is a new running minimum (strictly better than all prior points).
    best_indices: list[int] = []
    running_min = float("inf")
    for i, (_, y, _) in enumerate(points):
        if y < running_min:
            running_min = y
            best_indices.append(i)

    best_idx_set = set(best_indices)
    best_points = [points[i] for i in best_indices]

    # Stair-step line for the running best trajectory.
    best_step_points: list[tuple[float, float]] = []
    if best_points:
        prev_x, prev_y, _ = best_points[0]
        best_step_points.append((float(prev_x), prev_y))
        for x, y, _ in best_points[1:]:
            best_step_points.append((float(x), prev_y))
            best_step_points.append((float(x), y))
            prev_x, prev_y = x, y
    line = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in best_step_points)

    non_best_dots = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.5" fill="#cbd5e1" opacity="0.9"/>'
        for i, (x, y, _) in enumerate(points)
        if i not in best_idx_set
    )
    best_dots = "".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="#10b981" stroke="#fff" stroke-width="1.2"/>'
        for x, y, _ in best_points
    )
    best_labels = "".join(
        (
            f'<text x="{sx(x) + 6:.2f}" y="{sy(y) - 7:.2f}" '
            'font-size="9" fill="#065f46" transform="rotate(-24 '
            f'{sx(x) + 6:.2f} {sy(y) - 7:.2f})">'
            f"{html.escape((title or 'untitled')[:36])}</text>"
        )
        for x, y, title in best_points
    )
    y0 = sy(ymin)
    svg_border = "border:1px solid #e2e8f0;border-radius:6px"
    label_y_min = (
        f'<text x="{pad_l - 6}" y="{sy(ymin_raw):.1f}" text-anchor="end" '
        'dominant-baseline="middle" font-size="11" '
        f'fill="#64748b">{ymin_raw:.4g}</text>'
    )
    label_y_max = (
        f'<text x="{pad_l - 6}" y="{sy(ymax_raw):.1f}" text-anchor="end" '
        'dominant-baseline="middle" font-size="11" '
        f'fill="#64748b">{ymax_raw:.4g}</text>'
    )
    label_x0 = (
        f'<text x="{sx(points[0][0]):.1f}" y="{h - 12}" text-anchor="middle" '
        f'font-size="11" fill="#64748b">round {int(points[0][0])}</text>'
    )
    label_x1 = ""
    if len(points) > 1:
        label_x1 = (
            f'<text x="{sx(points[-1][0]):.1f}" y="{h - 12}" text-anchor="middle" '
            f'font-size="11" fill="#64748b">round {int(points[-1][0])}</text>'
        )
    polyline = (
        '<polyline fill="none" stroke="#10b981" stroke-width="2.2" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{line}"/>'
    )
    axis_line = (
        f'<line x1="{pad_l}" y1="{y0:.2f}" x2="{pad_l + gw:.2f}" y2="{y0:.2f}" stroke="#cbd5e1" stroke-width="1"/>'
    )
    y_axis_label = (
        f'<text transform="translate(14,{pad_t + gh / 2:.0f}) rotate(-90)" '
        'text-anchor="middle" font-size="11" fill="#64748b">val_bpb</text>'
    )
    exp_round_label = (
        f'<text x="{pad_l + gw / 2:.0f}" y="{h - 4}" text-anchor="middle" '
        'font-size="11" fill="#64748b">Experiment round</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="100%" style="max-width:{w}px;display:block;background:#f8fafc;{svg_border}">'
        "\n"
        f"{axis_line}\n"
        f"{polyline}\n"
        f"{non_best_dots}\n"
        f"{best_dots}\n"
        f"{best_labels}\n"
        f"{label_y_min}\n"
        f"{label_y_max}\n"
        f"{label_x0}\n"
        f"{label_x1}\n"
        f"{exp_round_label}\n"
        f"{y_axis_label}\n"
        f'<circle cx="{pad_l + 8}" cy="{pad_t + 8}" r="3" fill="#cbd5e1"/>\n'
        f'<text x="{pad_l + 16}" y="{pad_t + 11}" font-size="10" '
        'fill="#64748b">discarded</text>\n'
        f'<circle cx="{pad_l + 82}" cy="{pad_t + 8}" r="3.8" fill="#10b981"/>\n'
        f'<text x="{pad_l + 91}" y="{pad_t + 11}" font-size="10" '
        'fill="#065f46">kept (running best)</text>\n'
        "</svg>"
    )


def build_history_section_html(history: list[HistoryEntry]) -> str:
    """Line plot of val_bpb vs round plus a compact per-round table."""
    points: list[tuple[int, float, str | None]] = []
    rows: list[str] = []
    td = "border:1px solid #e2e8f0;padding:6px"
    for h in history:
        r = h.round
        val_cell = "—"
        status_html = "—"
        if h.metrics is not None:
            try:
                val = float(h.metrics.val_metric)
                val_cell = f"{val:.6g}"
                points.append((r, val, h.title))
                status_html = "ok"
            except (TypeError, ValueError):
                status_html = html.escape("invalid val_metric")
        elif h.oom:
            status_html = html.escape("OOM")
        elif h.error is not None:
            status_html = html.escape(str(h.error)[:200])

        rows.append(
            "<tr>"
            f"<td style='{td}'>{r}</td>"
            f"<td style='{td};font-variant-numeric:tabular-nums'>{val_cell}</td>"
            f"<td style='{td};max-width:28rem;word-break:break-word'>{status_html}</td>"
            "</tr>"
        )

    if points:
        chart = history_val_bpb_svg(points)
    else:
        chart = '<p style="color:#64748b;margin:0">No val_metric points in history (only failures or empty run).</p>'

    empty_row = "<tr><td colspan='3' style='padding:8px;color:#64748b'>No rounds.</td></tr>"
    table_inner = "".join(rows) if rows else empty_row
    table = (
        '<table style="border-collapse:collapse;font-size:14px;width:100%;'
        'max-width:48rem;margin-top:12px">'
        "<thead><tr style='background:#f1f5f9'>"
        f"<th style='{td};text-align:left'>Round</th>"
        f"<th style='{td};text-align:left'>val_bpb</th>"
        f"<th style='{td};text-align:left'>Status</th>"
        "</tr></thead><tbody>"
        f"{table_inner}"
        "</tbody></table>"
    )

    return (
        "\n<h2>History</h2>\n"
        '<h3 style="margin:0 0 10px;font-size:1rem;font-weight:600">'
        "val_bpb by round</h3>\n"
        f"{chart}\n"
        '<details style="margin-top:16px">'
        '<summary style="cursor:pointer">Per-round table</summary>\n'
        f"{table}\n"
        "</details>\n"
    )


def build_summary_html(
    user_intent: str,
    research_topic: str,
    literature: str,
    best: HistoryEntry | None,
    history: list[HistoryEntry],
) -> str:
    topic_display = research_topic or "(none — arXiv search skipped)"
    literature_display = literature or "(none — no literature retrieved)"
    history_section = build_history_section_html(history)
    best_json = json.dumps(dataclasses.asdict(best), indent=2) if best is not None else "none"
    best_pre = html.escape(best_json)
    pre_style = (
        "white-space:pre-wrap;word-break:break-word;background:#f8fafc;"
        "padding:12px;border-radius:6px;border:1px solid #e2e8f0"
    )
    body_style = "font-family:system-ui,sans-serif;line-height:1.45;max-width:52rem;margin:1rem auto;padding:0 12px"
    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8"><title>Autoresearch summary</title></head>\n'
        f'<body style="{body_style}">\n'
        "<h1>Autoresearch run</h1>\n"
        f"<p><b>User intent</b>: {html.escape(user_intent)}</p>\n"
        f"<p><b>Topic</b>: {html.escape(topic_display)}</p>\n"
        f"{history_section}"
        "<h2>Literature snippet</h2>\n"
        f'<pre style="{pre_style}">{html.escape(literature_display[:4000])}</pre>\n'
        "<h2>Best result</h2>\n"
        f'<pre style="{pre_style}">{best_pre}</pre>\n'
        "</body></html>\n"
    )
