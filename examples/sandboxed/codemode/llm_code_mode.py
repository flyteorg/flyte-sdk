"""LLM Code Mode — AI-Generated Data Analytics
================================================

Inspired by Cloudflare's "code mode": instead of an LLM making individual
tool calls one at a time (N round-trips), it writes a **complete program**
using available tools. The program runs in a Monty sandbox with tasks as
the only side-effects. This is Flyte's answer to MCP tool use — tasks
*are* the tools.

**Why code mode > tool use:**
- Single LLM call vs multi-turn tool-calling loop
- LLM writes loops, conditionals, aggregations (not expressible in single tool calls)
- Deterministic program execution
- Sandbox guarantees safety (no arbitrary I/O)

Architecture::

    User: "Analyze 2024 sales trends by region"
      │
      ▼
    analyze(request) ── @env.task(report=True)
      ├─ generate_code(request) ── @flyte.trace (calls Claude)
      ├─ run_local_sandbox(code, functions={...})
      │     ├─ fetch_data(dataset) ── @env.task
      │     └─ create_chart(...)   ── @env.task
      └─ flyte.report.replace(html) + flush()

Install dependencies::

    pip install 'flyte[sandboxed]' anthropic

Run::

    flyte run examples/sandboxed/llm_code_mode.py analyze \\
        --request "Show me monthly revenue trends for 2024, broken down by region"
"""

import html
import os
import re
import textwrap

import flyte
import flyte.report
import flyte.sandboxed

env = flyte.TaskEnvironment(
    name="llm-code-mode",
    secrets=[flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_debian_base().with_pip_packages("anthropic", "pydantic-monty"),
)

# ---------------------------------------------------------------------------
# Tool tasks (available inside the sandbox)
# ---------------------------------------------------------------------------

# Union golden palette for charts (based on union.ai brand gold #e69812)
CHART_COLORS = [
    "rgba(230, 152, 18, 0.8)",  # #e69812 — primary gold
    "rgba(242, 189, 82, 0.8)",  # #f2bd52 — lighter amber
    "rgba(184, 119, 10, 0.8)",  # #b8770a — darker bronze
    "rgba(250, 210, 130, 0.8)",  # #fad282 — honey
    "rgba(140, 90, 5, 0.8)",  # #8c5a05 — deep gold
]

CHART_BORDERS = [
    "#e69812",
    "#f2bd52",
    "#b8770a",
    "#fad282",
    "#8c5a05",
]


@env.task
def fetch_data(dataset: str) -> list:
    """Fetch tabular data by dataset name.

    Available datasets:
    - "sales_2024": monthly sales with columns month, region, revenue, units

    In production this would query a database or API.
    """
    if dataset == "sales_2024":
        regions = ["North", "South", "East", "West"]
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        rows = []
        # Deterministic mock data with realistic seasonal patterns
        base_revenue = {
            "North": 120000,
            "South": 95000,
            "East": 110000,
            "West": 105000,
        }
        seasonal = [0.85, 0.88, 0.95, 1.0, 1.05, 1.10, 1.08, 1.12, 1.06, 1.02, 1.15, 1.25]
        for i, month in enumerate(months):
            for region in regions:
                revenue = int(base_revenue[region] * seasonal[i])
                units = revenue // 45  # ~$45 per unit
                rows.append(
                    {
                        "month": month,
                        "region": region,
                        "revenue": revenue,
                        "units": units,
                    }
                )
        return rows
    return []


@env.task
def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    """Generate a self-contained Chart.js HTML snippet.

    Args:
        chart_type: One of "bar", "line", "pie", "doughnut".
        title: Chart title displayed above the canvas.
        labels: X-axis labels (or slice labels for pie/doughnut).
        values: Either a flat list of numbers, or a list of
                {"label": str, "data": list[number]} dicts for multi-series.

    Returns:
        HTML string with a <canvas> and <script> block.
    """
    import json as _json

    canvas_id = "chart-" + title.lower().replace(" ", "-").replace("/", "-")

    # Build datasets
    if values and isinstance(values[0], dict):
        # Multi-series: [{"label": "North", "data": [1, 2, ...]}, ...]
        datasets = []
        for i, series in enumerate(values):
            color_idx = i % len(CHART_COLORS)
            datasets.append(
                {
                    "label": series["label"],
                    "data": series["data"],
                    "backgroundColor": CHART_COLORS[color_idx],
                    "borderColor": CHART_BORDERS[color_idx],
                    "borderWidth": 2,
                    "tension": 0.3,
                    "fill": False,
                }
            )
    else:
        # Single series
        bg_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(values))]
        border_colors = [CHART_BORDERS[i % len(CHART_BORDERS)] for i in range(len(values))]
        datasets = [
            {
                "label": title,
                "data": values,
                "backgroundColor": bg_colors if chart_type in ("pie", "doughnut") else CHART_COLORS[0],
                "borderColor": border_colors if chart_type in ("pie", "doughnut") else CHART_BORDERS[0],
                "borderWidth": 2,
                "tension": 0.3,
                "fill": chart_type == "line",
            }
        ]

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": title, "font": {"size": 16}}},
        },
    }

    return (
        f'<div style="position:relative;height:350px;margin:20px 0;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f"<script>new Chart(document.getElementById('{canvas_id}'),"
        f"{_json.dumps(config)});</script>"
    )


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data analyst. Write Python code to analyze data and produce charts.

    Available functions:
    - fetch_data(dataset: str) -> list[dict]
        Returns rows as list of dicts. Available datasets:
        - "sales_2024": columns are month, region, revenue, units
          Months: Jan-Dec. Regions: North, South, East, West.

    - create_chart(chart_type: str, title: str, labels: list, values: list) -> str
        Creates a Chart.js chart. chart_type: "bar", "line", "pie", "doughnut".
        For multi-series, pass values as: [{"label": "Series A", "data": [1,2,3]}, ...]
        Returns an HTML string.

    CRITICAL — Sandbox syntax restrictions (Monty runtime):
    - No imports.
    - No subscript assignment: `d[key] = value` and `l[i] = value` are FORBIDDEN.
    - Reading subscripts is OK: `x = d[key]` and `x = l[i]` work fine.
    - Build lists with .append() and list literals, NOT by index assignment.
    - Build dicts ONLY as literals: {"k": v, ...}. Never mutate them after creation.
    - To aggregate data, use lists of tuples/dicts, not mutating a dict.
    - The last expression in your code must be the return value.
    - Return a dict: {"charts": [<html strings from create_chart>], "summary": "<text>"}

    Example — group sales by region (correct pattern):
        data = fetch_data("sales_2024")
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        regions = ["North", "South", "East", "West"]

        # Build per-region series using list comprehensions (NO dict mutation)
        series = []
        for region in regions:
            region_data = [row["revenue"] for row in data if row["region"] == region]
            series.append({"label": region, "data": region_data})

        chart1 = create_chart("line", "Revenue by Region", months, series)

        total = 0
        for row in data:
            total = total + row["revenue"]

        {"charts": [chart1], "summary": "Total 2024 revenue: $" + str(total)}
""")


@flyte.trace
async def generate_code(request: str) -> str:
    """Call Claude to generate analysis code for the given request."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request}],
    )

    # Extract code from markdown fences if present
    text = response.content[0].text
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@env.task(report=True)
async def analyze(request: str) -> str:
    """Analyze data using LLM-generated code in a sandboxed environment.

    1. Calls Claude to generate analysis code (traced, not a task)
    2. Executes the code in a Monty sandbox with fetch_data and create_chart as tools
    3. Renders an interactive HTML report with Chart.js visualizations
    """
    # Step 1: Generate analysis code via LLM
    code = await generate_code(request)

    # Step 2: Execute in sandbox with tasks as tools
    # Monty requires at least one declared input variable, even if unused
    result = await flyte.sandboxed.run_local_sandbox(
        code,
        inputs={"_unused": 0},
        functions={"fetch_data": fetch_data, "create_chart": create_chart},
    )

    # Step 3: Build HTML report
    charts_html = "\n".join(result.get("charts", []))
    summary = result.get("summary", "No summary generated.")
    escaped_code = html.escape(code)

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis: {html.escape(request)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1408 0%, #3d2e0f 50%, #1a1408 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2em;
            background: linear-gradient(90deg, #e69812, #f2bd52);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }}
        .summary {{
            background: rgba(230, 152, 18, 0.1);
            border-left: 4px solid #e69812;
            border-radius: 0 12px 12px 0;
            padding: 20px 24px;
            margin-bottom: 30px;
            line-height: 1.7;
            font-size: 1.05em;
        }}
        .charts {{ margin-bottom: 30px; }}
        details {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-top: 20px;
        }}
        summary {{
            cursor: pointer;
            font-weight: 600;
            color: #f2bd52;
            padding: 4px 0;
        }}
        pre {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.5;
            margin-top: 12px;
        }}
        code {{ color: #fad282; font-family: 'Fira Code', Consolas, monospace; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Analysis: {html.escape(request)}</h1>
        <div class="summary">{html.escape(summary)}</div>
        <div class="charts">{charts_html}</div>
        <details>
            <summary>Generated Code</summary>
            <pre><code>{escaped_code}</code></pre>
        </details>
        <div class="footer">
            Generated by LLM Code Mode &middot; Powered by Anthropic Claude &amp; Union
        </div>
    </div>
</body>
</html>"""

    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    return summary


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        analyze,
        request="Show me monthly revenue trends for 2024, broken down by region",
    )
    print(f"View at: {run.url}")
