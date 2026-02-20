"""Durable Analytics Agent — Code Mode via ``flyte.run``
=======================================================

Same ``CodeModeAgent`` and tool definitions as the chat app, but each tool
is wrapped with ``@env.task`` so the sandbox dispatches them as durable Flyte
tasks through the controller (rather than calling them directly in-process).

Architecture::

    User: "Analyze 2024 sales trends by region"
      |
      v
    analyze(request) -- @env.task(report=True)
      +-- CodeModeAgent.run(request, [])
             +-- LLM call (generate code)
             +-- run_local_sandbox(code, functions=durable_tools)
                    +-- fetch_data(dataset) -- @env.task (durable)
                    +-- create_chart(...)   -- @env.task (durable)
      +-- flyte.report.replace(html) + flush()

Run::

    flyte run examples/sandbox/codemode/durable_agent.py analyze \\
        --request "Show me monthly revenue trends for 2024, broken down by region"
"""

import html as _html

import _tools
from _agent import CodeModeAgent
from _tools import ALL_TOOLS

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="llm-code-mode",
    secrets=[flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_debian_base().with_pip_packages("httpx", "pydantic-monty", "unionai-reuse"),
    reusable=flyte.ReusePolicy(
        replicas=1,
        concurrency=10,
    ),
)

# ---------------------------------------------------------------------------
# Durable tool tasks — module-level declarations so the task runner can
# resolve them by name.  Each delegates to the plain function in _tools.py.
# ---------------------------------------------------------------------------


@env.task
async def fetch_data(dataset: str) -> list:
    return _tools.fetch_data(dataset)


@env.task
async def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    return _tools.create_chart(chart_type, title, labels, values)


@env.task
async def calculate_statistics(data: list, column: str) -> dict:
    return _tools.calculate_statistics(data, column)


@env.task
async def filter_data(data: list, column: str, operator: str, value: object) -> list:
    return _tools.filter_data(data, column, operator, value)


@env.task
async def group_and_aggregate(data: list, group_by: str, agg_column: str, agg_func: str) -> list:
    return _tools.group_and_aggregate(data, group_by, agg_column, agg_func)


@env.task
async def sort_data(data: list, column: str, descending: bool = False) -> list:
    return _tools.sort_data(data, column, descending)


# Build execution tools dict from the module-level task declarations
durable_tools = {name: globals()[name] for name in ALL_TOOLS}

# Plain tools for prompt generation, task-wrapped for sandbox execution
agent = CodeModeAgent(tools=ALL_TOOLS, execution_tools=durable_tools)


@env.task(report=True)
async def analyze(request: str) -> str:
    """Analyze data using LLM-generated code in a sandboxed environment.

    1. Calls Claude to generate analysis code
    2. Executes the code in a Monty sandbox with durable task tools
    3. Renders an interactive HTML report with Chart.js visualizations
    """
    result = await agent.run(request, [])

    # Build HTML report
    charts_html = "\n".join(result.charts)
    summary = result.summary or "No summary generated."
    escaped_code = _html.escape(result.code)

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis: {_html.escape(request)}</title>
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
        <h1>Analysis: {_html.escape(request)}</h1>
        <div class="summary">{_html.escape(summary)}</div>
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
