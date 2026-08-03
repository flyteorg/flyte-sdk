"""Agent in CODE MODE — the LLM plans in Python, executed in the Monty sandbox.

This is the same :class:`flyte.ai.agents.Agent` used elsewhere, but with
``code_mode=True``. Instead of emitting one JSON tool call at a time, each turn
the model writes a short Python program. That program runs in the Monty sandbox
with the agent's tools exposed as plain functions, and the value of its last
expression is fed back as the observation for the next turn. The loop ends when
the model replies with plain text (no code block).

Why code mode? The model can express real control flow — loops, filtering,
aggregation, and parallel fan-out via ``flyte_map(...)`` — in a single step,
while the ``@env.task`` tools still dispatch durably on the cluster.

The tasks below have heterogeneous, data-parallel shapes:

- ``list_tickers``        — return the universe of symbols to analyze.
- ``fetch_prices``        — fetch a price series for one ticker (durable, per-ticker).
- ``moving_average``      — compute a trailing moving average for a series.
- ``render_report``       — visualize the final rankings as a Flyte report (bar chart).

A natural plan is: list the tickers, ``flyte_map`` ``fetch_prices`` across them in
parallel, compute moving averages, rank, and finally call ``render_report`` to
publish an HTML bar chart to the Flyte report UI — all as generated code. The
report is attached to the durable ``render_report`` task; open that task in the
UI to see the visualization.

Run::

    pip install 'flyte' litellm pydantic-monty
    python examples/agents/flyte_agent/code_mode_agent.py
"""

from __future__ import annotations

from typing import Any

import flyte
import flyte.report
from flyte.ai.agents import Agent

env = flyte.TaskEnvironment(
    name="code-mode-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm", "pydantic-monty"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


# ---------------------------------------------------------------------------
# Tools — durable Flyte tasks exposed to the sandbox as plain functions.
# ---------------------------------------------------------------------------


@env.task
async def list_tickers() -> list[str]:
    """Return the universe of ticker symbols available for analysis."""
    return ["AAA", "BBB", "CCC", "DDD", "EEE"]


@env.task
async def fetch_prices(ticker: str) -> list[float]:
    """Fetch a daily closing-price series for a single ticker.

    Args:
        ticker: The ticker symbol (e.g. "AAA").

    Returns:
        A list of daily closing prices, oldest first.
    """
    # Deterministic pseudo-series so the example is reproducible offline.
    seed = sum(ord(c) for c in ticker)
    return [round(100.0 + (seed % 17) * 0.5 + i * ((seed % 5) - 2) * 0.3, 2) for i in range(30)]


@env.task
async def moving_average(prices: list[float], window: int = 5) -> float:
    """Compute the trailing moving average of a price series.

    Args:
        prices: A list of prices, oldest first.
        window: Number of trailing points to average.

    Returns:
        The mean of the last ``window`` prices (or all prices if fewer).
    """
    if not prices:
        return 0.0
    tail = prices[-window:]
    return round(sum(tail) / len(tail), 4)


def _bar_chart_html(title: str, rankings: list[dict[str, Any]]) -> str:
    """Build a self-contained HTML horizontal bar chart (no external deps)."""
    normalized: list[dict[str, Any]] = [
        {"ticker": str(r.get("ticker", "?")), "value": float(r.get("value", 0.0))} for r in rankings
    ]
    rows = sorted(normalized, key=lambda r: r["value"], reverse=True)
    max_value = max((r["value"] for r in rows), default=0.0) or 1.0

    bars = []
    for r in rows:
        width = max(2.0, r["value"] / max_value * 100.0)
        bars.append(
            f'<div style="display:flex;align-items:center;margin:6px 0;font-family:system-ui,sans-serif;">'
            f'<span style="width:64px;font-weight:600;">{r["ticker"]}</span>'
            f'<span style="flex:1;background:#eef1f6;border-radius:4px;overflow:hidden;">'
            f'<span style="display:block;width:{width:.1f}%;background:#4f8cff;color:#fff;'
            f'padding:4px 8px;border-radius:4px;white-space:nowrap;">{r["value"]:.2f}</span>'
            f"</span></div>"
        )

    return (
        f'<div style="max-width:640px;font-family:system-ui,sans-serif;">'
        f"<h2>{title}</h2>"
        f'<p style="color:#666;">{len(rows)} tickers, ranked by value (highest first).</p>'
        f"{''.join(bars)}"
        f"</div>"
    )


@env.task(report=True)
async def render_report(rankings: list[dict], title: str = "Ticker rankings") -> str:
    """Visualize the final rankings as a bar chart in the Flyte report UI.

    Call this once at the very end, after you have computed and sorted the
    rankings, to publish the result.

    Args:
        rankings: A list of ``{"ticker": str, "value": float}`` dicts.
        title: A heading shown above the chart.

    Returns:
        A short confirmation string describing what was published.
    """
    flyte.report.get_tab("Rankings").log(_bar_chart_html(title, rankings))
    await flyte.report.flush.aio()
    return f"Published a bar-chart report titled {title!r} with {len(rankings)} tickers."


INSTRUCTIONS = """\
You are a quantitative analyst agent. You answer questions about a small universe
of tickers by orchestrating the available functions in Python.

Typical plan:
1. Call list_tickers() to get the symbols.
2. Use flyte_map("fetch_prices", tickers) to fetch every price series in parallel.
3. Compute a trailing moving average per ticker with moving_average(prices, window).
4. Rank the tickers (highest first) as a list of {"ticker": ..., "value": ...} dicts.
5. Call render_report(rankings, title=...) to visualize the result as a Flyte report.
6. Then reply in plain text with the top tickers and their values.

Keep intermediate results as plain lists/dicts of primitives.
"""

agent = Agent(
    name="code-mode-agent",
    instructions=INSTRUCTIONS,
    model="claude-sonnet-4-6",
    tools=[list_tickers, fetch_prices, moving_average, render_report],
    code_mode=True,
    max_turns=10,
)


@env.task(report=True)
async def run_code_mode_agent(request: str) -> str:
    """Drive the code-mode agent inside a durable Flyte task."""
    result = await agent.run.aio(request, memory=[])
    return result.summary or result.error or ""


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        run_code_mode_agent,
        request=(
            "Rank all available tickers by their 5-day trailing moving average, "
            "highest first, visualize the rankings as a Flyte report, and tell me "
            "the top 3 with their values. Make sure the render the report at the end."
        ),
    )
    print(f"Run URL: {run.url}")
    run.wait()
    print(run.outputs()[0])
