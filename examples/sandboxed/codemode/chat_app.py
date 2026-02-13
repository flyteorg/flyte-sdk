"""Chat Analytics Agent — Code Mode as a flyte.app
===================================================

A persistent chat UI served as a ``flyte.app`` (FastAPI) where users
interactively ask data-analysis questions and get back Chart.js
visualizations + text summaries.

The core pattern is identical to ``llm_code_mode.py``: single LLM call
-> Python code string -> Monty sandbox execution with tool functions
-> charts + summary.  The difference is that this example wraps it in a
conversational web interface instead of a one-shot ``flyte.run``.

Architecture::

    Browser (Chat UI)
      |
      +-- GET /           -> Embedded HTML/CSS/JS chat interface
      +-- GET /api/tools  -> JSON list of available tool descriptions
      +-- POST /api/chat  -> { message, history } -> { code, charts, summary, error }
             |
             +-- generate_code(message, history)  -- AsyncAnthropic call
             +-- run_local_sandbox(code, functions={fetch_data, create_chart})
                    +-- fetch_data(dataset)    -- plain function
                    +-- create_chart(...)      -- plain function returning Chart.js HTML

Install dependencies::

    pip install 'flyte[sandboxed]' anthropic

Run::

    python examples/sandboxed/codemode/chat_app.py
"""

import os
import pathlib
import re
import textwrap

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import flyte
import flyte.sandboxed
from flyte.app.extras import FastAPIAppEnvironment

# ---------------------------------------------------------------------------
# FastAPI + AppEnvironment setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Chat Data Analytics Agent")

env = FastAPIAppEnvironment(
    name="chat-analytics-agent",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic-monty",
    ),
    secrets=flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    requires_auth=False,
)

# ---------------------------------------------------------------------------
# Tool functions — plain functions (the sandbox bridge handles callables)
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
                units = revenue // 45
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

    if values and isinstance(values[0], dict):
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


# Tool metadata exposed via /api/tools
TOOL_DESCRIPTIONS = [
    {
        "name": "fetch_data",
        "signature": "fetch_data(dataset: str) -> list[dict]",
        "description": (
            "Fetch tabular data by dataset name. Available datasets: "
            '"sales_2024" (columns: month, region, revenue, units).'
        ),
    },
    {
        "name": "create_chart",
        "signature": "create_chart(chart_type: str, title: str, labels: list, values: list) -> str",
        "description": (
            "Generate a Chart.js HTML snippet. chart_type: bar, line, pie, doughnut. "
            'For multi-series pass values as [{"label": "Series", "data": [...]}].'
        ),
    },
]

# ---------------------------------------------------------------------------
# System prompt + code generation
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


async def generate_code(message: str, history: list[dict]) -> str:
    """Call Claude to generate analysis code, with conversation history for context.

    Uses httpx directly instead of the Anthropic SDK to avoid a signal-handling
    conflict between the SDK and ``flyte serve --local`` (which blocks the main
    thread on ``signal.pause()``).
    """
    import httpx

    api_key = os.environ["ANTHROPIC_API_KEY"]
    messages = [*history, {"role": "user", "content": message}]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["content"][0]["text"]
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    code: str = ""
    charts: list[str] = []
    summary: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/tools")
async def get_tools() -> list[dict]:
    """Return JSON descriptions of available tool functions."""
    return TOOL_DESCRIPTIONS


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Core endpoint: generate code, run in sandbox, return results."""
    try:
        code = await generate_code(req.message, req.history)
    except Exception as exc:
        return ChatResponse(error=f"Code generation failed: {exc}")

    try:
        result = await flyte.sandboxed.run_local_sandbox(
            code,
            inputs={"_unused": 0},
            functions={"fetch_data": fetch_data, "create_chart": create_chart},
        )
    except Exception as exc:
        return ChatResponse(code=code, error=f"Sandbox execution failed: {exc}")

    charts = result.get("charts", []) if isinstance(result, dict) else []
    summary = result.get("summary", "No summary generated.") if isinstance(result, dict) else str(result)

    return ChatResponse(code=code, charts=charts, summary=summary)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the embedded chat UI."""
    return HTMLResponse(content=CHAT_HTML)


# ---------------------------------------------------------------------------
# Embedded chat UI (HTML / CSS / JS)
# ---------------------------------------------------------------------------

CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Analytics Agent</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1408;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
        }

        /* --- Left sidebar: tool cards --- */
        .sidebar {
            width: 280px;
            min-width: 280px;
            background: #231c0e;
            border-right: 1px solid rgba(230, 152, 18, 0.2);
            display: flex;
            flex-direction: column;
            padding: 20px 16px;
            overflow-y: auto;
        }
        .sidebar h2 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #e69812;
            margin-bottom: 16px;
        }
        .tool-card {
            background: rgba(230, 152, 18, 0.06);
            border: 1px solid rgba(230, 152, 18, 0.15);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .tool-card h3 {
            font-size: 14px;
            color: #f2bd52;
            margin-bottom: 4px;
        }
        .tool-card .sig {
            font-family: 'Fira Code', Consolas, monospace;
            font-size: 11px;
            color: #fad282;
            margin-bottom: 8px;
            word-break: break-all;
        }
        .tool-card p {
            font-size: 12px;
            color: #aaa;
            line-height: 1.5;
        }

        /* --- Main chat area --- */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .header {
            padding: 16px 24px;
            border-bottom: 1px solid rgba(230, 152, 18, 0.15);
            background: #1f170a;
        }
        .header h1 {
            font-size: 20px;
            background: linear-gradient(90deg, #e69812, #f2bd52);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        /* --- Message bubbles --- */
        .msg {
            max-width: 85%;
            margin-bottom: 20px;
            animation: fadeIn 0.2s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

        .msg.user {
            margin-left: auto;
            text-align: right;
        }
        .msg.user .bubble {
            display: inline-block;
            background: rgba(230, 152, 18, 0.15);
            border: 1px solid rgba(230, 152, 18, 0.3);
            border-radius: 14px 14px 4px 14px;
            padding: 12px 16px;
            text-align: left;
        }

        .msg.assistant .bubble {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px 14px 14px 4px;
            padding: 16px;
        }

        /* Code block inside assistant bubble */
        .msg.assistant details {
            margin-top: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
        }
        .msg.assistant details summary {
            cursor: pointer;
            font-weight: 600;
            color: #f2bd52;
            font-size: 13px;
        }
        .msg.assistant pre {
            margin-top: 8px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.5;
            color: #fad282;
            font-family: 'Fira Code', Consolas, monospace;
        }

        /* Chart container */
        .chart-container {
            margin-top: 14px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 12px;
        }

        /* Summary text */
        .summary-text {
            margin-top: 12px;
            background: rgba(230, 152, 18, 0.08);
            border-left: 3px solid #e69812;
            border-radius: 0 8px 8px 0;
            padding: 12px 14px;
            line-height: 1.6;
            font-size: 14px;
        }

        /* Error box */
        .error-box {
            margin-top: 12px;
            background: rgba(220, 53, 53, 0.12);
            border-left: 3px solid #dc3535;
            border-radius: 0 8px 8px 0;
            padding: 12px 14px;
            color: #ff8888;
            font-size: 13px;
        }

        /* --- Input bar --- */
        .input-bar {
            padding: 16px 24px;
            border-top: 1px solid rgba(230, 152, 18, 0.15);
            background: #1f170a;
            display: flex;
            gap: 12px;
        }
        .input-bar input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid rgba(230, 152, 18, 0.25);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.05);
            color: #e0e0e0;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-bar input:focus {
            border-color: #e69812;
        }
        .input-bar input::placeholder { color: #666; }
        .input-bar button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #e69812, #b8770a);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .input-bar button:hover { opacity: 0.9; }
        .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Spinner */
        .typing { color: #999; font-style: italic; font-size: 13px; margin-bottom: 16px; }
    </style>
</head>
<body>

<div class="sidebar">
    <h2>Available Tools</h2>
    <div id="toolCards"><p style="color:#666;font-size:13px;">Loading...</p></div>
</div>

<div class="main">
    <div class="header">
        <h1>Chat Analytics Agent</h1>
    </div>

    <div class="messages" id="messages"></div>

    <div class="input-bar">
        <input type="text" id="userInput"
               placeholder="Ask a data analysis question..."
               autocomplete="off" />
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const messagesDiv = document.getElementById('messages');
const userInput   = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');

// Conversation history sent to the server (text only, no chart HTML)
let history = [];

// ---- Load tool cards from /api/tools ----
(async () => {
    try {
        const resp = await fetch('/api/tools');
        const tools = await resp.json();
        const container = document.getElementById('toolCards');
        container.innerHTML = '';
        tools.forEach(t => {
            const card = document.createElement('div');
            card.className = 'tool-card';
            card.innerHTML =
                '<h3>' + escapeHtml(t.name) + '</h3>' +
                '<div class="sig">' + escapeHtml(t.signature) + '</div>' +
                '<p>' + escapeHtml(t.description) + '</p>';
            container.appendChild(card);
        });
    } catch(e) {
        document.getElementById('toolCards').innerHTML =
            '<p style="color:#ff8888;font-size:13px;">Failed to load tools</p>';
    }
})();

// ---- Send message ----
const PROGRESS_PHASES = [
    'Generating analysis code...',
    'Running code in sandbox...',
    'Building charts...',
];

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendUser(text);
    userInput.value = '';
    sendBtn.disabled = true;

    // Animated progress indicator that cycles through phases
    const statusEl = document.createElement('div');
    statusEl.className = 'typing';
    statusEl.textContent = PROGRESS_PHASES[0];
    messagesDiv.appendChild(statusEl);
    scrollBottom();

    let phase = 0;
    const progressTimer = setInterval(() => {
        phase = Math.min(phase + 1, PROGRESS_PHASES.length - 1);
        statusEl.textContent = PROGRESS_PHASES[phase];
    }, 3000);

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: text, history: history }),
        });
        const data = await resp.json();

        clearInterval(progressTimer);
        statusEl.remove();
        appendAssistant(data);

        // Update history with text-only entries
        history.push({ role: 'user', content: text });
        const assistantContent = data.summary || data.error || '';
        if (assistantContent) {
            history.push({ role: 'assistant', content: assistantContent });
        }
    } catch(e) {
        clearInterval(progressTimer);
        statusEl.remove();
        appendAssistant({ error: 'Request failed: ' + e.message });
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// ---- Render helpers ----
function appendUser(text) {
    const msg = document.createElement('div');
    msg.className = 'msg user';
    msg.innerHTML = '<div class="bubble">' + escapeHtml(text) + '</div>';
    messagesDiv.appendChild(msg);
    scrollBottom();
}

function appendAssistant(data) {
    const msg = document.createElement('div');
    msg.className = 'msg assistant';

    let html = '<div class="bubble">';

    // Collapsible code
    if (data.code) {
        html += '<details><summary>Generated Code</summary>'
              + '<pre>' + escapeHtml(data.code) + '</pre></details>';
    }

    // Charts
    if (data.charts && data.charts.length) {
        data.charts.forEach(chartHtml => {
            html += '<div class="chart-container">' + chartHtml + '</div>';
        });
    }

    // Summary
    if (data.summary) {
        html += '<div class="summary-text">' + escapeHtml(data.summary) + '</div>';
    }

    // Error
    if (data.error) {
        html += '<div class="error-box">' + escapeHtml(data.error) + '</div>';
    }

    html += '</div>';
    msg.innerHTML = html;
    messagesDiv.appendChild(msg);

    // Re-execute <script> tags so Chart.js renders
    executeScripts(msg);
    scrollBottom();
}

function executeScripts(container) {
    container.querySelectorAll('script').forEach(old => {
        const s = document.createElement('script');
        s.textContent = old.textContent;
        old.parentNode.replaceChild(s, old);
    });
}

function scrollBottom() {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// Enter key sends message
userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
        e.preventDefault();
        sendMessage();
    }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env)
    d = deployments[0]
    print(f"Deployed Chat Analytics Agent: {d.table_repr()}")
