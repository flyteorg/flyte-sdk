"""MLE Orchestrator Agent (interactive sandbox) — Builds orchestration code.

This is a port of ``mle_orchestrator_agent.py`` that swaps the Monty
orchestration sandbox (``flyte.sandbox.orchestrate_local``) for a live
``union.sandbox`` *interactive sandbox* session (``unionai-sandbox``). The agent
logic, LLM calls, reporting, and error-injection demonstrations are preserved;
only the sandbox execution constructs change.

Where the original dispatched to pre-defined Flyte *tasks* from inside the Monty
sandbox, this version provides the same tools as plain Python functions staged
into the sandbox work dir, and runs the LLM-generated orchestration code against
them inside the isolated session.

This agent demonstrates how to:
1. Use an LLM to generate orchestration code that calls pre-defined tools
2. Execute that code inside an isolated ``union.sandbox`` session
3. Iteratively fix errors by re-generating orchestration code

Prerequisites:
- Pin ``unionai-sandbox`` to the same version in the agent image and when deploying
  the sandbox-server (client/server interfaces must match)::

    pip install 'unionai-sandbox[deploy]==0.0.1b10'
    unionai-sandbox-deploy

Run remotely::

    # 1. Upload a CSV (or use an existing remote path)
    python -c "
    import asyncio
    from pathlib import Path
    import flyte, flyte.remote as remote
    async def main():
        flyte.init_from_config()
        _, uri = await remote.upload_file.aio(Path('data.csv'), fname='data.csv')
        print(uri)
    asyncio.run(main())
    "

    # 2. Launch the agent (builds image, submits run, follows logs)
    flyte run --follow examples/genai/mle_agent/mle_orchestrator_agent_interactive.py \\
        mle_orchestrator_agent \\
        --prompt "Build a pipeline that loads data, preprocesses it, trains models, and evaluates them." \\
        --data s3://.../data.csv \\
        --feature_columns '["feature1","feature2"]' \\
        --target_column target

Debug with Flyte MCP tools (``uvx --from 'flyte[mcp]' flyte-mcp``) or the CLI::

    flyte get run <run_id> --details
    flyte get action-logs <run_id> a0
"""

import inspect
import os
import re
from datetime import timedelta

from union import sandbox as sb

import flyte
import flyte.report
from flyte.io import File

agent_env = flyte.TaskEnvironment(
    "mle-orchestrator-interactive",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mle-orchestrator-interactive-image").with_pip_packages(
            "httpx", "unionai-sandbox[remote]==0.0.1b10"
        )
    ),
)

# Python packages the tools need; installed into the session venv at runtime.
TOOL_DEPENDENCIES = ["pandas", "scikit-learn", "numpy", "matplotlib"]


# ---------------------------------------------------------------------------
# Tools — plain Python functions (no Flyte task wrapping). These are both
# introspected to build the system prompt *and* serialized (via inspect) into a
# `tools.py` module staged inside the sandbox so the orchestration code can call
# them directly.
# ---------------------------------------------------------------------------


def load_data(data_path: str) -> list:
    """Load CSV data from a file path and return as a list of dicts.

    Args:
        data_path: Path to the CSV file

    Returns:
        List of row dictionaries
    """
    import pandas as pd

    with open(data_path, "rb") as f:
        df = pd.read_csv(f)
    return df.to_dict(orient="records")


def preprocess_data(data: list, feature_columns: list, target_column: str) -> dict:
    """Preprocess data by extracting features and target.

    Args:
        data: List of row dictionaries
        feature_columns: List of column names to use as features
        target_column: Name of the target column

    Returns:
        Dict with 'X' (features as list of lists) and 'y' (target as list)
    """
    X = [[row[col] for col in feature_columns] for row in data]
    y = [row[target_column] for row in data]
    return {"X": X, "y": y}


def train_linear_model(preprocessed: dict) -> dict:
    """Train a linear regression model.

    Args:
        preprocessed: Dict with 'X' and 'y' from preprocess_data

    Returns:
        Dict with 'coefficients', 'intercept', and 'r2_score'
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = preprocessed["X"]
    y = preprocessed["y"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    score = r2_score(y, predictions)

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2_score": float(score),
    }


def train_random_forest(preprocessed: dict, n_estimators: int = 100) -> dict:
    """Train a random forest regressor.

    Args:
        preprocessed: Dict with 'X' and 'y' from preprocess_data
        n_estimators: Number of trees in the forest

    Returns:
        Dict with 'feature_importances' and 'r2_score'
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    X = preprocessed["X"]
    y = preprocessed["y"]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    score = r2_score(y, predictions)

    return {
        "feature_importances": model.feature_importances_.tolist(),
        "r2_score": float(score),
    }


def evaluate_model(model_result: dict) -> str:
    """Evaluate a model and return a summary.

    Args:
        model_result: Dict containing model results (must have 'r2_score')

    Returns:
        Human-readable evaluation summary
    """
    r2 = model_result.get("r2_score", 0)
    if r2 > 0.9:
        quality = "excellent"
    elif r2 > 0.7:
        quality = "good"
    elif r2 > 0.5:
        quality = "moderate"
    else:
        quality = "poor"

    summary = f"Model R² score: {r2:.4f} ({quality} fit)"

    if "coefficients" in model_result:
        summary += f"\nLinear model coefficients: {model_result['coefficients']}"
        summary += f"\nIntercept: {model_result['intercept']:.4f}"
    elif "feature_importances" in model_result:
        summary += f"\nFeature importances: {model_result['feature_importances']}"

    return summary


def create_visualization_report(model_results: list) -> dict:
    """Create a visualization report comparing multiple model results.

    Writes an HTML report to ``report.html`` in the current working directory
    (the sandbox work dir). The caller pulls it out and publishes it.

    Args:
        model_results: List of model result dicts, each containing at minimum
            'r2_score' and optionally 'model_type', 'params', 'feature_importances',
            or 'coefficients'.

    Returns:
        Dict with 'best_model' info and 'num_models' compared
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(model_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison Report", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    model_names = []
    r2_scores = []
    for i, result in enumerate(model_results):
        model_type = result.get("model_type", f"Model {i + 1}")
        params = result.get("params", {})
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:2])
            name = f"{model_type}\n({param_str})"
        else:
            name = model_type
        model_names.append(name)
        r2_scores.append(result.get("r2_score", 0))

    colors = plt.cm.viridis([i / max(n_models - 1, 1) for i in range(n_models)])
    bars = ax1.bar(range(n_models), r2_scores, color=colors)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("R² Score")
    ax1.set_title("Model Performance Comparison")
    ax1.set_ylim(0, 1.1)
    for bar, score in zip(bars, r2_scores):
        ax1.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = axes[0, 1]
    rf_results = [r for r in model_results if "feature_importances" in r]
    if rf_results:
        best_rf = max(rf_results, key=lambda x: x.get("r2_score", 0))
        importances = best_rf["feature_importances"]
        feature_names = [f"Feature {i + 1}" for i in range(len(importances))]
        ax2.barh(feature_names, importances, color="steelblue")
        ax2.set_xlabel("Importance")
        ax2.set_title("Feature Importances (Best RF)")
    else:
        ax2.text(
            0.5,
            0.5,
            "No Random Forest\nmodels to display",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Feature Importances")

    ax3 = axes[1, 0]
    linear_results = [r for r in model_results if "coefficients" in r]
    if linear_results:
        best_linear = max(linear_results, key=lambda x: x.get("r2_score", 0))
        coeffs = best_linear["coefficients"]
        feature_names = [f"Feature {i + 1}" for i in range(len(coeffs))]
        colors_coef = ["green" if c >= 0 else "red" for c in coeffs]
        ax3.barh(feature_names, coeffs, color=colors_coef)
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Coefficient Value")
        ax3.set_title("Linear Model Coefficients (Best)")
    else:
        ax3.text(
            0.5, 0.5, "No Linear\nmodels to display", ha="center", va="center", transform=ax3.transAxes, fontsize=12
        )
        ax3.set_title("Linear Model Coefficients")

    ax4 = axes[1, 1]
    summary_text = "Model Summary\n" + "=" * 40 + "\n\n"
    sorted_results = sorted(model_results, key=lambda x: x.get("r2_score", 0), reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        model_type = result.get("model_type", f"Model {i + 1}")
        r2 = result.get("r2_score", 0)
        params = result.get("params", {})
        rank = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i + 1}"))
        summary_text += f"{rank} {model_type}: R²={r2:.4f}\n"
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            summary_text += f"   Params: {param_str}\n"
        summary_text += "\n"

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    ax4.axis("off")
    ax4.set_title("Rankings & Summary")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    best_model = sorted_results[0] if sorted_results else {}
    report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
        }}
        .stat {{
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Comparison Report</h1>
        <p>Comprehensive analysis of {n_models} trained models</p>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{n_models}</div>
                <div class="stat-label">Models Compared</div>
            </div>
            <div class="stat">
                <div class="stat-value">{best_model.get("r2_score", 0):.4f}</div>
                <div class="stat-label">Best R² Score</div>
            </div>
            <div class="stat">
                <div class="stat-value">{best_model.get("model_type", "N/A")}</div>
                <div class="stat-label">Best Model Type</div>
            </div>
        </div>
    </div>

    <div class="chart-container">
        <img src="data:image/png;base64,{img_base64}" alt="Model Comparison Charts">
    </div>
</body>
</html>
"""

    with open("report.html", "w") as f:
        f.write(report_html)

    return {
        "num_models": n_models,
        "best_model": best_model,
    }


TOOLS = [
    load_data,
    preprocess_data,
    train_linear_model,
    train_random_forest,
    evaluate_model,
    create_visualization_report,
]

# The tools, serialized to a module that gets staged into the sandbox work dir.
TOOLS_SOURCE = "\n\n".join(inspect.getsource(tool) for tool in TOOLS)


async def _call_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """Send a chat-completion request to Anthropic and return the text response."""
    import httpx

    api_key = os.environ["ANTHROPIC_API_KEY"]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["content"][0]["text"]


def _extract_code(text: str) -> str:
    """Extract Python code from markdown fences, or return the raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# Describes the interactive-sandbox restrictions to the LLM, replacing the
# Monty-specific ``flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT``.
SANDBOX_CONSTRAINTS_PROMPT = """\
CRITICAL — Interactive sandbox restrictions (union.sandbox):
- Your orchestration code runs inside an isolated, network-blocked sandbox session.
- The tools listed above are already imported and available; call them directly as functions.
- Network access is BLOCKED: do not make any outbound network calls.
- The filesystem is restricted to the sandbox work dir (the current working directory).
  Do not read or write any paths outside it.
- The last expression in your code is the return value."""


def _build_system_prompt(tools: list) -> str:
    """Build system prompt with tool signatures."""
    tool_docs = []
    for tool in tools:
        sig = inspect.signature(tool)
        doc = inspect.getdoc(tool) or "No documentation"
        tool_docs.append(f"- {tool.__name__}{sig}\n    {doc}")

    tools_section = "\n\n".join(tool_docs)

    restrictions = SANDBOX_CONSTRAINTS_PROMPT.replace("{", "{{").replace("}", "}}")

    return f"""\
You are an ML pipeline orchestrator. Write Python code to orchestrate ML tools.

Available tools (call these as functions):
{tools_section}

{restrictions}

IMPORTANT RULES:
- The last expression in your code is the return value
- You can define helper functions inside your code
- Call the tools as regular (synchronous) functions
- The input 'data_path' is the path to a CSV file that can be passed directly to load_data

Example orchestration code:
```python
# Load and preprocess data
raw_data = load_data(data_path)
preprocessed = preprocess_data(raw_data, feature_columns, target_column)

# Train model
model_result = train_linear_model(preprocessed)

# Evaluate
evaluate_model(model_result)
```

Return ONLY the Python code in a markdown code block.
"""


@flyte.trace
async def write_pipeline_code(
    prompt: str,
    tools: list,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Generate orchestration code using the LLM."""
    system = _build_system_prompt(tools)
    messages = [{"role": "user", "content": prompt}]
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


@flyte.trace
async def fix_pipeline_code(
    prompt: str,
    previous_code: str,
    error: str,
    tools: list,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Fix orchestration code based on an error."""
    system = _build_system_prompt(tools)
    user_content = f"""\
Previous orchestration code that failed:
```python
{previous_code}
```

Error encountered:
```
{error}
```

Original request: {prompt}

Please fix the orchestration code. Remember the interactive sandbox restrictions.
"""
    messages = [{"role": "user", "content": user_content}]
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


def _build_driver(code: str, feature_columns: list[str], target_column: str) -> str:
    """Wrap the orchestration code with tool imports and input bindings."""
    header = (
        "import os\n"
        "import sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from tools import (\n"
        "    load_data,\n"
        "    preprocess_data,\n"
        "    train_linear_model,\n"
        "    train_random_forest,\n"
        "    evaluate_model,\n"
        "    create_visualization_report,\n"
        ")\n\n"
        "data_path = 'data'\n"
        f"feature_columns = {feature_columns!r}\n"
        f"target_column = {target_column!r}\n\n"
        "# --- orchestration code ---\n"
    )
    return header + code + "\n"


async def _read_file_bytes(data: File) -> bytes:
    """Download the input file and return its raw bytes for staging into the sandbox."""
    local_path = await data.download()
    with open(local_path, "rb") as f:  # noqa: ASYNC230
        return f.read()


async def run_orchestration_in_sandbox(
    code: str,
    data_bytes: bytes,
    feature_columns: list[str],
    target_column: str,
) -> None:
    """Run LLM-generated orchestration code inside an interactive sandbox.

    Raises ``RuntimeError`` if the orchestration code (or dependency install)
    fails, mirroring the original ``orchestrate_local`` exception behaviour.
    """
    driver = _build_driver(code, feature_columns, target_column)

    # Session-level allow-list lets `uv pip install` reach PyPI; the
    # orchestration run() itself tightens to network_mode="blocked".
    async with await sb.session(
        network_mode="allowlist",
        network_allowlist=sb.PYPI_HOSTS,
        max_runtime_s=600,
        timeout=timedelta(minutes=15),
    ) as sbx:
        # Stage the input CSV and the tools module into the work dir.
        await sbx.put_bytes(f"{sbx.work_dir}/data", data_bytes)
        await sbx.put_bytes(f"{sbx.work_dir}/tools.py", TOOLS_SOURCE.encode())

        # Install the tools' dependencies into the session venv.
        proc = await sbx.run(
            f"uv pip install {' '.join(TOOL_DEPENDENCIES)}",
            stdout=True,
            stderr=True,
        )
        _out, err = await proc.communicate_text()
        if proc.returncode != 0:
            raise RuntimeError(f"Dependency install failed:\n{err}")

        # Run the orchestration code with the network blocked.
        proc = await sbx.run(
            driver,
            script_type="python",
            stdout=True,
            stderr=True,
            network_mode="blocked",
        )
        out, err = await proc.communicate_text()
        if proc.returncode != 0:
            raise RuntimeError(err or out)

        # If the orchestration produced a visualization report, publish it.
        try:
            report_bytes = await sbx.get_bytes(f"{sbx.work_dir}/report.html", max_bytes=20 * 1024 * 1024)
        except Exception:
            report_bytes = b""
        if report_bytes:
            viz_tab = flyte.report.get_tab("Visualization")
            viz_tab.replace(report_bytes.decode())
            await flyte.report.flush.aio()


async def _build_report(code: str, attempt: int, error: str | None = None) -> str:
    """Build an HTML report showing the orchestration code and status."""
    status_color = "red" if error else "green"
    status_text = f"Error: {error}" if error else "Success"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLE Orchestrator Agent - Attempt {attempt}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: {status_color};
            color: white;
            font-weight: bold;
        }}
        .code-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .code-section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        pre {{
            background: #f8f8f8;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }}
        .error-section {{
            background: #fff5f5;
            border: 1px solid #feb2b2;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }}
        .error-section h3 {{
            color: #c53030;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MLE Orchestrator Agent</h1>
        <p>Attempt: {attempt + 1}</p>
        <span class="status">{status_text}</span>
    </div>

    <div class="code-section">
        <h2>Generated Orchestration Code</h2>
        <pre><code class="language-python">{code}</code></pre>
    </div>

    {"<div class='error-section'><h3>Error Details</h3><pre>" + error + "</pre></div>" if error else ""}
</body>
</html>
"""


@agent_env.task(retries=3, report=True)
async def mle_orchestrator_agent(
    prompt: str,
    data: File,
    feature_columns: list[str],
    target_column: str,
    max_iter: int = 5,
) -> str:
    """MLE agent that builds orchestration code using pre-defined tools.

    This agent:
    1. Takes a user prompt describing what ML pipeline to build
    2. Generates orchestration code that calls pre-defined tools
    3. Executes the code inside an isolated union.sandbox session
    4. On failure, regenerates code and retries

    Args:
        prompt: Natural language description of the ML pipeline to build
        data: Input data file (CSV format expected)
        feature_columns: List of column names to use as features
        target_column: Name of the target column
        max_iter: Maximum number of iterations to try fixing errors

    Returns:
        The final working orchestration code that was executed
    """
    code = await write_pipeline_code(prompt, TOOLS)

    # Stage the input once; it's pushed into each fresh session's work dir.
    data_bytes = await _read_file_bytes(data)

    for attempt in range(1, max_iter + 1):
        tab = flyte.report.get_tab(f"Attempt {attempt}")

        if attempt == 1:
            # 🔥 python code error
            code = f"1234 / 0\n\n{code}\n"  # division by zero

        if attempt == 2:
            # 🔥 sandbox filesystem isolation: writing outside the work dir is denied
            code = f"open('/root/escape.txt', 'w').write('x')\n\n{code}\n"

        if attempt == 3:
            # 🔥 sandbox network isolation: outbound connections are blocked
            code = f"import urllib.request\nurllib.request.urlopen('https://www.google.com', timeout=5)\n\n{code}\n"

        try:
            tab.replace(await _build_report(code, attempt, error=None))
            await flyte.report.flush.aio()
            await run_orchestration_in_sandbox(
                code,
                data_bytes=data_bytes,
                feature_columns=feature_columns,
                target_column=target_column,
            )

            break
        except Exception as exc:
            error_msg = str(exc)
            tab.replace(await _build_report(code, attempt, error=error_msg))
            await flyte.report.flush.aio()

            if attempt < max_iter - 1:
                code = await fix_pipeline_code(
                    prompt=prompt,
                    previous_code=code,
                    error=error_msg,
                    tools=TOOLS,
                )
            else:
                raise RuntimeError(
                    f"Failed to generate working orchestration code after {max_iter} attempts. Last error: {error_msg}"
                ) from exc

    await flyte.report.replace.aio(await _build_report(code, attempt, error=None))
    await flyte.report.flush.aio()
    return code


if __name__ == "__main__":
    import tempfile

    flyte.init_from_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n")
        for i in range(100):
            f.write(f"{i},{i * 2},{i * 3 + 5}\n")
        data_path = f.name

    async def main():
        data_file = await File.from_local(data_path)
        run = flyte.run(
            mle_orchestrator_agent,
            prompt=(
                "Build a pipeline that loads data, preprocesses it, "
                "trains a linear model and random forest model, evaluates the models, "
                "visualizes all the results, and returns the evaluation of the best model."
            ),
            data=data_file,
            feature_columns=["feature1", "feature2"],
            target_column="target",
            max_iter=5,
        )
        print(f"View at: {run.url}")
        run.wait()
        print(f"Result: {run.outputs()}")

    import asyncio

    asyncio.run(main())
