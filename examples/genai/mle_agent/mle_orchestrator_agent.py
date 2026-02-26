"""MLE Orchestrator Agent — Builds orchestration code using pre-defined tools.

This agent demonstrates how to:
1. Use an LLM to generate orchestration code that calls pre-defined tools
2. Execute that code using the Monty sandbox (orchestrator_from_str)
3. Iteratively fix errors by re-generating orchestration code

The agent takes a user prompt, data, and max_iter budget and outputs
the orchestration code that fulfills the prompt.
"""

import os
import re

import flyte
import flyte.report
import flyte.sandbox
from flyte.io import File

tool_env = flyte.TaskEnvironment(
    "mle-tools",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=(
        flyte.Image.from_debian_base(name="mle-tools-image")
        .with_pip_packages("pandas", "scikit-learn", "numpy", "matplotlib")
    ),
)

agent_env = flyte.TaskEnvironment(
    "mle-orchestrator",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="niels-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mle-orchestrator-image")
        .with_pip_packages("httpx", "pydantic-monty")
    ),
    depends_on=[tool_env],
)


@tool_env.task
async def load_data(data_path: File) -> list:
    """Load CSV data from a file path and return as a list of dicts.

    Args:
        data_path: Path to the CSV file

    Returns:
        List of row dictionaries
    """
    import pandas as pd

    with open(await data_path.download(), "rb") as f:
        df = pd.read_csv(f)
    return df.to_dict(orient="records")


@tool_env.task
async def preprocess_data(data: list, feature_columns: list, target_column: str) -> dict:
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


@tool_env.task
async def train_linear_model(preprocessed: dict) -> dict:
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


@tool_env.task
async def train_random_forest(preprocessed: dict, n_estimators: int = 100) -> dict:
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


@tool_env.task
async def evaluate_model(model_result: dict) -> str:
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


@tool_env.task(report=True)
async def create_visualization_report(model_results: list) -> dict:
    """Create a visualization report comparing multiple model results.

    Args:
        model_results: List of model result dicts, each containing at minimum
            'r2_score' and optionally 'model_type', 'params', 'feature_importances',
            or 'coefficients'.

    Returns:
        Dict with 'best_model' info and 'num_models' compared
    """
    import base64
    import io
    import matplotlib.pyplot as plt

    n_models = len(model_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison Report", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    model_names = []
    r2_scores = []
    for i, result in enumerate(model_results):
        model_type = result.get("model_type", f"Model {i+1}")
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
        ax1.annotate(f"{score:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)

    ax2 = axes[0, 1]
    rf_results = [r for r in model_results if "feature_importances" in r]
    if rf_results:
        best_rf = max(rf_results, key=lambda x: x.get("r2_score", 0))
        importances = best_rf["feature_importances"]
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
        ax2.barh(feature_names, importances, color="steelblue")
        ax2.set_xlabel("Importance")
        ax2.set_title("Feature Importances (Best RF)")
    else:
        ax2.text(0.5, 0.5, "No Random Forest\nmodels to display",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Feature Importances")

    ax3 = axes[1, 0]
    linear_results = [r for r in model_results if "coefficients" in r]
    if linear_results:
        best_linear = max(linear_results, key=lambda x: x.get("r2_score", 0))
        coeffs = best_linear["coefficients"]
        feature_names = [f"Feature {i+1}" for i in range(len(coeffs))]
        colors_coef = ["green" if c >= 0 else "red" for c in coeffs]
        ax3.barh(feature_names, coeffs, color=colors_coef)
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Coefficient Value")
        ax3.set_title("Linear Model Coefficients (Best)")
    else:
        ax3.text(0.5, 0.5, "No Linear\nmodels to display",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Linear Model Coefficients")

    ax4 = axes[1, 1]
    summary_text = "Model Summary\n" + "=" * 40 + "\n\n"
    sorted_results = sorted(model_results, key=lambda x: x.get("r2_score", 0), reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        model_type = result.get("model_type", f"Model {i+1}")
        r2 = result.get("r2_score", 0)
        params = result.get("params", {})
        rank = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
        summary_text += f"{rank} {model_type}: R²={r2:.4f}\n"
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            summary_text += f"   Params: {param_str}\n"
        summary_text += "\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
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
                <div class="stat-value">{best_model.get('r2_score', 0):.4f}</div>
                <div class="stat-label">Best R² Score</div>
            </div>
            <div class="stat">
                <div class="stat-value">{best_model.get('model_type', 'N/A')}</div>
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

    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    return {
        "num_models": n_models,
        "best_model": best_model,
    }


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


def _build_system_prompt(tools: list) -> str:
    """Build system prompt with tool signatures."""
    import inspect

    tool_docs = []
    for tool in tools:
        func = tool.func if hasattr(tool, "func") else tool
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No documentation"
        tool_docs.append(f"- {func.__name__}{sig}\n    {doc}")

    tools_section = "\n\n".join(tool_docs)

    restrictions = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT.replace("{", "{{").replace("}", "}}")

    return f"""\
You are an ML pipeline orchestrator. Write Python code to orchestrate ML tools.

Available tools (call these as functions):
{tools_section}

{restrictions}

IMPORTANT RULES:
- The last expression in your code is the return value
- You can define helper functions inside your code
- All tool calls are async but you call them like regular functions
- The input 'data_path' is a File object (flyte.io.File) that can be passed directly to load_data

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
    model: str = "claude-sonnet-4-20250514",
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
    model: str = "claude-sonnet-4-20250514",
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

Please fix the orchestration code. Remember the Monty sandbox restrictions.
"""
    messages = [{"role": "user", "content": user_content}]
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


TOOLS = [
    load_data,
    preprocess_data,
    train_linear_model,
    train_random_forest,
    evaluate_model,
    create_visualization_report,
]


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
    3. Executes the code using the Monty sandbox
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

    for attempt in range(1, max_iter + 1):
        tab = flyte.report.get_tab(f"Attempt {attempt}")

        if attempt == 1:
            # 🔥 python code error
            code = f"1234 / 0\n\n{code}\n"  # division by zero

        if attempt == 2:
            # 🔥 monty sandbox error: try to use imports
            code = f"import os\nos._exit(1)\n\n{code}\n"  # exit with error

        if attempt == 3:
            # 🔥 monty sandbox error: try to use network
            code = f"import httpx\nhttpx.get('https://www.google.com')\n\n{code}\n"  # network access
        
        try:
            tab.replace(await _build_report(code, attempt, error=None))
            await flyte.report.flush.aio()
            await flyte.sandbox.orchestrate_local(
                code,
                inputs={
                    "data_path": data,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                },
                tasks=TOOLS,
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
                    f"Failed to generate working orchestration code after {max_iter} attempts. "
                    f"Last error: {error_msg}"
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
            f.write(f"{i},{i*2},{i*3 + 5}\n")
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
