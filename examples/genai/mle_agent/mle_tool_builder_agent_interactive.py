"""MLE Tool Builder Agent (interactive sandbox) — Builds its own tools.

This is a port of ``mle_tool_builder_agent.py`` that swaps the one-shot
``flyte.sandbox.create()`` container for a live ``union.sandbox`` *interactive
sandbox* session (``unionai-sandbox``). The agent logic, LLM calls, reporting,
and error-injection demonstrations are preserved verbatim; only the sandbox
execution constructs change.

This agent demonstrates how to:
1. Use an LLM to generate Python code that processes data
2. Execute that code in a live, isolated ``union.sandbox`` session
3. Iteratively fix errors by re-generating code with tests

The agent takes a user prompt, data, and max_iter budget and outputs
the code that fulfills the prompt.

Prerequisites:
- Pin ``unionai-sandbox`` to the same version in the agent image and when deploying
  the sandbox-server (client/server interfaces must match)::

    pip install 'unionai-sandbox[deploy]==0.0.1b10'
    unionai-sandbox-deploy

Run remotely::

    flyte run --follow examples/genai/mle_agent/mle_tool_builder_agent_interactive.py \\
        mle_tool_builder_agent \\
        --prompt "Train a linear regression model to predict target from feature1 and feature2." \\
        --data s3://.../data.csv

Debug with Flyte MCP tools (``uvx --from 'flyte[mcp]' flyte-mcp``) or the CLI::

    flyte get run <run_id> --details
    flyte get action-logs <run_id> a0
"""

import json
import os
import re
from dataclasses import asdict
from datetime import timedelta

from union import sandbox as sb

import flyte
import flyte.report
from flyte.io import File

agent_env = flyte.TaskEnvironment(
    "mle-tool-builder-interactive",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mle-tool-builder-interactive-image").with_pip_packages(
            "httpx", "unionai-sandbox[remote,deploy]==0.0.1b10"
        )
    ),
)


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


def _extract_resources(text: str) -> str:
    """Extract resources from text."""
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "{}"


def _extract_python_dependencies(text: str) -> list[str]:
    """Extract Python dependencies from text."""
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    return []


SYSTEM_PROMPT = """\
You are an expert ML engineer. Write Python code to process data and train models.

IMPORTANT: Your code will run in an isolated interactive sandbox with the following constraints:
- The current working directory is the sandbox's persistent work dir.
- The input data is available as a CSV file named `data` in the current working directory.
- You must save your output to a file named `model` in the current working directory.
- You have access to: pandas, numpy, scikit-learn
- You cannot use any network calls or external APIs

Your code should:
1. Load and process the input CSV from `data`
2. Train a model or perform the requested analysis
3. Save the result to `model` (use pickle, joblib, write text/json, save plot images, etc.)

Example structure:
```python
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data")
X = data[['feature1', 'feature2']]
y = data['target']

model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model', 'wb') as f:
    pickle.dump(model, f)
```
"""

SYSTEM_PROMPT_PROVISION_SANDBOX_RESOURCES = """\
You are an expert ML engineer. Provision sandbox resources for this dataset.

IMPORTANT: Your code will run in an isolated interactive sandbox with the following constraints:
- The input data is available as a CSV file named `data` in the current working directory.
- You must save your output to a file named `model` in the current working directory.
- You have access to: pandas, numpy, scikit-learn
- You cannot use any network calls or external APIs

The output should be a json string in the format of:
```json
{{
    "cpu": int,  # number of CPUs
    "memory": str (e.g. "10Mi", "1Gi", "2Gi", etc.)
}}
```
"""

SYSTEM_PROMPT_GET_PYTHON_DEPENDENCIES = """\
You are an expert ML engineer. Get Python dependencies from this code. Make sure
to use your knowledge of the actual python package names, since that might be
different from the import names. Ignore built-in Python packages, only include
third-party packages on PyPI.

The output should be a list of Python dependencies in the format of:
```json
[
    "pandas",
    "numpy",
    "scikit-learn",
]
```
"""


@flyte.trace
async def write_code(
    prompt: str,
    previous_code: str = "",
    error: str = "",
    model: str = "claude-haiku-4-5",
) -> str:
    """Generate code using the LLM."""
    messages = [{"role": "user", "content": prompt}]
    if previous_code:
        messages.append({"role": "user", "content": f"Previous code that failed: {previous_code}"})
    if error:
        messages.append({"role": "user", "content": f"Error encountered: {error}"})
    raw = await _call_llm(model, SYSTEM_PROMPT, messages)
    return _extract_code(raw)


@flyte.trace
async def adjust_sandbox_resources(
    dataset_stats: str,
    previous_resources: str = "",
    error: str = "",
    model: str = "claude-haiku-4-5",
) -> str:
    """Provision sandbox resources."""
    messages = [{"role": "user", "content": f"Provision sandbox resources for this dataset: {dataset_stats}"}]
    if previous_resources:
        messages.append({"role": "user", "content": f"Previous resources: {previous_resources}"})
    if error:
        messages.append({"role": "user", "content": f"Error encountered: {error}"})
    raw = await _call_llm(model, SYSTEM_PROMPT_PROVISION_SANDBOX_RESOURCES, messages)
    return _extract_resources(raw)


async def get_python_dependencies(
    code: str,
    model: str = "claude-haiku-4-5",
) -> list[str]:
    """Get Python dependencies from code."""
    messages = [{"role": "user", "content": f"Get Python dependencies from this code: {code}"}]
    raw = await _call_llm(model, SYSTEM_PROMPT_GET_PYTHON_DEPENDENCIES, messages)
    return _extract_python_dependencies(raw)


@flyte.trace
async def get_data_stats(data: File) -> str:
    """Get file-level statistics for resource provisioning without loading the file."""
    import flyte.storage as storage

    fs = storage.get_underlying_filesystem(path=data.path)
    info = fs.info(data.path)

    stats = {
        "size_bytes": info.get("size", 0),
        "path": data.path,
        "name": data.name,
    }
    return json.dumps(stats, indent=2)


def _memory_to_mb(mem: object) -> int:
    """Convert a Kubernetes-style memory string (e.g. "2Gi", "512Mi") to MB."""
    s = str(mem).strip()
    for suffix, factor in (
        ("Ti", 1024 * 1024),
        ("Gi", 1024),
        ("Mi", 1),
        ("Ki", 1 / 1024),
        ("T", 1000 * 1000),
        ("G", 1000),
        ("M", 1),
        ("K", 1 / 1000),
    ):
        if s.endswith(suffix):
            return max(1, int(float(s[: -len(suffix)]) * factor))
    return max(1, int(float(s) / (1024 * 1024)))


def _resource_ceilings(resources: flyte.Resources) -> tuple[int, int]:
    """Derive the in-pod sandbox ``(mem_ceiling_mb, cpu_ceiling_milli)`` from resources."""
    mem_mb = _memory_to_mb(resources.memory)
    cpu_milli = int(float(resources.cpu) * 1000)
    return mem_mb, cpu_milli


def _looks_like_oom(returncode: int | None, stderr: str) -> bool:
    """Heuristic for an out-of-memory kill of a sandboxed process."""
    if returncode in (137, -9):
        return True
    low = (stderr or "").lower()
    return any(s in low for s in ("memoryerror", "out of memory", "oomkilled", "killed", "cannot allocate memory"))


async def _read_file_bytes(data: File) -> bytes:
    """Download the input file and return its raw bytes for staging into the sandbox."""
    local_path = await data.download()
    with open(local_path, "rb") as f:  # noqa: ASYNC230
        return f.read()


async def _exec(
    sbx: sb.SandboxSession,
    *,
    cmd: str,
    script_type: str,
    network_mode: str | None,
) -> tuple[int | None, str, str, bool]:
    """Run a single command in the sandbox and classify the outcome.

    Returns ``(returncode, stdout, stderr, oom)``. A non-zero exit is *not*
    raised; ``SandboxExecutionError`` (abnormal termination, e.g. an OOM-kill)
    is mapped to ``oom=True``.
    """
    proc = await sbx.run(
        cmd,
        script_type=script_type,
        stdout=True,
        stderr=True,
        network_mode=network_mode,
    )
    try:
        out, err = await proc.communicate_text()
    except sb.SandboxExecutionError as exc:
        return None, "", str(exc), True
    return proc.returncode, out, err, _looks_like_oom(proc.returncode, err)


async def run_in_sandbox(
    *,
    code: str,
    data_bytes: bytes,
    dependencies: list[str],
    resources: flyte.Resources,
) -> tuple[str, str]:
    """Execute generated code in a fresh interactive sandbox session.

    Returns ``(status, error)`` where ``status`` is one of ``"ok"``, ``"oom"``,
    or ``"error"`` — mirroring the original agent's OOMError vs Exception split.
    """
    mem_mb, cpu_milli = _resource_ceilings(resources)

    # Session-level allow-list lets `uv pip install` reach PyPI; individual
    # run() calls tighten to network_mode="blocked".
    async with await sb.session(
        network_mode="allowlist",
        network_allowlist=sb.PYPI_HOSTS,
        mem_ceiling_mb=mem_mb,
        cpu_ceiling_milli=cpu_milli,
        max_runtime_s=600,
        timeout=timedelta(minutes=15),
    ) as sbx:
        # Stage the input CSV into the persistent work dir (cwd defaults here).
        await sbx.put_bytes(f"{sbx.work_dir}/data", data_bytes)

        # Install dependencies into the session venv (uses the session's
        # allow-list default). An OOM here counts as a resource failure too.
        if dependencies:
            rc, _out, err, oom = await _exec(
                sbx,
                cmd=f"uv pip install {' '.join(dependencies)}",
                script_type="shell",
                network_mode=None,
            )
            if oom:
                return "oom", err
            if rc != 0:
                return "error", f"Dependency install failed:\n{err}"

        # Run the generated code with the network blocked.
        rc, out, err, oom = await _exec(
            sbx,
            cmd=code,
            script_type="python",
            network_mode="blocked",
        )
        if oom:
            return "oom", err
        if rc == 0:
            return "ok", ""
        return "error", err or out


async def _build_report(code: str) -> str:
    import html

    escaped = html.escape(code)
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLE Tool Builder Agent</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 24px;
            background: #f6f8fa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }}
        pre {{
            margin: 0;
            border-radius: 8px;
            border: 1px solid #d1d9e0;
            background: #ffffff;
            overflow-x: auto;
        }}
        pre code {{
            display: block;
            padding: 20px 24px;
            font-family: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
            font-size: 13px;
            line-height: 1.6;
            tab-size: 4;
        }}
    </style>
</head>
<body>
    <pre><code class="language-python">{escaped}</code></pre>
    <script>hljs.highlightAll();</script>
</body>
</html>"""


@agent_env.task
async def deploy_sandbox_environment(resources_str: str, dependencies: list[str]) -> str:
    """Deploy the validated tool as a reusable custom ``SandboxEnvironment``."""
    flyte.init_in_cluster()

    image = sb.base_sandbox_image
    if dependencies:
        image = image.with_pip_packages(*dependencies)

    resources = flyte.Resources(**json.loads(resources_str)) if resources_str else flyte.Resources(cpu=1, memory="2Gi")

    sandbox_env = sb.SandboxEnvironment(
        name="mle-sandbox-training",
        image=image,
        resources=resources,
    )
    v = flyte.deploy(sandbox_env)
    print("Deployed sandbox environment:", v[0].summary_repr())
    return v[0].summary_repr()


@agent_env.task(retries=0, report=True)
async def mle_tool_builder_agent(
    prompt: str,
    data: File,
    max_iter: int = 10,
) -> tuple[str, str, list[str], str]:
    """MLE agent that builds its own tools via an interactive code sandbox.

    This agent:
    1. Takes a user prompt describing what to do with the data
    2. Generates Python code to fulfill the request
    3. Executes the code in a live, isolated ``union.sandbox`` session
    4. On failure, regenerates code with tests and retries

    Args:
        prompt: Natural language description of what to do with the data
        data: Input data file (CSV format expected)
        max_iter: Maximum number of iterations to try fixing errors

    Returns:
        The final working code that was executed
    """
    code = await write_code(prompt)

    # Stage the input once; it's pushed into each fresh session's work dir.
    data_bytes = await _read_file_bytes(data)

    # 🔥 the first attempt will OOM (tiny memory ceiling), so the agent will
    # provision more resources.
    resources_str = ""
    resources = flyte.Resources(cpu=1, memory="10Mi")

    for attempt in range(1, max_iter + 1):
        if attempt == 2:
            # 🔥 on the second attempt, introduce a bug in the code, which the
            # agent should fix.
            code = f"1234 / 0\n\n{code}\n"  # division by zero

        dependencies = await get_python_dependencies(code)
        if attempt == 3:
            # 🔥 on the third attempt, remove all dependencies, which the agent
            # should be able to fix by adding them back.
            dependencies = []

        tab = flyte.report.get_tab(f"Attempt {attempt}")
        tab.replace(await _build_report(code))
        await flyte.report.flush.aio()

        status, error = await run_in_sandbox(
            code=code,
            data_bytes=data_bytes,
            dependencies=dependencies,
            resources=resources,
        )

        if status == "ok":
            break
        elif status == "oom":
            if attempt < max_iter - 1:
                resources_str = await adjust_sandbox_resources(
                    dataset_stats=await get_data_stats(data),
                    previous_resources=str(asdict(resources)),
                    error=error,
                )
                resources = flyte.Resources(**json.loads(resources_str))
            else:
                raise RuntimeError(f"Failed to run code after {max_iter} attempts. Last error: {error}")
        else:
            if attempt < max_iter - 1:
                code = await write_code(
                    prompt=prompt,
                    previous_code=code,
                    error=error,
                )
            else:
                raise RuntimeError(f"Failed to generate working code after {max_iter} attempts. Last error: {error}")

    await flyte.report.replace.aio(await _build_report(code))
    await flyte.report.flush.aio()
    deploy_summary = await deploy_sandbox_environment(resources_str, dependencies)
    return code, resources_str, dependencies, deploy_summary


if __name__ == "__main__":
    import asyncio
    import tempfile

    flyte.init_from_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n")
        for i in range(10_000):
            f.write(f"{i},{i * 2},{i * 3}\n")
        data_path = f.name

    async def main():
        data_file = await File.from_local(data_path)
        run = flyte.run(
            mle_tool_builder_agent,
            prompt=(
                "Train a linear regression model to predict 'target' from 'feature1' "
                "and 'feature2' from this csv file using pandas."
            ),
            data=data_file,
            max_iter=10,
        )
        print(f"View at: {run.url}")
        run.wait()

        print(f"Result: {run.outputs()}")

    asyncio.run(main())
