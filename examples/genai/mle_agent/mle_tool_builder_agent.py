"""MLE Tool Builder Agent — Builds its own tools via code sandbox.

This agent demonstrates how to:
1. Use an LLM to generate Python code that processes data
2. Execute that code in an isolated code sandbox
3. Iteratively fix errors by re-generating code with tests

The agent takes a user prompt, data, and max_iter budget and outputs
the code that fulfills the prompt.
"""

import json
import os
import re
from dataclasses import asdict

import flyte
import flyte.errors
import flyte.report
import flyte.sandbox
from flyte.io import File

agent_env = flyte.TaskEnvironment(
    "mle-tool-builder",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="niels-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mle-tool-builder-image")
        .with_pip_packages("httpx", "pandas", "scikit-learn", "numpy")
    ),
    depends_on=[flyte.sandbox.sandbox_environment],
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

IMPORTANT: Your code will run in an isolated sandbox with the following constraints:
- The input data is available as a pandas DataFrame in the file at `/var/inputs/data`
- You must save your output to a file at `/var/outputs/model`
- You have access to: pandas, numpy, scikit-learn
- You cannot use any network calls or external APIs

Your code should:
1. Process the input DataFrame as needed
2. Train a model or perform the requested analysis
3. Save the result to `/var/outputs/model` (use pickle, joblib, write text/json, save plot images, etc.)

Example structure:
```python
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# data is already available as a pandas DataFrame
X = data[['feature1', 'feature2']]
y = data['target']

model = LinearRegression()
model.fit(X, y)

# Save the model
with open('/var/outputs/model', 'wb') as f:
    pickle.dump(model, f)
```
"""

SYSTEM_PROMPT_PROVISION_SANDBOX_RESOURCES = """\
You are an expert ML engineer. Provision sandbox resources for this dataset.

IMPORTANT: Your code will run in an isolated sandbox with the following constraints:
- The input data is available as a pandas DataFrame in the file at `/var/inputs/data`
- You must save your output to a file at `/var/outputs/model`
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
    model: str = "claude-sonnet-4-20250514",
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
    model: str = "claude-sonnet-4-20250514",
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
    model: str = "claude-sonnet-4-20250514",
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


async def _build_report(code: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLE Tool Builder Agent</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
    </head>

    <body>
        <pre><code class="language-python">{code}</code></pre>
    </body>
    </html>
    """


@agent_env.task
async def deploy_sandbox_task(code: str, resources_str: str, dependencies: list[str]) -> str:
    flyte.init_in_cluster()

    sandbox = flyte.sandbox.create(
        name="mle-sandbox-training",
        code=code,
        inputs={"data": File},
        outputs={"model": File},
        packages=dependencies,
        resources=flyte.Resources(**json.loads(resources_str)),
        auto_io=False,
        block_network=True,
    )
    image = sandbox.image or await sandbox._build.aio()
    task_name = sandbox._task_name()
    task = sandbox._make_container_task(image, task_name)
    env = flyte.TaskEnvironment.from_task("mle-sandbox-training", task)
    v = flyte.deploy(env)
    print("Deployed environment:", v[0].summary_repr())
    return v[0].summary_repr()


@agent_env.task(retries=0, report=True)
async def mle_tool_builder_agent(
    prompt: str,
    data: File,
    max_iter: int = 10,
) -> tuple[str, str, list[str], str]:
    """MLE agent that builds its own tools via code sandbox.

    This agent:
    1. Takes a user prompt describing what to do with the data
    2. Generates Python code to fulfill the request
    3. Executes the code in an isolated sandbox
    4. On failure, regenerates code with tests and retries

    Args:
        prompt: Natural language description of what to do with the data
        data: Input data file (CSV format expected)
        max_iter: Maximum number of iterations to try fixing errors

    Returns:
        The final working code that was executed
    """
    code = await write_code(prompt)

    # 🔥 the first attempt will OOM, so the agent will provision more resources
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

        sandbox = flyte.sandbox.create(
            name=f"mle-sandbox-attempt-{attempt}",
            code=code,
            inputs={"data": File},
            outputs={"model": File},
            packages=dependencies,
            resources=resources,
            auto_io=False,
            block_network=True,
        )

        try:
            await sandbox.run.aio(data=data)
            break
        except flyte.errors.OOMError as exc:
            error = str(exc)
            if attempt < max_iter - 1:
                resources_str = await adjust_sandbox_resources(
                    dataset_stats=await get_data_stats(data),
                    previous_resources=str(asdict(resources)),
                    error=error,
                )
                resources = flyte.Resources(**json.loads(resources_str))
            else:
                raise RuntimeError(
                    f"Failed to run code after {max_iter} attempts. "
                    f"Last error: {error}"
                ) from exc
        except Exception as exc:
            error = str(exc)
            if attempt < max_iter - 1:
                code = await write_code(
                    prompt=prompt,
                    previous_code=code,
                    error=error,
                )
            else:
                raise RuntimeError(
                    f"Failed to generate working code after {max_iter} attempts. "
                    f"Last error: {error}"
                ) from exc

    await flyte.report.replace.aio(await _build_report(code))
    await flyte.report.flush.aio()
    deploy_summary = await deploy_sandbox_task(code, resources_str, dependencies)
    return code, resources_str, dependencies, deploy_summary


if __name__ == "__main__":
    import asyncio
    import tempfile

    flyte.init_from_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n")
        for i in range(10_000):
            f.write(f"{i},{i*2},{i*3}\n")
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
