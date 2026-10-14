"""MLE Tool Builder Agent — Builds its own tools via Flyte training tasks.

This agent demonstrates how to:
1. Use an LLM to generate Python code that processes data
2. Execute that code in an isolated Flyte training task (separate pod)
3. Iteratively fix errors by re-generating code with tests

The agent takes a user prompt, data, and max_iter budget and outputs
the code that fulfills the prompt.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import flyte
import flyte.errors
import flyte.report
from flyte.extras import ContainerTask
from flyte.io import File

training_env = flyte.TaskEnvironment(
    "mle-training",
    image=flyte.Image.from_debian_base(name="mle-training-image").with_pip_packages("pip"),
)

agent_env = flyte.TaskEnvironment(
    "mle-tool-builder",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mle-tool-builder-image").with_pip_packages(
            "httpx",
            "pandas",
            "scikit-learn",
            "numpy",
        )
    ),
    depends_on=[training_env],
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

IMPORTANT: Your code will run in an isolated Flyte training task with the following constraints:
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
You are an expert ML engineer. Provision training-task resources for this dataset.

IMPORTANT: Your code will run in an isolated Flyte training task with the following constraints:
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
    model: str = "claude-sonnet-4-6",
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
    model: str = "claude-sonnet-4-6",
) -> str:
    """Provision sandbox resources."""
    messages = [{"role": "user", "content": f"Provision sandbox resources for this dataset: {dataset_stats}"}]
    if previous_resources:
        messages.append({"role": "user", "content": f"Previous resources: {previous_resources}"})
    if error:
        messages.append({"role": "user", "content": f"Error encountered: {error}"})
    raw = await _call_llm(model, SYSTEM_PROMPT_PROVISION_SANDBOX_RESOURCES, messages)
    return _extract_resources(raw)


def _training_image(dependencies: list[str], image_name: str) -> flyte.Image:
    """Build an image spec for a generated training script."""
    image = flyte.Image.from_debian_base(install_flyte=False, name=image_name)
    image = image.with_apt_packages("gcc", "g++", "make")
    if dependencies:
        image = image.with_pip_packages(*dependencies)
    return image


async def _build_training_image_uri(dependencies: list[str], image_name: str) -> str:
    result = await flyte.build.aio(_training_image(dependencies, image_name))
    if result.uri is None:
        raise RuntimeError("Image build succeeded but returned no URI.")
    return result.uri


def _make_training_container_task(
    *,
    image_uri: str,
    task_name: str,
    resources: flyte.Resources,
) -> ContainerTask:
    """Mirror the verbatim-mode container layout used by ``flyte.sandbox``."""
    bash_cmd = (
        "set -o pipefail && "
        "echo '--- script ---' && cat \"$1\" && "
        "echo '--- running ---' && "
        "python $1 --data $2; "
        "_exit=$?; echo $_exit > /var/outputs/exit_code; exit $_exit"
    )
    return ContainerTask(
        name=task_name,
        image=image_uri,
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={"data": File, "_script": File},
        outputs={"model": File},
        command=["/bin/bash", "-c", bash_cmd],
        arguments=["/bin/bash", "/var/inputs/_script", "/var/inputs/data"],
        resources=resources,
        retries=0,
    )


async def _upload_script(code: str, task_name: str) -> File:
    script_path = Path(tempfile.gettempdir()) / f"{task_name}_generated.py"
    script_path.write_text(code)
    return await File.from_local(
        str(script_path),
        hash_method=hashlib.sha256(code.encode()).hexdigest(),
    )


@training_env.task(retries=0, cache="auto")
async def run_training_subjob(code: str, data: File, dependencies_json: str) -> File:
    """Execute generated training code in an isolated training-task pod."""
    import asyncio
    import sys
    import traceback

    dependencies: list[str] = json.loads(dependencies_json)
    if dependencies:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *dependencies,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Dependency install failed:\n{stderr.decode()}")

    inputs_dir = Path("/var/inputs")
    outputs_dir = Path("/var/outputs")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    local_data = await data.download()
    shutil.copy(local_data, inputs_dir / "data")

    try:
        exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})
    except Exception:
        raise RuntimeError(traceback.format_exc()) from None

    model_path = outputs_dir / "model"
    if not model_path.exists():
        raise RuntimeError("Generated code did not create /var/outputs/model")
    return await File.from_local(str(model_path))


async def get_python_dependencies(
    code: str,
    model: str = "claude-sonnet-4-6",
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
async def deploy_training_task(code: str, resources_str: str, dependencies: list[str]) -> str:
    """Deploy the validated training script as a reusable TaskEnvironment."""
    flyte.init_in_cluster()

    resources = flyte.Resources(**json.loads(resources_str)) if resources_str else flyte.Resources(cpu=1, memory="2Gi")
    task_name = "mle-training"
    image_uri = await _build_training_image_uri(dependencies, f"{task_name}-deploy")
    task = _make_training_container_task(
        image_uri=image_uri,
        task_name=task_name,
        resources=resources,
    )
    script_file = await _upload_script(code, task_name)
    from flyte.models import NativeInterface

    new_inputs = dict(task.interface.inputs)
    new_inputs["_script"] = (File, script_file)
    task.interface = NativeInterface(new_inputs, dict(task.interface.outputs))
    env = flyte.TaskEnvironment.from_task("mle-training-deploy", task)
    v = flyte.deploy(env)
    print("Deployed environment:", v[0].summary_repr())
    return v[0].summary_repr()


@agent_env.task(retries=0, report=True)
async def mle_tool_builder_agent(
    prompt: str,
    data: File,
    max_iter: int = 10,
) -> tuple[str, str, list[str], str]:
    """MLE agent that builds its own tools via Flyte training tasks.

    This agent:
    1. Takes a user prompt describing what to do with the data
    2. Generates Python code to fulfill the request
    3. Executes the code in an isolated training-task pod
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

        try:
            await run_training_subjob.override(
                resources=resources,
                short_name=f"mle-training-attempt-{attempt}",
            )(code=code, data=data, dependencies_json=json.dumps(dependencies))
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
                raise RuntimeError(f"Failed to run code after {max_iter} attempts. Last error: {error}") from exc
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
                    f"Failed to generate working code after {max_iter} attempts. Last error: {error}"
                ) from exc

    await flyte.report.replace.aio(await _build_report(code))
    await flyte.report.flush.aio()
    deploy_summary = await deploy_training_task(code, resources_str, dependencies)
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
