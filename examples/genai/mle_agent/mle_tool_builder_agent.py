"""MLE Tool Builder Agent — Builds its own tools via code sandbox.

This agent demonstrates how to:
1. Use an LLM to generate Python code that processes data
2. Execute that code in an isolated code sandbox
3. Iteratively fix errors by re-generating code with tests

The agent takes a user prompt, data, and max_iter budget and outputs
the code that fulfills the prompt.
"""

import os
import re

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

SYSTEM_PROMPT_WITH_TESTS = """\
You are an expert ML engineer. Write Python code with unit tests to process data and train models.

IMPORTANT: Your code will run in an isolated sandbox with the following constraints:
- The input data is available as a pandas DataFrame in the variable `data`
- You must save your output to a file at `/var/outputs/model`
- You have access to: pandas, numpy, scikit-learn
- You cannot use any network calls or external APIs

Based on the previous error, rewrite the code and include simple unit tests.

Return TWO code blocks:
1. First block: the main code (labeled ```python)
2. Second block: the test code (labeled ```python test)

The test code should:
- Import the necessary modules
- Define test functions that verify the logic works correctly
- Use assert statements for validation
"""


@flyte.trace
async def write_code(
    prompt: str,
    previous_code: str = "",
    error_msg: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate code using the LLM."""
    messages = [{"role": "user", "content": prompt}]
    if previous_code:
        messages.append({"role": "user", "content": f"Previous code that failed: {previous_code}"})
    if error_msg:
        messages.append({"role": "user", "content": f"Error encountered: {error_msg}"})
    raw = await _call_llm(model, SYSTEM_PROMPT, messages)
    return _extract_code(raw)


@flyte.trace
async def write_code_with_tests(
    prompt: str,
    previous_code: str = "",
    error: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> tuple[str, str]:
    """Generate code with tests based on a previous error."""
    user_content = f"""\
Previous code that failed:
```python
{previous_code}
```

Error encountered:
```
{error}
```

User request: {prompt}

Please fix the code and add tests to prevent this error.
"""
    messages = [{"role": "user", "content": user_content}]
    raw = await _call_llm(model, SYSTEM_PROMPT_WITH_TESTS, messages)

    code_blocks = re.findall(r"```(?:python(?: test)?)\s*\n(.*?)```", raw, re.DOTALL)
    if len(code_blocks) >= 2:
        return code_blocks[0].strip(), code_blocks[1].strip()
    elif len(code_blocks) == 1:
        return code_blocks[0].strip(), ""
    return _extract_code(raw), ""


async def _build_report(code: str, tests: str) -> str:
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
        <pre><code class="language-python">{tests}</code></pre>
    </body>
    </html>
    """


@agent_env.task(retries=3, report=True)
async def tool_builder_agent(
    prompt: str,
    data: File,
    max_iter: int = 3,
) -> tuple[str, File]:
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
    code, tests = await write_code_with_tests(prompt)
    
    for attempt in range(max_iter):
        tab = flyte.report.get_tab(f"Attempt {attempt}")
        tab.replace(await _build_report(code, tests))
        await flyte.report.flush.aio()

        sandbox = flyte.sandbox.create(
            name=f"mle-sandbox-attempt-{attempt}",
            code=code,
            inputs={"data": File},
            outputs={"model": File},
            packages=["pandas", "numpy", "scikit-learn", "joblib"],
            auto_io=False,
            block_network=True,
        )

        try:
            model_file = await sandbox.run.aio(data=data)
            return code, model_file
        except flyte.errors.RuntimeUserError as exc:
            error = str(exc)
            if attempt < max_iter - 1:
                code, tests = await write_code_with_tests(
                    prompt=prompt,
                    previous_code=code,
                    error=error,
                )
            else:
                raise RuntimeError(
                    f"Failed to generate working code after {max_iter} attempts. "
                    f"Last error: {error}"
                ) from exc

    return code


if __name__ == "__main__":
    import tempfile

    flyte.init_from_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,target\n")
        for i in range(100):
            f.write(f"{i},{i*2},{i*3}\n")
        data_path = f.name

    async def main():
        data_file = await File.from_local(data_path)
        run = flyte.run(
            tool_builder_agent,
            prompt="Train a linear regression model to predict 'target' from 'feature1' and 'feature2' from this csv file using pandas.",
            data=data_file,
            max_iter=3,
        )
        print(f"View at: {run.url}")
        run.wait()
        print(f"Result: {run.outputs()}")

    import asyncio

    asyncio.run(main())
