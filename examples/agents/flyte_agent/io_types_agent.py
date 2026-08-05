"""Agent tools with flyte.io File, Dir, and DataFrame inputs/outputs.

Demonstrates passing Flyte blob and structured-dataset types through the
:class:`flyte.ai.agents.Agent` tool loop: the LLM receives JSON schemas for
``File``, ``Dir``, and ``DataFrame``, tool arguments are coerced from those
schemas, and tool results are serialized back for the model.

Run locally (requires ``ANTHROPIC_API_KEY``)::

    uv run python examples/agents/flyte_agent/io_types_agent.py

Run on the demo cluster::

    flyte run examples/agents/flyte_agent/io_types_agent.py run_io_agent \\
        --prompt "Create a CSV with columns name,score for Ada:98 and Bob:87, save it, and summarize the file path."
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import flyte
from flyte.ai.agents import Agent
from flyte.io import DataFrame, File

img = flyte.Image.from_debian_base().with_pip_packages("litellm", "pandas", "pyarrow")

env = flyte.TaskEnvironment(
    name="io-types-agent",
    image=img,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def write_scores_csv(rows: list[dict[str, str | int]]) -> File:
    """Write *rows* to a CSV file and return a flyte.io.File reference."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scores.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return await File.from_local(str(path))


@env.task
async def describe_file(file: File) -> dict[str, str]:
    """Return metadata about a flyte.io.File (path and format)."""
    return {"path": file.path, "format": file.format or "unknown"}


@env.task
async def dataframe_from_csv(csv_file: File) -> DataFrame:
    """Wrap a CSV flyte.io.File as a flyte.io.DataFrame reference."""
    return DataFrame.from_existing_remote(remote_path=csv_file.path, format="csv")


@env.task
async def summarize_dataframe(df: DataFrame) -> dict[str, str | int]:
    """Load a flyte.io.DataFrame and return basic column/shape metadata."""
    import pandas as pd

    table = await df.open(pd.DataFrame).all()
    return {
        "columns": ",".join(table.columns),
        "rows": len(table),
        "uri": df.uri or "",
    }


agent = Agent(
    name="io-types-helper",
    instructions=(
        "You demonstrate flyte.io File and DataFrame tools. For CSV requests you MUST "
        "call tools in this exact order before answering:\n"
        "1. write_scores_csv — create the CSV from the requested rows\n"
        "2. describe_file — pass the File returned from step 1 as `file`\n"
        "3. dataframe_from_csv — pass the same File as `csv_file`\n"
        "4. summarize_dataframe — pass the DataFrame from step 3 as `df`\n"
        "Only after all four succeed, reply with a one-sentence summary of the table metadata."
    ),
    model="claude-sonnet-4-20250514",
    tools=[write_scores_csv, describe_file, dataframe_from_csv, summarize_dataframe],
    max_turns=10,
)

# Code mode is more reliable for multi-step IO pipelines: the LLM writes a short
# Python program that passes File/DataFrame handles between tools explicitly.
code_mode_agent = Agent(
    name="io-types-helper",
    instructions=(
        "Write Python that calls the available tools in order: write_scores_csv, "
        "describe_file, dataframe_from_csv, summarize_dataframe. Pass return values "
        "directly between calls. End with a string summary of the summarize_dataframe result."
    ),
    model="claude-sonnet-4-20250514",
    tools=[write_scores_csv, describe_file, dataframe_from_csv, summarize_dataframe],
    max_turns=8,
    code_mode=True,
)


@env.task
async def run_io_pipeline(rows: list[dict[str, str | int]]) -> dict[str, str | int]:
    """Deterministic IO-type pipeline (no LLM): CSV → File → DataFrame → summary."""
    csv_file = await write_scores_csv(rows=rows)
    meta = await describe_file(file=csv_file)
    df = await dataframe_from_csv(csv_file=csv_file)
    summary = await summarize_dataframe(df=df)
    return {
        "file_path": meta["path"],
        "columns": summary["columns"],
        "rows": summary["rows"],
        "dataframe_uri": summary["uri"],
    }


@env.task
async def run_io_agent(prompt: str) -> str:
    """Run the IO-types agent (code mode) with *prompt* and return the final summary."""
    result = await code_mode_agent.run.aio(prompt)
    return result.summary or result.error


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        or "Create a CSV with name,score for Ada:98 and Bob:87, describe the file, "
        "wrap it as a DataFrame, and summarize the table."
    )
    result = agent.run(prompt)
    if result.error:
        print(f"[error] {result.error}")
        sys.exit(1)
    print(result.summary)


if __name__ == "__main__":
    main()
