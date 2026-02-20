"""
Opaque IO — Passing Files and DataFrames Through a Sandbox
==========================================================

Sandboxed code runs in Monty — a Rust-based Python interpreter with **no
filesystem, network, or OS access**. But ``flyte.io.File``, ``flyte.io.Dir``,
and ``flyte.io.DataFrame`` are allowed at the sandbox boundary as **opaque
handles**. The sandbox can receive them, pass them between regular worker
tasks, and return them — but it can never read, download, or inspect their
contents.

This is useful when:
- An orchestrator decides *which* data to process and *how* to route it,
  but should never see the data itself (e.g. PII isolation, compliance)
- You want to compose a multi-step data pipeline from a code string or
  LLM-generated code, while keeping all actual IO in auditable worker tasks
- You need to pass large artifacts through control-flow logic without
  copying them

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

import os
import tempfile

import aiofiles
import pandas as pd

import flyte
import flyte.io
import flyte.sandbox

img = flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow", "pydantic-monty", "aiofiles")
env = flyte.TaskEnvironment(name="opaque-io-demo", image=img)


# --- Worker tasks — these have real IO access --------------------------------


@env.task
async def create_csv_file() -> flyte.io.File:
    """Write a CSV file to remote storage and return an opaque handle."""
    f = flyte.io.File.new_remote("sales.csv")
    async with f.open("wb") as fp:
        await fp.write(b"region,quarter,revenue\nnorth,Q1,120000\nsouth,Q1,95000\nnorth,Q2,135000\nsouth,Q2,110000\n")
    return f


@env.task
async def create_dataframe() -> flyte.io.DataFrame:
    """Create a DataFrame and return it as an opaque handle."""
    df = pd.DataFrame(
        {
            "product": ["Widget A", "Widget B", "Widget C", "Widget D"],
            "category": ["electronics", "furniture", "electronics", "furniture"],
            "price": [29.99, 149.99, 59.99, 249.99],
            "units_sold": [500, 120, 340, 80],
        }
    )
    return flyte.io.DataFrame.wrap_df(df)


@env.task
async def create_report_dir() -> flyte.io.Dir:
    """Create a directory of report files and upload it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, content in [
            ("summary.txt", "Q1 total revenue: $215,000\n"),
            ("notes.txt", "South region underperforming target by 5%.\n"),
        ]:
            async with aiofiles.open(os.path.join(tmpdir, name), "w") as f:
                await f.write(content)
        return await flyte.io.Dir.from_local(tmpdir)


@env.task
async def count_csv_rows(f: flyte.io.File) -> int:
    """Download a CSV file and count its data rows."""
    async with f.open("rb") as fp:
        data = await fp.read()
    lines = bytes(data).decode().strip().split("\n")
    return len(lines) - 1  # exclude header


@env.task
async def filter_dataframe(df: flyte.io.DataFrame, category: str) -> flyte.io.DataFrame:
    """Filter a DataFrame to rows matching *category*."""
    pandas_df = await df.open(pd.DataFrame).all()
    filtered = pandas_df[pandas_df["category"] == category]
    return flyte.io.DataFrame.wrap_df(filtered)


@env.task
async def total_revenue(df: flyte.io.DataFrame) -> float:
    """Compute total revenue (price * units_sold) from a product DataFrame."""
    pandas_df = await df.open(pd.DataFrame).all()
    return float((pandas_df["price"] * pandas_df["units_sold"]).sum())


@env.task
async def count_dir_files(d: flyte.io.Dir) -> int:
    """Count files in a directory."""
    files = await d.list_files()
    return len(files)


@env.task
async def merge_files_to_dir(f: flyte.io.File, d: flyte.io.Dir) -> flyte.io.Dir:
    """Download a file and a directory, merge them into a new directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download original directory contents
        dir_path = await d.download(os.path.join(tmpdir, "base"))
        # Download the file into the same directory
        await f.download(os.path.join(dir_path, "sales.csv"))
        return await flyte.io.Dir.from_local(dir_path)


# --- Example 1: File pass-through ------------------------------------------
# The sandbox receives a File, passes it to a worker, and returns the count.
# It never sees the CSV contents.

file_pipeline = env.sandbox.orchestrator(
    """
    csv_file = create_csv_file()
    row_count = count_csv_rows(csv_file)
    result = {"file": csv_file, "rows": row_count}
    """,
    inputs={},
    output=dict,
    tasks=[create_csv_file, count_csv_rows],
    name="file-pipeline",
)
# The sandbox orchestrates the flow: create -> count -> return
# It holds the File handle but cannot open or read it.


# --- Example 2: DataFrame routing ------------------------------------------
# The sandbox creates a DataFrame, filters it by category, and computes
# revenue — all via worker tasks. The sandbox only sees the opaque handle
# and the final float.

dataframe_pipeline = env.sandbox.orchestrator(
    """
    products = create_dataframe()
    electronics = filter_dataframe(products, "electronics")
    electronics_revenue = total_revenue(electronics)
    all_revenue = total_revenue(products)
    result = {
        "electronics_revenue": electronics_revenue,
        "all_revenue": all_revenue,
    }
    """,
    inputs={},
    output=dict,
    tasks=[create_dataframe, filter_dataframe, total_revenue],
    name="dataframe-pipeline",
)
# The sandbox decides *what* to filter and *which* revenue to compute,
# but never materializes the DataFrame itself.


# --- Example 3: Directory pass-through -------------------------------------
# The sandbox passes a Dir handle to a worker that counts its files.

dir_pipeline = env.sandbox.orchestrator(
    """
    report = create_report_dir()
    n_files = count_dir_files(report)
    result = {"dir": report, "file_count": n_files}
    """,
    inputs={},
    output=dict,
    tasks=[create_report_dir, count_dir_files],
    name="dir-pipeline",
)


# --- Example 4: Combining File, Dir, and DataFrame -------------------------
# A single sandbox orchestrates all three IO types, merging a File into a Dir
# and computing DataFrame stats — without touching any data directly.

combined_pipeline = env.sandbox.orchestrator(
    """
    csv_file = create_csv_file()
    report_dir = create_report_dir()
    products = create_dataframe()

    merged = merge_files_to_dir(csv_file, report_dir)
    merged_count = count_dir_files(merged)

    electronics = filter_dataframe(products, "electronics")
    revenue = total_revenue(electronics)
    row_count = count_csv_rows(csv_file)

    result = {
        "merged_file_count": merged_count,
        "electronics_revenue": revenue,
        "csv_rows": row_count,
    }
    """,
    inputs={},
    output=dict,
    tasks=[
        create_csv_file,
        create_report_dir,
        create_dataframe,
        merge_files_to_dir,
        count_dir_files,
        filter_dataframe,
        total_revenue,
        count_csv_rows,
    ],
    name="combined-pipeline",
)


# --- Example 5: Parameterized sandbox with IO inputs -----------------------
# The sandbox receives a File and DataFrame as inputs from the caller,
# routes them to workers, and returns derived results.

parameterized_pipeline = env.sandbox.orchestrator(
    """
    rows = count_csv_rows(csv_file)
    revenue = total_revenue(product_df)
    result = {"csv_rows": rows, "revenue": revenue}
    """,
    inputs={"csv_file": flyte.io.File, "product_df": flyte.io.DataFrame},
    output=dict,
    tasks=[count_csv_rows, total_revenue],
    name="parameterized-pipeline",
)
# flyte.run(parameterized_pipeline, csv_file=some_file, product_df=some_df)


# --- Example 6: @env.sandbox.orchestrator decorator -------------------------
# Instead of a code string, use a decorated function as the orchestrator.
# The sandbox can call regular worker tasks but still cannot access file
# contents directly.


@env.sandbox.orchestrator
def orchestrate_etl(category: str) -> dict:
    csv_file = create_csv_file()
    row_count = count_csv_rows(csv_file)

    products = create_dataframe()
    filtered = filter_dataframe(products, category)
    revenue = total_revenue(filtered)

    report_dir = create_report_dir()
    merged = merge_files_to_dir(csv_file, report_dir)
    file_count = count_dir_files(merged)

    return {
        "csv_rows": row_count,
        "revenue": revenue,
        "merged_file_count": file_count,
    }


# --- Attach code-string tasks to an environment for ``flyte run`` ---------
# The code-string tasks call workers defined in ``env``, so declare that
# dependency with ``depends_on`` so both environments are deployed together.

sandbox_env = flyte.TaskEnvironment.from_task(
    "opaque-io-code-tasks",
    file_pipeline,
    dataframe_pipeline,
    dir_pipeline,
    combined_pipeline,
    parameterized_pipeline,
    depends_on=[env],
)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(combined_pipeline)
    print(r.url)
