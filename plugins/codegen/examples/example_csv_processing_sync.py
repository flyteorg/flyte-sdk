"""Example: CSV processing with synchronous tasks (LLM approach).

Demonstrates:
- Synchronous (non-async) task definitions using agent.generate() and result.run()
- Passing File data (auto-sampled by LLM)
- Using result.run() for one-off execution
- Multiple typed outputs (float, int)
"""

import logging
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

agent = AutoCoderAgent(
    name="csv-sync",
    model="gpt-4.1",
    max_iterations=5,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    litellm_params={"temperature": 0.2, "max_tokens": 4096},
)

env = flyte.TaskEnvironment(
    name="csv-sync-example",
    secrets=[
        flyte.Secret(key="niels_openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="5Gi"),
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-codegen",
                pre=True,
            ),
        )
        .with_apt_packages("git")
        .with_pip_packages(
            "git+https://github.com/flyteorg/flyte-sdk.git@86f88fece16d956e28667d3f0d8d49108c8cdd68"
        )
    ),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def prepare_sales_data() -> File:
    """Create a sample CSV file for processing."""
    csv_content = """date,product,quantity,price
2024-01-01,Widget A,10,25.50
2024-01-02,Widget B,5,30.00
2024-01-03,Widget A,8,25.50
2024-01-04,Widget B,12,30.00
2024-01-05,Widget A,3,25.50
"""
    csv_path = Path(tempfile.gettempdir()) / "sales_data.csv"
    csv_path.write_text(csv_content)
    return File.from_local_sync(str(csv_path))


@env.task
def generate_and_run(sales_csv: File) -> dict[str, float | int | dict]:
    """Generate and run sales analysis code synchronously."""
    result = agent.generate(
        prompt="""Process sales data and calculate summary statistics.

Return total_revenue (sum of quantity * price), total_units (sum of quantity),
and transaction_count (number of rows).""",
        samples={"sales_csv": sales_csv},
        outputs={
            "total_revenue": float,
            "total_units": int,
            "transaction_count": int,
        },
    )

    if not result.success:
        return {"error": result.error, "attempts": result.attempts}

    total_revenue, total_units, transaction_count = result.run()

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "transaction_count": transaction_count,
        "attempts": result.attempts,
        "tokens": {
            "input": result.total_input_tokens,
            "output": result.total_output_tokens,
        },
    }


@env.task
def csv_sync_workflow() -> dict[str, float | int | dict]:
    """Process CSV data using synchronous code generation and execution."""
    sales_csv = prepare_sales_data()
    return generate_and_run(sales_csv=sales_csv)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(csv_sync_workflow)
    print(f"\nRun URL: {run.url}")
