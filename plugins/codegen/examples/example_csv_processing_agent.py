import asyncio
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

env = flyte.TaskEnvironment(
    name="batch-processing-agent-example",
    secrets=[
        flyte.Secret(key="samhita_anthropic_api_key", as_env_var="ANTHROPIC_API_KEY"),
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
        .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@86f88fece16d956e28667d3f0d8d49108c8cdd68")
    ),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def prepare_sales_data() -> list[list[str]]:
    """Prepare test data in different CSV formats."""
    return [
        [
            """date,product,quantity,price
        2024-01-01,Widget A,10,25.50
        2024-01-02,Widget B,5,30.00
        2024-01-03,Widget A,8,25.50""",
            "Vendor A (comma-separated, standard columns)",
        ],
        [
            """transaction_date,item_name,units_sold,total_revenue
                2024-01-01,Widget C,12,306.00
                2024-01-02,Widget D,3,45.00
                2024-01-03,Widget C,7,178.50""",
            "Vendor B (comma-separated, different column names)",
        ],
        [
            """product_id,sale_date,qty,unit_price
        WA,2024-01-01,15,20.00
        WB,2024-01-02,8,35.50
        WA,2024-01-03,10,20.00""",
            "Vendor C (comma-separated, different column order)",
        ],
    ]


@env.task
async def generate_and_run_with_agent(
    prompt: str, csv_data: str, description: str, index: int
) -> dict[str, float | int]:
    """Generate code and run for a single CSV format using Agent SDK."""

    # Create temp CSV file
    csv_file = Path(tempfile.gettempdir()) / f"sales_data_agent_{index}.csv"
    csv_file.write_text(csv_data)

    # Generate code with Agent SDK - agent autonomously handles the workflow
    print(f"\n[{description}] Starting Agent SDK code generation...")

    agent = AutoCoderAgent(
        name=f"sales-agent-{index}",
        use_agent_sdk=True,
        resources=flyte.Resources(cpu=1, memory="512Mi"),
        model="claude-sonnet-4-5-20250929",
    )

    result = await agent.generate.aio(
        prompt=prompt,
        samples={"csv_data": await File.from_local(str(csv_file))},
        outputs={
            "total_revenue": float,
            "total_units": int,
            "transaction_count": int,
        },
    )

    if not result.success:
        print(f"{description}: Agent SDK code generation failed")
        print(f"Error: {result.error}")
        return None

    print(f"[{description}] Agent SDK generated solution successfully!")
    print(f"{len(result.detected_packages)} packages detected")

    # Run generated code on the original data
    total_revenue, total_units, transaction_count = await result.run.aio(name=f"run_sales_agent_{index}")

    print(f"[{description}] Success!")
    print(f"  - Total revenue: ${total_revenue:.2f}")
    print(f"  - Total units: {total_units}")
    print(f"  - Transactions: {transaction_count}")

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "transaction_count": transaction_count,
    }


@env.task
async def sales_processing_agent_workflow(
    prompt: str = """Process sales data and calculate total revenue.

Return JSON with: total_revenue (float), total_units (int), transaction_count (int)""",
) -> list[dict[str, float | int]]:
    """Process multiple CSV formats using Agent SDK.

    This demonstrates the experimental Agent SDK mode where Claude autonomously:
    1. Analyzes the data
    2. Generates code
    3. Writes tests
    4. Runs tests
    5. Fixes issues
    6. Iterates until tests pass

    All of this happens inside a sandbox without human intervention.
    """
    sales_data = prepare_sales_data()

    tasks = [
        generate_and_run_with_agent(prompt=prompt, csv_data=csv_data, description=desc, index=i)
        for i, (csv_data, desc) in enumerate(sales_data)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(sales_processing_agent_workflow)
    print(f"\nRun URL: {run.url}")
