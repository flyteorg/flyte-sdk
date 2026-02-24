"""Example: CSV processing with data and result.run() (LLM approach).

Demonstrates:
- Passing File data (auto-sampled by LLM)
- Using result.run() for one-off execution
- Multiple typed outputs (float, int)
- Custom model selection
- litellm_params for LLM tuning (temperature, max_tokens)
- max_retries for controlling retry budget
- resources for sandbox execution
"""

import asyncio
import logging
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)


env = flyte.TaskEnvironment(
    name="batch-processing-agent-example",
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
            "git+https://github.com/flyteorg/flyte-sdk.git@bc446803e1d5d8baee479dddc72e720746a93d54"
        )
    ),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def prepare_sales_data() -> list[list[str]]:
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
async def generate_and_run(
    prompt: str, csv_data: str, description: str, index: int
) -> dict[str, float | int]:
    """Generate code and run for a single CSV format."""
    # Create temp CSV file
    csv_file = Path(tempfile.gettempdir()) / f"sales_data_{index}.csv"
    csv_file.write_text(csv_data)

    # Create agent per item for unique tracing name
    agent = AutoCoderAgent(
        name=f"sales-data-{index}",
        model="gpt-4.1",
        max_retries=5,
        resources=flyte.Resources(cpu=1, memory="512Mi"),
        litellm_params={"temperature": 0.2, "max_tokens": 4096},
    )

    # Generate code with automatic LLM sampling from the data
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
        print(f"{description}: Code generation failed")
        return None

    # Run generated code on the original data
    total_revenue, total_units, transaction_count = await result.run.aio()

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "transaction_count": transaction_count,
    }


@env.task
async def sales_processing_workflow(
    prompt: str = """Process sales data and calculate total revenue.

Return JSON with: total_revenue (float), total_units (int), transaction_count (int)""",
) -> list[dict[str, float | int]]:
    """Process multiple CSV formats: generate code with LLM sampling, then execute."""
    sales_data = prepare_sales_data()

    tasks = [
        generate_and_run(prompt=prompt, csv_data=csv_data, description=desc, index=i)
        for i, (csv_data, desc) in enumerate(sales_data)
    ]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(sales_processing_workflow)
    print(f"\nRun URL: {run.url}")
