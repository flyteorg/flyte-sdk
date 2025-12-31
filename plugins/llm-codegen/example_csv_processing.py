import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from flyteplugins.llm_codegen import code_gen_environment, code_gen_eval

import flyte
from flyte._image import PythonWheels
from flyte.io import File

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("flyteplugins.code_generation.code_gen_eval").setLevel(logging.INFO)

env = flyte.TaskEnvironment(
    name="csv-processing-example",
    secrets=[
        flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base().clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent / "dist",
            package_name="flyteplugins-code-generation",
            pre=True,
        ),
        name="csv-processing-example",
    ),
    depends_on=[code_gen_environment],
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
async def generate_and_run(sales_data: list[list[str]]) -> list[Optional[dict]]:
    prompt = """Process sales data and calculate total revenue.

Return JSON with: total_revenue (float), total_units (int), transaction_count (int)"""

    async def process_format(csv_data: str, description: str, index: int) -> dict:
        """Generate code and run for a single CSV format."""
        # Create temp CSV file
        csv_file = Path(tempfile.gettempdir()) / f"sales_data_{index}.csv"
        csv_file.write_text(csv_data)

        # Generate code with automatic LLM sampling from the data
        result = await code_gen_eval.aio(
            name=f"sales-data-{index}",
            prompt=prompt,
            data={"csv_data": await File.from_local(str(csv_file))},
            outputs={
                "total_revenue": float,
                "total_units": int,
                "transaction_count": int,
            },
        )

        if not result.success:
            print(f"{description}: Code generation failed")
            return None

        with flyte.group(f"sales-data-{index}"):
            # Run generated code on the original data
            outputs = await result.run.aio()

        if outputs["exit_code"] != 0:
            print(f"{description}: Execution failed (exit code {outputs['exit_code']})")
            return None

        return outputs

    semaphore = asyncio.Semaphore(3)

    async def process_with_limit(csv_data: str, description: str, index: int):
        async with semaphore:
            return await process_format(csv_data, description, index)

    tasks = [
        process_with_limit(csv_data, desc, i)
        for i, (csv_data, desc) in enumerate(sales_data)
    ]

    result = await asyncio.gather(*tasks)
    return result


@env.task
async def sales_processing_workflow() -> list[Optional[dict]]:
    """Process multiple CSV formats: generate code with LLM sampling, then execute."""
    sales_data = prepare_sales_data()
    return await generate_and_run(sales_data=sales_data)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(sales_processing_workflow)
    print(f"\nRun URL: {run.url}")
