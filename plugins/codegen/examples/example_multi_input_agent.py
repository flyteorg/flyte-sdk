"""Example: Multi-input data join with primitive inputs (Agent SDK approach).

Demonstrates:
- Multiple data inputs in a single call (two DataFrames)
- Explicit inputs mixing data (File) with primitives (float, bool)
- image_config for custom image settings
- Agent SDK mode (use_agent_sdk=True)
"""

import logging
from pathlib import Path

import flyte
import pandas as pd
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent, ImageConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

agent = AutoCoderAgent(
    name="multi-input-join-agent",
    model="claude-sonnet-4-5-20250929",
    use_agent_sdk=True,
    image_config=ImageConfig(python_version=(3, 11)),
)


env = flyte.TaskEnvironment(
    name="multi-input-agent-example",
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
        .with_pip_packages(
            "git+https://github.com/flyteorg/flyte-sdk.git@86f88fece16d956e28667d3f0d8d49108c8cdd68"
        )
    ).with_pip_packages("pyarrow"),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def build_orders() -> pd.DataFrame:
    """Create sample order data."""
    return pd.DataFrame(
        {
            "order_id": range(1, 11),
            "product_id": ["P1", "P2", "P3", "P1", "P2", "P3", "P1", "P2", "P3", "P1"],
            "quantity": [2, 1, 5, 3, 2, 1, 4, 6, 2, 1],
            "unit_price": [
                10.0,
                25.0,
                5.0,
                10.0,
                25.0,
                5.0,
                10.0,
                25.0,
                5.0,
                10.0,
            ],
        }
    )


@env.task(cache="auto")
def build_products() -> pd.DataFrame:
    """Create sample product catalog."""
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "name": ["Widget", "Gadget", "Doohickey"],
            "category": ["electronics", "electronics", "accessories"],
            "in_stock": [True, True, False],
        }
    )


@env.task
async def join_and_analyze_with_agent(
    prompt: str, orders: pd.DataFrame, products: pd.DataFrame
) -> tuple[File, File, float]:
    """Join two datasets using Agent SDK, filter by threshold, produce report dir."""
    result = await agent.generate.aio(
        prompt=prompt,
        samples={"orders": orders, "products": products},
        inputs={
            "orders": File,
            "products": File,
            "min_order_value": float,
            "include_out_of_stock": bool,
        },
        outputs={
            "product_summary.csv": File,
            "overall_summary.csv": File,
            "grand_total": float,
        },
    )

    if not result.success:
        return {"error": result.error, "attempts": result.attempts}

    task = result.as_task(name="run_multi_input_join_agent")
    product_summary, overall_summary, grand_total = await task.aio(
        orders=result.original_samples["orders"],
        products=result.original_samples["products"],
        min_order_value=15.0,
        include_out_of_stock=False,
    )

    return product_summary, overall_summary, grand_total


@env.task
async def multi_input_agent_workflow(
    prompt: str = """Join orders with products on product_id and analyze sales.

Calculate total_price (quantity * unit_price) per order. Filter out orders below
min_order_value. If include_out_of_stock is false, exclude out-of-stock products.

Output a product_summary.csv (product_id, name, category, total_orders, total_revenue)
and an overall_summary.csv (metric, value). Also output grand_total as the total revenue
across all filtered orders.""",
) -> tuple[File, File, float]:
    """Join two datasets with Agent SDK: multiple data inputs, primitive args."""
    orders = build_orders()
    products = build_products()
    return await join_and_analyze_with_agent(prompt=prompt, orders=orders, products=products)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(multi_input_agent_workflow)
    print(f"\nRun URL: {run.url}")
