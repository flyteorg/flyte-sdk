"""Example: DataFrame analysis with constraints and reusable tasks (LLM approach).

Demonstrates:
- Passing pd.DataFrame directly (auto-converted to CSV File)
- Using constraints to enforce business rules
- Using base_packages for specific libraries
- Using as_task() for reusable execution
- Multiple typed outputs including File
- max_sample_rows to control data sampling size
"""

import logging
from pathlib import Path

import flyte
import pandas as pd
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

agent = AutoCoderAgent(
    name="sensor-analysis",
    base_packages=["numpy"],
    max_sample_rows=30,
)

env = flyte.TaskEnvironment(
    name="batch-processing-agent-example",
    secrets=[
        flyte.Secret(key="niels_openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-codegen",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
                pre=True,
            ),
            name="dataframe-analysis",
        )
    ).with_pip_packages("pyarrow"),
    depends_on=[sandbox_environment],
)


@env.task(cache="auto")
def build_sensor_data() -> pd.DataFrame:
    """Create sample IoT sensor data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "sensor_id": ["S1", "S2", "S3", "S4", "S5"] * 10,
            "temperature": [22.1, 23.5, 19.8, 25.0, 21.3, 24.7, 18.9, 26.2, 20.5, 22.8] * 5,
            "humidity": [45.0, 52.3, 38.1, 60.5, 42.0, 55.2, 35.8, 63.1, 40.0, 48.7] * 5,
            "pressure": [
                1013.2,
                1012.8,
                1014.1,
                1011.5,
                1013.0,
                1012.5,
                1014.3,
                1011.0,
                1013.5,
                1012.0,
            ]
            * 5,
            "status": ["normal", "normal", "warning", "normal", "normal"] * 10,
        }
    )


@env.task
async def analyze_sensor_data(
    prompt: str, sensor_df: pd.DataFrame
) -> dict[str, File | int | str | list[str] | dict[str, int]]:
    """Generate analysis code, then create a reusable task for future data."""
    result = await agent.generate.aio(
        prompt=prompt,
        samples={"readings": sensor_df},
        constraints=[
            "Temperature values must be between -40 and 60 Celsius",
            "Humidity values must be between 0 and 100 percent",
            "Output report must have one row per unique sensor_id",
        ],
        outputs={
            "report": File,
            "total_anomalies": int,
        },
    )

    if not result.success:
        return {"error": result.error, "attempts": result.attempts}

    task = result.as_task(
        name="run_sensor_analysis",
        resources=flyte.Resources(cpu=1, memory="512Mi"),
    )

    report, total_anomalies = await task.aio(
        readings=result.original_samples["readings"],
    )

    return {
        "report": report,
        "total_anomalies": total_anomalies,
        "attempts": result.attempts,
        "packages": result.detected_packages,
        "tokens": {
            "input": result.total_input_tokens,
            "output": result.total_output_tokens,
        },
    }


@env.task
async def dataframe_analysis_workflow(
    prompt: str = """Analyze IoT sensor data. For each sensor, calculate mean/min/max temperature,
mean humidity, and count warnings.

Output a summary CSV (report) with columns: sensor_id, mean_temp, min_temp, max_temp,
mean_humidity, warning_count. Also output total_anomalies as the total warning count.""",
) -> dict[str, File | int | str | list[str] | dict[str, int]]:
    """Analyze sensor data: generate code from DataFrame, run with reusable task."""
    sensor_df = build_sensor_data()
    return await analyze_sensor_data(prompt=prompt, sensor_df=sensor_df)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(dataframe_analysis_workflow)
    print(f"\nRun URL: {run.url}")
