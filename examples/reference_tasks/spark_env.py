"""
This example demonstrates how to use the Spark environment in Flyte.
"""

from dataclasses import dataclass

import flyte

env = flyte.TaskEnvironment(name="spark_env")


@dataclass
class InferenceRequest:
    feature_a: float
    feature_b: float


@env.task
async def spark_task(x: str) -> InferenceRequest:
    return InferenceRequest(feature_a=1.0, feature_b=2.0)


@env.task
async def spark_task2(r: InferenceRequest) -> float:
    return r.feature_a + r.feature_b
