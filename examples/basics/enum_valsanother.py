from typing import Literal

import flyte

Intensity = Literal["low", "medium", "high"]
env = flyte.TaskEnvironment(name="enum_vals")


@env.task
def literal_task(i: Intensity) -> str:
    return f"Intensity is {i}"


@env.task
def main() -> str:
    some = literal_task()
    return "he"
