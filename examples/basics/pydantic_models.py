"""Using Pydantic BaseModels as task inputs/outputs, including Optional[BaseModel].

Covers all complex field types: str, int, float, bool, Optional, List, Dict,
Enum, nested BaseModel, and combinations thereof.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel

import flyte
from flyte._image import DIST_FOLDER, PythonWheels

env = flyte.TaskEnvironment(
    name="ex-pydantic-models",
    image=flyte.Image.from_debian_base().clone(
        addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyte"),
        name="ex-pydantic-models-image",
    ),
)


class BatchMode(str, Enum):
    LINES = "lines"
    BYTES = "bytes"


class RetryPolicy(BaseModel):
    max_retries: int = 3
    backoff_seconds: float = 1.0
    retryable_codes: list[int] = [429, 500, 503]


class BatchConfig(BaseModel):
    # primitives
    name: str = "default"
    max_lines_per_batch: int = 100
    threshold: float = 0.95
    enabled: bool = True

    # enum
    batch_by: BatchMode = BatchMode.LINES

    # optional primitive
    max_lines_per_file: Optional[int] = None
    description: Optional[str] = None

    # list
    tags: list[str] = []
    weights: list[float] = []

    # dict
    metadata: dict[str, str] = {}
    limits: dict = {}

    # nested model
    retry: RetryPolicy = RetryPolicy()

    # optional nested model
    fallback_retry: Optional[RetryPolicy] = None

    # list of nested models
    extra_retries: list[RetryPolicy] = []

    # dict with nested model values
    per_stage_retry: dict[str, RetryPolicy] = {}

    # Literal
    mode: Literal["fast", "slow"] = "fast"


class ProcessingResult(BaseModel):
    total_lines: int
    batches_created: int
    config_used: BatchConfig


@env.task
def process_data(
    input_lines: list[str],
    batch_config: Optional[BatchConfig] = None,
) -> ProcessingResult:
    """Process input lines using the given batch configuration.

    ``batch_config`` is Optional[BaseModel] — the scenario that exercises
    Pydantic model serialization inside a Union type (Optional = Union[X, None]).
    """
    cfg = batch_config or BatchConfig()

    if cfg.batch_by == BatchMode.LINES:
        n_batches = max(1, len(input_lines) // cfg.max_lines_per_batch)
    else:
        total_bytes = sum(len(line.encode()) for line in input_lines)
        n_batches = max(1, total_bytes // (1024 * 1024))

    return ProcessingResult(
        total_lines=len(input_lines),
        batches_created=n_batches,
        config_used=cfg,
    )


@env.task
def run_pipeline(num_lines: int) -> ProcessingResult:
    lines = [f"line {i}" for i in range(num_lines)]
    cfg = BatchConfig(
        name="full-test",
        max_lines_per_batch=50,
        max_lines_per_file=500,
        threshold=0.8,
        enabled=True,
        batch_by=BatchMode.BYTES,
        description="integration test config",
        tags=["prod", "v2"],
        weights=[0.6, 0.4],
        metadata={"team": "infra", "env": "staging"},
        limits={"cpu": 4, "memory": 8192},
        retry=RetryPolicy(max_retries=5, backoff_seconds=2.0, retryable_codes=[429, 502]),
        fallback_retry=RetryPolicy(max_retries=1),
        extra_retries=[
            RetryPolicy(max_retries=2, backoff_seconds=0.5),
            RetryPolicy(max_retries=10, backoff_seconds=5.0),
        ],
        per_stage_retry={
            "fetch": RetryPolicy(max_retries=3),
            "transform": RetryPolicy(max_retries=1, backoff_seconds=0.1),
        },
        mode="slow",
    )
    return process_data(input_lines=lines, batch_config=cfg)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(run_pipeline, num_lines=200)
    print(run.name)
    print(run.url)
