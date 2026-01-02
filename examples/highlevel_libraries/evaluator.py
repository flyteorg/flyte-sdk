import asyncio
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Protocol, overload, runtime_checkable

from eval_reporting import generate_evaluation_report
from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(
    "evaluator",
    reusable=flyte.ReusePolicy(replicas=1, concurrency=10),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.7", "plotly"),
)


class Dataset(BaseModel):
    """Input dataset for evaluation."""

    input: str
    context: str
    response: str


class ReferenceDataset(BaseModel):
    """Reference/ground truth dataset for evaluation."""

    input: str
    output: str


@runtime_checkable
class EvaluationMetric(Protocol):
    """
    Protocol for async evaluation functions that score model outputs.

    Evaluation functions can operate with or without reference data.
    They return a JSON-serializable dictionary containing string metrics.
    All evaluation functions are async to enable parallel execution.
    """

    @overload
    async def __call__(self, actual: Dataset) -> Dict[str, str]:
        """
        Evaluate without reference data.

        Args:
            actual: The dataset containing model input and output

        Returns:
            Dictionary containing evaluation metrics (e.g., {"score": "0.85", "reasoning": "..."})
        """
        ...

    @overload
    async def __call__(self, actual: Dataset, reference: ReferenceDataset) -> Dict[str, str]:
        """
        Evaluate with reference data.

        Args:
            actual: The dataset containing model input and output
            reference: The reference/ground truth data

        Returns:
            Dictionary containing evaluation metrics (e.g., {"accuracy": "0.9", "f1": "0.88"})
        """
        ...

    async def __call__(self, actual: Dataset, reference: Optional[ReferenceDataset] = None) -> Dict[str, str]:
        """
        Evaluate model output with optional reference data (async).

        Args:
            actual: The dataset containing model input and output
            reference: Optional reference/ground truth data

        Returns:
            Dictionary containing evaluation metrics
        """
        ...


@runtime_checkable
class AggregationMetric(Protocol):
    """
    Protocol for aggregation functions that combine individual evaluation results.

    Takes all evaluation results and produces a final aggregated score.
    All values are strings for JSON serialization.
    """

    def __call__(self, eval_results: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Aggregate evaluation results into a final score.

        Args:
            eval_results: List of evaluation results from individual samples

        Returns:
            Dictionary containing aggregated metrics (e.g., {"mean_score": "0.82", "std": "0.12"})
        """
        ...


@env.task(short_name="single-evaluation")
async def _run_single_eval(
    eval_func: EvaluationMetric, dataset: Dataset, reference_dataset: Optional[ReferenceDataset] = None
) -> Dict[str, str]:
    """Run a single evaluation on one dataset item."""
    if reference_dataset is not None:
        return await eval_func(dataset, reference_dataset)
    else:
        return await eval_func(dataset)


@env.task(report=True, short_name="batch-evaluator")
async def _run_batch_eval(
    dataset: List[Dataset],
    eval_func: EvaluationMetric,
    aggregate_func: Optional[AggregationMetric] = None,
    reference_dataset: Optional[List[ReferenceDataset]] = None,
) -> Dict[str, str]:
    """Run evaluation on a batch of datasets in parallel and generate visualization report."""
    # Create tasks for all evaluations to run in parallel
    tasks = []
    for i, actual in enumerate(dataset):
        if reference_dataset is not None:
            task = _run_single_eval(eval_func, actual, reference_dataset[i])
        else:
            task = _run_single_eval(eval_func, actual)
        tasks.append(task)

    # Execute all evaluations in parallel
    eval_results = await asyncio.gather(*tasks)

    # Prepare output
    output: Dict[str, str] = {"num_samples": str(len(dataset))}

    # Apply aggregation if provided
    aggregated = None
    if aggregate_func is not None:
        aggregated = aggregate_func(eval_results)
        for key, value in aggregated.items():
            output[f"aggregated_{key}"] = value

    # Generate visualization report
    await generate_evaluation_report(eval_results, aggregated, len(dataset))

    return output


@dataclass
class Evaluator:
    """
    LLM Evaluation framework for running evaluations on datasets.

    Evaluates model outputs using a provided evaluation function and optionally
    aggregates results using a scoring function.
    """

    eval_func: EvaluationMetric
    aggregate_func: Optional[AggregationMetric] = None

    async def run(
        self,
        dataset: List[Dataset],
        reference_dataset: Optional[List[ReferenceDataset]] = None,
        mode: Literal["local", "remote"] = "local",
    ) -> Dict[str, str]:
        """
        Run evaluation loop over all inputs in parallel.

        All evaluations are executed concurrently using asyncio.gather for maximum performance.

        Args:
            dataset: List of datasets to evaluate
            reference_dataset: Optional list of reference datasets (must match dataset size)
            mode: Either "local" or "remote" - local runs in current process, remote runs on Flyte

        Returns:
            Dictionary containing aggregated results with string values

        Raises:
            ValueError: If reference_dataset size doesn't match dataset size
        """
        if reference_dataset is not None and len(reference_dataset) != len(dataset):
            raise ValueError(
                f"Reference dataset size ({len(reference_dataset)}) must match dataset size ({len(dataset)})"
            )

        print(f"Mode = {mode}")
        await flyte.init_from_config.aio(root_dir=pathlib.Path(__file__).parent)
        r = await flyte.with_runcontext(mode=mode, copy_style="all").run.aio(
            _run_batch_eval, dataset, self.eval_func, self.aggregate_func, reference_dataset
        )
        print(f"Evaluation results: {r.url}")
        await r.wait.aio()
        outputs = await r.outputs.aio()
        if outputs is None:
            raise RuntimeError(f"Failed to get outputs from evaluation")
        return outputs[0]
