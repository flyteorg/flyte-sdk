# /// script
# requires-python = "==3.12"
# dependencies = [
#    "anyio",
#    "aioresult",
#    "flyte>=2.0.0b6",
# ]
# ///
from dataclasses import dataclass
from typing import List

import aioresult
import anyio
from pydantic import BaseModel

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    name="ex-dataclasses",
    image=flyte.Image.from_uv_script(__file__, registry="ghcr.io/flyteorg", name="flyte"),
)


@dataclass
class InferenceRequest:
    feature_a: float
    feature_b: float


@env.task
async def predict_one(request: InferenceRequest) -> float:
    """
    A dummy linear model: prediction = 2 * feature_a + 3 * feature_b + bias(=1.0)
    """
    return 2.0 * request.feature_a + 3.0 * request.feature_b + 1.0


@dataclass
class BatchRequest:
    requests: List[InferenceRequest]


class BatchPredictionResults(BaseModel):
    predictions: List[float]
    results_file: File

    class Config:
        arbitrary_types_allowed = True


@env.task
async def predict_batch(batch: BatchRequest) -> BatchPredictionResults:
    """
    Runs the same dummy linear model on each element in the batch.
    """
    # tasks = [predict_one(request=req) for req in batch.requests]
    # results = await asyncio.gather(*tasks)
    captured_results_obj = []
    async with anyio.create_task_group() as tg:
        # Start multiple tasks that run at the same time
        for req in batch.requests:
            captured_results_obj.append(aioresult.ResultCapture.start_soon(tg, predict_one, req))

    results = [r.result() for r in captured_results_obj]

    # 2) Write predictions to a local CSV
    import csv

    output_path = "predictions.csv"
    with open(output_path, mode="w", newline="") as f:  # noqa: ASYNC230
        writer = csv.writer(f)
        writer.writerow(["prediction"])
        for p in results:
            writer.writerow([p])

    csv_file = await File.from_local(output_path)

    return BatchPredictionResults(predictions=results, results_file=csv_file)


@env.task
async def avg_from_file(results: BatchPredictionResults) -> float:
    """
    Reads the CSV in results.results_file, computes the average of the 'prediction' column.
    """
    total = 0.0
    count = 0
    async with results.results_file.open() as f:
        iter_f = iter(f)
        next(iter_f)  # Skip header
        for row in iter_f:
            total += float(row)
            count += 1

    return total / count if count else 0.0


@env.task
async def dc_wf(batch: BatchRequest):
    """
    Runs the batch prediction and computes the average of the predictions.
    """
    results = await predict_batch(batch=batch)
    avg = await avg_from_file(results=results)

    print(f"Average prediction: {avg}")


if __name__ == "__main__":
    # result_one = asyncio.run(predict_one(InferenceRequest(feature_a=1.0, feature_b=2.0)))
    # print(f"Prediction for single request: {result_one}")
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    # Can run programmatically
    # run = flyte.run(predict_one, InferenceRequest(feature_a=1.0, feature_b=2.0))
    # print(run.url)
    # Or through the CLI
    # flyte run basics/dataclass_examples.py predict_one --request '{"feature_a": 1, "feature_b": 2}'

    batch_req = BatchRequest(
        requests=[
            InferenceRequest(feature_a=1.0, feature_b=2.0),
            InferenceRequest(feature_a=3.0, feature_b=4.0),
            InferenceRequest(feature_a=5.0, feature_b=6.0),
        ]
    )
    # run = flyte.run(predict_batch, batch_req)
    # print(run.url)

    # Run in local mode to allow File writing.
    # run = flyte.with_runcontext(mode="local").run(dc_wf, batch_req)
    # run = flyte.run(dc_wf, batch_req)
    x = flyte.with_runcontext(mode="remote").run(dc_wf, batch_req)

    print(f"Run URL: {x}")
