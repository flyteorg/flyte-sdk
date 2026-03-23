# /// script
# requires-python = "==3.12"
# dependencies = [
#    "scikit-learn==1.6.1",
#    "pandas",
#    "pyarrow",
#    "flyte>=2.0.0b3",
# ]
# ///


import asyncio

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import flyte

worker = flyte.TaskEnvironment(
    "worker",
    image=flyte.Image.from_uv_script(
        __file__,
        name="flyte",
        registry="ghcr.io/flyteorg",
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


@worker.task
async def train(features: list[str], drop: str) -> float:
    features.remove(drop)

    X, y = fetch_california_housing(as_frame=True, return_X_y=True)

    fold = KFold(n_splits=5, random_state=42, shuffle=True)

    model = LinearRegression()

    scores = cross_val_score(estimator=model, X=X[features], y=y, cv=fold, scoring="r2")

    score = float(scores.mean())

    return score


@worker.task
async def rfe():  # -> list[dict[str, float]]:
    x, _y = fetch_california_housing(as_frame=True, return_X_y=True)

    features = list(x.columns)

    out: list[dict[str, float]] = []

    for i in range(len(features) - 1):
        with flyte.group(f"iteration-{i}"):
            runs = {feature: train(list(features), drop=feature) for feature in features}

            values = await asyncio.gather(*(runs[feature] for feature in runs))
            scores = dict(zip(runs.keys(), values))
            best = max(scores, key=scores.get)

            print(f"Iteration {i}: Best feature to drop is '{best}' resulting in a score {scores[best]:.4f}")

            features.remove(best)

            out.append(scores)

    # return out


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(rfe)
    print(run.url)

    # Run with:
    # uv run --prerelease=allow examples/ml/rfe.py
