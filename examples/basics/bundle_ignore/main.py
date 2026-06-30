# bundle_ignore/main.py
#
# When Flyte runs a task remotely it packages your source tree into a code
# bundle and ships it to the worker.  By default every file that isn't
# already excluded by .gitignore ends up in that bundle — including large
# files or secrets that are committed to the repo but are not needed at
# runtime.
#
# A .flyteignore file (placed next to your source code) lets you exclude
# those tracked files from the bundle without removing them from git.
# Patterns follow standard .gitignore syntax and are applied to ALL files,
# regardless of whether they are tracked or untracked.
#
# This example ships alongside a .flyteignore file that excludes:
#   - training_data/   (large CSVs committed for local dev, loaded from a
#                       remote store at runtime instead)
#   - docs/            (documentation, not needed inside the container)
#   - *.ipynb          (exploration notebooks, not needed at runtime)
#
# Run locally:
#   flyte run examples/basics/bundle_ignore/main.py train
#
# The bundle will only contain main.py — the three excluded paths stay on
# disk but never enter the tar archive that gets shipped to the worker.

import flyte

env = flyte.TaskEnvironment(
    name="bundle_ignore_example_test",
    resources=flyte.Resources(memory="250Mi"),
)


@env.task
async def train(n_samples: int = 1000) -> float:
    """Simulate a training step.

    In a real workflow the training data would be passed as a flyte.io.File
    or Dir input, not read from a relative path — so it never needs to be
    in the bundle.
    """
    import random

    random.seed(42)
    samples = [random.gauss(0, 1) for _ in range(n_samples)]
    mean = sum(samples) / len(samples)
    print(f"Trained on {n_samples} synthetic samples, mean={mean:.4f}")
    return mean


@env.task
async def evaluate(mean: float) -> str:
    result = "pass" if abs(mean) < 0.1 else "fail"
    print(f"Evaluation: mean={mean:.4f} → {result}")
    return result


@env.task
async def pipeline(n_samples: int = 1000) -> str:
    mean = await train(n_samples=n_samples)
    return await evaluate(mean=mean)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline)
    print(run.url)
    run.wait()
