# /// script
# requires-python = "==3.13"
# dependencies = [
#    "optuna>=4.0.0,<5.0.0",
#    "flyte>=0.2.0b17",
#    "scikit-learn==1.7.0",
#    "unionai-reuse",
# ]
# ///

import asyncio
import typing
from collections import Counter
from typing import Optional, Union

import optuna
from optuna import Trial
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

import flyte
import flyte.errors

driver = flyte.TaskEnvironment(
    name="driver",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="optimizer",
        registry="ghcr.io/flyteorg",
        platform=("linux/amd64", "linux/arm64"),
    ),
    reusable=flyte.ReusePolicy(replicas=(1, 5), idle_ttl=300),
)


class Optimizer:
    def __init__(
        self,
        objective: callable,
        n_trials: int,
        concurrency: int = 1,
        delay: float = 0.1,
        study: Optional[optuna.Study] = None,
        log_delay: float = 0.1,
    ):
        self.n_trials: int = n_trials
        self.concurrency: int = concurrency
        self.objective: typing.Callable = objective
        self.delay: float = delay
        self.log_delay = log_delay

        self.study = study if study else optuna.create_study()

    async def log(self):
        while True:
            await asyncio.sleep(self.log_delay)

            counter = Counter()

            for trial in self.study.trials:
                counter[trial.state.name.lower()] += 1

            counts = dict(counter, queued=self.n_trials - len(self))

            # print items in dictionary in a readable format
            formatted = [f"{name}: {count}" for name, count in counts.items()]
            print(f"{'    '.join(formatted)}")

    async def spawn(self, semaphore: asyncio.Semaphore):
        async with semaphore:
            trial: Trial = self.study.ask()

            try:
                print("Starting trial", trial.number)

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 1.0),
                }

                output = await self.objective(params)

                self.study.tell(trial, output, state=optuna.trial.TrialState.COMPLETE)
            except flyte.errors.RuntimeUserError as e:
                print(f"Trial {trial.number} failed: {e}")

                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

            await asyncio.sleep(self.delay)

    async def __call__(self):
        # create semaphore to manage concurrency
        semaphore = asyncio.Semaphore(self.concurrency)

        # create list of async trials
        trials = [self.spawn(semaphore) for _ in range(self.n_trials)]

        logger: Optional[asyncio.Task] = None
        if self.log_delay:
            logger = asyncio.create_task(self.log())

        # await all trials to complete
        await asyncio.gather(*trials)

        if self.log_delay and logger:
            logger.cancel()
            try:
                await logger
            except asyncio.CancelledError:
                pass

    def __len__(self) -> int:
        """Return the number of trials in history."""
        return len(self.study.trials)


@driver.task
async def objective(params: dict[str, Union[int, float]]) -> float:
    data = load_iris()
    X, y = shuffle(data.data, data.target, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )

    # Use cross-validation to evaluate performance
    score = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()

    return score.item()


@driver.task
async def optimize(
    n_trials: int = 20,
    concurrency: int = 5,
    delay: float = 0.05,
    log_delay: float = 0.1,
) -> dict[str, Union[int, float]]:
    optimizer = Optimizer(
        objective=objective,
        n_trials=n_trials,
        concurrency=concurrency,
        delay=delay,
        log_delay=log_delay,
        study=optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)),
    )

    await optimizer()

    best = optimizer.study.best_trial

    print("✅ Best Trial")
    print("  Number :", best.number)
    print("  Params :", best.params)
    print("  Score  :", best.value)

    return best.params


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(optimize, 100, 10)
    print(run.url)
