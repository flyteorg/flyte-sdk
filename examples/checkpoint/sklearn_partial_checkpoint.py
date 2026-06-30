"""
Scikit-learn partial_fit with checkpoints
=========================================

`sklearn.linear_model.SGDClassifier.partial_fit` is resumed across task retries by pickling the estimator to
`sgd_bundle.pkl`, matching `huggingface_trainer_checkpoint.py`: resolve a bundle directory from
`await checkpoint.load()`, `await checkpoint.save` after each training
chunk (incremental training step).

**Note:** `uv run --script` may install an older `flyte` from PyPI without
`flyte.models.TaskContext.checkpoint`. Run with a venv where this repository is installed
editable (`pip install -e .`), or pin a released SDK that includes checkpointing.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_sklearn_partial",
    image=flyte.Image.from_debian_base().with_pip_packages("scikit-learn"),
    resources=flyte.Resources(cpu="2", memory="1Gi"),
)

BUNDLE_FILE = "sgd_bundle.pkl"
RETRIES = 3


def bundle_path(root: pathlib.Path) -> pathlib.Path:
    direct = root / BUNDLE_FILE
    if direct.exists():
        return direct
    found = list(root.rglob(BUNDLE_FILE))
    return found[0] if found else direct


@env.task(retries=RETRIES)
async def incremental_sgd(chunks: int = 4) -> float:
    assert chunks > RETRIES
    ctx = flyte.ctx()
    assert ctx is not None
    cp = ctx.checkpoint
    assert cp is not None

    prev_cp_path: pathlib.Path | None = await cp.load()
    prev = None if prev_cp_path is None else prev_cp_path.read_bytes()
    if prev:
        # load clf and chunks_done from previous checkpoint
        bundle = pickle.loads(prev)
        chunks_start = bundle["chunks_done"]
        clf = bundle["clf"]
    else:
        chunks_start = 0
        clf = SGDClassifier(max_iter=1, tol=None, random_state=0)

    # create local checkpoint directory
    bundle_path = pathlib.Path("sklearn_partial") / "sgd_bundle.pkl"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    classes = np.array([0, 1])

    failure_interval = chunks // RETRIES
    print(f"Start at: {chunks_start} of {chunks} chunks")
    for i in range(chunks_start, chunks):
        x = rng.standard_normal((32, 8))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        clf.partial_fit(x, y, classes=classes)

        if i > chunks_start and i % failure_interval == 0:
            # Simulate a failure at a regular interval
            raise RuntimeError(f"Failed at iteration {i}, failure_interval {failure_interval}.")

        # save checkpoint to object storage, which can be loaded by await cp.load() at the top of the script on the
        bundle_path.write_bytes(pickle.dumps({"clf": clf, "chunks_done": i + 1}))
        await cp.save(bundle_path)

    assert clf is not None
    x_test = rng.standard_normal((64, 8))
    y_test = (x_test[:, 0] + x_test[:, 1] > 0).astype(int)
    return float(clf.score(x_test, y_test))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(incremental_sgd, chunks=10)
    print(run.url)
