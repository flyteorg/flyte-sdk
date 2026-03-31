# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flyte",
#   "scikit-learn>=1.3",
# ]
# ///

"""
Scikit-learn partial_fit with checkpoints
=========================================

``SGDClassifier.partial_fit`` is resumed across task retries by pickling the estimator to
``sgd_bundle.pkl``, matching ``huggingface_trainer_checkpoint.py``: resolve a bundle directory from
``await checkpoint.load.aio()``, call blocking :meth:`~flyte.AsyncCheckpoint.save` after each training
chunk (incremental training step), then ``await checkpoint.save.aio`` for the final state.

**Note:** ``uv run --script`` may install an older ``flyte`` from PyPI without
``TaskContext.checkpoint``. Run with a venv where this repository is installed
editable (``pip install -e .``), or pin a released SDK that includes checkpointing.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier

import flyte

env = flyte.TaskEnvironment(name="checkpoint_sklearn_partial")

BUNDLE_FILE = "sgd_bundle.pkl"


def bundle_path(root: pathlib.Path) -> pathlib.Path:
    direct = root / BUNDLE_FILE
    if direct.exists():
        return direct
    found = list(root.rglob(BUNDLE_FILE))
    return found[0] if found else direct


@env.task()
async def incremental_sgd(chunks: int = 4) -> float:
    ctx = flyte.ctx()
    assert ctx is not None
    ck = ctx.checkpoint
    assert ck is not None

    checkpoint_path: pathlib.Path | None = await ck.load.aio()
    if checkpoint_path is None:
        bundle_root = pathlib.Path("sklearn_partial")
    else:
        bundle_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent / "sklearn_partial"
    bundle_root.mkdir(parents=True, exist_ok=True)
    bpath = bundle_path(bundle_root)

    rng = np.random.default_rng(0)
    clf: SGDClassifier | None = None
    classes = np.array([0, 1])
    done = 0

    if bpath.exists():
        bundle = pickle.loads(bpath.read_bytes())
        clf = bundle["clf"]
        done = int(bundle["chunks_done"])

    for i in range(done, chunks):
        x = rng.standard_normal((32, 8))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        if clf is None:
            clf = SGDClassifier(max_iter=1, tol=None, random_state=0)
            clf.partial_fit(x, y, classes=classes)
        else:
            clf.partial_fit(x, y)

        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_bytes(pickle.dumps({"clf": clf, "chunks_done": i + 1}))
        ck.save(bpath)

    await ck.save.aio(bpath)

    assert clf is not None
    x_test = rng.standard_normal((64, 8))
    y_test = (x_test[:, 0] + x_test[:, 1] > 0).astype(int)
    return float(clf.score(x_test, y_test))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local").run(incremental_sgd, chunks=3)
    print(run.outputs())
