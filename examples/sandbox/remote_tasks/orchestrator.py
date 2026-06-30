"""
Remote Task Orchestrator — Decoupled from Heavy Dependencies
=============================================================

This orchestrator references worker tasks via ``flyte.remote.Task.get()``
instead of importing them directly.  This means:

- **No shared code**: the orchestrator file never imports numpy, scipy,
  or sklearn — not even transitively.
- **Independent deploy**: ``compute_tasks.py`` and this file can be
  deployed, versioned, and scaled independently.
- **Tiny image**: the orchestrator container only needs ``pydantic-monty``.

The worker tasks are fetched from the Flyte control plane at deploy/run
time using ``auto_version="latest"``, so the orchestrator always picks
up the newest version of each worker.

Deploy the workers first, then the orchestrator::

    flyte deploy compute_tasks.py
    flyte deploy orchestrator.py

Install the optional dependency first::

    pip install 'flyte[sandbox]'
"""

import flyte
import flyte.remote

# ---------------------------------------------------------------------------
# Orchestrator environment — only needs pydantic-monty, no heavy deps.
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="lightweight-orchestrator",
    image=flyte.Image.from_debian_base().with_pip_packages("pydantic-monty"),
)

# ---------------------------------------------------------------------------
# Remote task references — fetched from the Flyte control plane.
#
# These are lazy handles: the actual task definitions are resolved at
# deploy/run time, not at import time.  The orchestrator never needs
# numpy, scipy, or sklearn installed.
# ---------------------------------------------------------------------------

generate_dataset = flyte.remote.Task.get("heavy-compute.generate_dataset", auto_version="latest")
fit_linear_model = flyte.remote.Task.get("heavy-compute.fit_linear_model", auto_version="latest")
compute_residuals = flyte.remote.Task.get("heavy-compute.compute_residuals", auto_version="latest")
detect_outliers = flyte.remote.Task.get("heavy-compute.detect_outliers", auto_version="latest")


# ---------------------------------------------------------------------------
# Sandboxed orchestrators — pure Python, zero heavy dependencies.
#
# Each call to a remote task pauses the sandbox, the Flyte controller
# dispatches the worker in its own container (with numpy/sklearn/etc.),
# and the sandbox resumes with the result.
# ---------------------------------------------------------------------------


@env.sandbox.orchestrator
def regression_pipeline(n_samples: int, noise: float, outlier_threshold: float) -> dict:
    """End-to-end regression: generate data -> fit -> residuals -> outliers."""

    # Step 1: Generate synthetic data (runs in heavy-compute container)
    dataset = generate_dataset(n_samples=n_samples, noise=noise)

    # Step 2: Fit a linear model (runs in heavy-compute container)
    model = fit_linear_model(dataset=dataset)

    # Step 3: Compute residuals (runs in heavy-compute container)
    residuals = compute_residuals(dataset=dataset, model=model)

    # Step 4: Detect outliers (runs in heavy-compute container)
    outlier_indices = detect_outliers(residuals=residuals, threshold=outlier_threshold)

    # Pure Python control flow — no imports needed
    n_outliers = len(outlier_indices)

    return {
        "model": model,
        "mean_abs_error": residuals["mean_abs_error"],
        "max_abs_error": residuals["max_abs_error"],
        "n_outliers": n_outliers,
        "outlier_indices": outlier_indices,
        "quality": "good" if model["r_squared"] > 0.9 else "poor",
    }


@env.sandbox.orchestrator
def adaptive_pipeline(n_samples: int) -> dict:
    """Fit a model, and only look for outliers if the fit is poor."""

    dataset = generate_dataset(n_samples=n_samples, noise=10.0)
    model = fit_linear_model(dataset=dataset)

    # Conditional: skip expensive outlier detection if the fit is good
    if model["r_squared"] > 0.95:
        return {
            "model": model,
            "skipped_outlier_detection": True,
            "reason": "R² already above 0.95",
        }

    # Fit is mediocre — investigate further
    residuals = compute_residuals(dataset=dataset, model=model)
    outlier_indices = detect_outliers(residuals=residuals, threshold=2.0)

    return {
        "model": model,
        "skipped_outlier_detection": False,
        "n_outliers": len(outlier_indices),
        "mean_abs_error": residuals["mean_abs_error"],
    }


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(regression_pipeline, n_samples=100, noise=10.0, outlier_threshold=2.0)
    print(r.url)
