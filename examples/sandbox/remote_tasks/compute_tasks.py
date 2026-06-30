"""
Heavy Compute Tasks
====================

Worker tasks that use heavy third-party libraries (numpy, scipy,
scikit-learn). Each task runs in its own container with these
dependencies installed.

Deploy this file independently::

    flyte deploy compute_tasks.py

The orchestrator in ``orchestrator.py`` references these tasks via
``flyte.remote.Task.get()`` — it never imports numpy, scipy, or sklearn.
"""

import numpy as np
from scipy import stats
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

import flyte

env = flyte.TaskEnvironment(
    name="heavy-compute",
    image=(
        flyte.Image.from_debian_base().with_pip_packages(
            "numpy",
            "scipy",
            "scikit-learn",
        )
    ),
)


@env.task
def generate_dataset(n_samples: int, noise: float) -> dict:
    """Generate a synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise,
        random_state=42,
    )
    return {
        "x_values": X.flatten().tolist(),
        "y_values": y.tolist(),
    }


@env.task
def fit_linear_model(dataset: dict) -> dict:
    """Fit a linear regression and return coefficients."""
    X = np.array(dataset["x_values"]).reshape(-1, 1)
    y = np.array(dataset["y_values"])

    model = LinearRegression()
    model.fit(X, y)

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(X, y)),
    }


@env.task
def compute_residuals(dataset: dict, model: dict) -> dict:
    """Calculate prediction residuals."""
    x = np.array(dataset["x_values"])
    y = np.array(dataset["y_values"])
    predicted = model["slope"] * x + model["intercept"]
    residuals = (y - predicted).tolist()

    return {
        "residuals": residuals,
        "mean_abs_error": float(np.mean(np.abs(y - predicted))),
        "max_abs_error": float(np.max(np.abs(y - predicted))),
    }


@env.task
def detect_outliers(residuals: dict, threshold: float) -> list:
    """Flag data points whose residual exceeds *threshold*."""
    r = np.array(residuals["residuals"])
    z_scores = np.abs(stats.zscore(r))
    outlier_indices = np.where(z_scores > threshold)[0].tolist()
    return outlier_indices


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
