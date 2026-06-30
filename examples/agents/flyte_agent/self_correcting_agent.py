"""Self-correcting Data Science agent that right-sizes heterogeneous compute.

This example shows how to drive a :class:`flyte.ai.agents.Agent` whose tools are
``@env.task`` Flyte tasks with *very different* compute profiles:

- ``generate_dataset``  — memory-bound (allocates a dense ``n_samples x n_features`` matrix).
- ``engineer_features`` — memory- and CPU-bound (polynomial feature expansion blows up fast).
- ``train_model``       — CPU-bound (ensembles parallelize across cores).

A single static ``flyte.Resources`` request can't serve all three well: pick it
small and the big jobs OOM; pick it big and you waste a cluster's worth of RAM on
the small ones. So instead of hard-coding resources, this agent **right-sizes
every tool call at runtime** and **recovers from OOM**.

The mechanism is a ``call_handler`` passed to the ``@tool`` decorator. A
``call_handler`` is an async callback ``(call_llm, tool_fn, **kwargs)`` that the
agent runs *in place of* the default tool execution, so on every invocation it:

1. Calls ``call_llm`` (the agent's own LLM callback) with the tool name, its
   docstring, and the concrete call arguments, asking the model to emit a JSON
   object describing the compute it needs (``{"cpu": ..., "memory": "...", "disk": "..."}``).
2. Turns that JSON into a :class:`flyte.Resources` and applies it via
   ``tool_fn.target.override(resources=...)`` *before* the task runs
   (``tool_fn.target`` is the underlying ``@env.task``).
3. Wraps the call in a ``try/except flyte.errors.OOMError`` loop: if the task is
   killed for running out of memory, it doubles the memory request and retries,
   up to ``max_oom_retries`` times.

Because the handler receives ``call_llm`` and the tool's ``model`` directly from
the agent, the tools no longer need a back-reference to the ``Agent`` instance.

This keeps the small jobs cheap, gives the big jobs room to breathe, and lets the
pipeline heal itself when the first estimate is too low.

Run::

    pip install 'flyte' litellm numpy scikit-learn
    python examples/agents/flyte_agent/self_correcting_agent.py
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import flyte
import flyte.errors
from flyte.ai.agents import Agent, LLMCallable, ToolFn, tool
from flyte.io import File

MODEL = "claude-haiku-4-5"

env = flyte.TaskEnvironment(
    name="self-correcting-ds-agent",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "litellm",
        "numpy",
        "scikit-learn",
    ),
    # Deliberately small defaults: the agent is expected to size *up* per call.
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


# ---------------------------------------------------------------------------
# Resource right-sizing helpers
# ---------------------------------------------------------------------------

RESOURCE_SIZING_SYSTEM_PROMPT = """\
You are a Kubernetes capacity planner for Flyte tasks. Given a task's name, its \
docstring, and the concrete arguments it is about to be called with, estimate the \
*minimum sensible* compute it needs to finish without being killed, while not \
wildly over-provisioning.

Reason about the work implied by the arguments:
- Memory-bound tasks (allocating large matrices) scale with the product of size \
arguments (e.g. n_samples * n_features) and with expansions like polynomial degree.
- CPU-bound tasks (training ensembles) benefit from more cores; scale CPU with the \
number of estimators / amount of data.

Respond with ONLY a JSON object (no prose, no code fences) with any of these keys:
  - "cpu":    a number of cores, e.g. 1, 2, 4
  - "memory": a Kubernetes memory string, e.g. "2Gi", "8Gi"
  - "disk":   a Kubernetes disk string, e.g. "10Gi" (omit unless large I/O)
Omit a key to accept the default. Do not include any other keys. No GPUs are \
available on this cluster.

Example response: {"cpu": 2, "memory": "4Gi"}
"""

# Floor applied to every estimate so a bad LLM guess can't under-provision below
# something that can at least start up.
RESOURCE_FLOOR = flyte.Resources(cpu=1, memory="1Gi")

_ALLOWED_RESOURCE_KEYS = ("cpu", "memory", "disk", "shm")
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$")


def _extract_json(text: str | None) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM reply."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resources_from_spec(spec: dict[str, Any], floor: flyte.Resources) -> flyte.Resources:
    """Merge an LLM-produced spec onto the floor, keeping only known keys."""
    kwargs: dict[str, Any] = {
        "cpu": floor.cpu,
        "memory": floor.memory,
        "gpu": floor.gpu,
        "disk": floor.disk,
        "shm": floor.shm,
    }
    for key in _ALLOWED_RESOURCE_KEYS:
        value = spec.get(key)
        if value in (None, "", "null"):
            continue
        kwargs[key] = value
    try:
        return flyte.Resources(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive against bad model output
        flyte.logger.warning("Invalid resource spec %s (%s); falling back to floor.", spec, exc)
        return floor


def _bump_memory(resources: flyte.Resources, factor: float = 2.0) -> flyte.Resources:
    """Return a copy of *resources* with its memory request multiplied by *factor*."""
    mem = resources.memory if isinstance(resources.memory, str) else None
    if not mem:
        return dataclasses.replace(resources, memory="2Gi")
    match = _MEM_RE.match(mem)
    if not match:
        return dataclasses.replace(resources, memory="4Gi")
    value = float(match.group(1))
    unit = match.group(2) or "Mi"
    return dataclasses.replace(resources, memory=f"{max(1, int(value * factor))}{unit}")


async def _right_size(
    call_llm: LLMCallable, model: str, tool_name: str, description: str, args: dict[str, Any]
) -> flyte.Resources:
    """Ask the LLM to size the compute for a single tool call."""
    user = json.dumps({"tool": tool_name, "description": description, "arguments": args}, default=str)
    try:
        reply = await call_llm(
            model,
            RESOURCE_SIZING_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
            None,
        )
        spec = _extract_json(reply.content)
    except Exception as exc:  # pragma: no cover - never let sizing break the tool
        flyte.logger.warning("Resource right-sizing LLM call failed (%s); using floor.", exc)
        spec = {}
    resources = _resources_from_spec(spec, RESOURCE_FLOOR)
    flyte.logger.info("right-size %s%s -> %s", tool_name, tuple(args.values()), resources)
    return resources


# ---------------------------------------------------------------------------
# The right-sizing call handler
# ---------------------------------------------------------------------------


def right_sizing_handler(*, max_oom_retries: int = 2):
    """Build a ``@tool`` ``call_handler`` that right-sizes and self-heals.

    The returned handler, on every call, (1) uses ``call_llm`` to estimate the
    ``flyte.Resources`` the task needs for the given arguments, (2) applies them via
    ``tool_fn.target.override(resources=...)``, and (3) retries on
    :class:`flyte.errors.OOMError` with doubled memory, up to ``max_oom_retries``.

    The handler is generic: it works for any tool whose ``target`` is a
    :class:`~flyte.TaskTemplate`, and it leans on the agent-provided ``call_llm``
    and ``tool_fn.model`` rather than capturing the :class:`Agent` itself.

    Args:
        max_oom_retries: How many times to double memory and retry after an OOM.
    """

    async def handle(call_llm: LLMCallable, tool_fn: ToolFn, **kwargs: Any) -> Any:
        resources = await _right_size(call_llm, tool_fn.model, tool_fn.name, tool_fn.description, kwargs)

        attempt = 0
        while True:
            try:
                with flyte.group(f"{tool_fn.name}-attempt-{attempt + 1}"):
                    sized_task = tool_fn.target.override(resources=resources)
                    return await sized_task.aio(**kwargs)
            except flyte.errors.OOMError:
                if attempt >= max_oom_retries:
                    flyte.logger.error("%s OOMed after %d retries; giving up.", tool_fn.name, attempt)
                    raise
                resources = _bump_memory(resources)
                attempt += 1
                flyte.logger.warning("%s OOMed; retrying with memory=%s", tool_fn.name, resources.memory)

    return handle


# A single stateless handler instance is shared across every tool below.
right_size = right_sizing_handler(max_oom_retries=2)


# ---------------------------------------------------------------------------
# Tools — Flyte tasks with heterogeneous compute profiles.
#
# Each task is wrapped with ``@tool(call_handler=right_size)`` so that the agent
# routes every invocation through the right-sizing handler defined above.
#
# Data is passed between tools as remote URI *strings* (not File objects),
# because the base Agent harness stringifies tool results back into the
# transcript: the model can only forward plain text between tool calls.
# ---------------------------------------------------------------------------


@tool(call_handler=right_size)
@env.task
async def generate_dataset(n_samples: int, n_features: int, noise: float = 10.0) -> str:
    """Generate a synthetic regression dataset and persist it as a compressed .npz file.

    This task is memory-bound: it materializes a dense float64 matrix of roughly
    ``n_samples * n_features * 8`` bytes in RAM (plus a transient copy during
    generation), so peak memory grows with the product of the two arguments.

    Args:
        n_samples: Number of rows (observations) to generate.
        n_features: Number of columns (features) per row.
        noise: Standard deviation of gaussian noise applied to the target.

    Returns:
        The remote URI of the saved ``.npz`` file (keys ``X`` and ``y``). Pass
        this string to ``engineer_features`` or ``train_model``.
    """
    import os
    import tempfile

    import numpy as np
    from sklearn.datasets import make_regression

    flyte.logger.info("generate_dataset: n_samples=%d n_features=%d", n_samples, n_features)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)

    local_path = os.path.join(tempfile.mkdtemp(), "dataset.npz")
    np.savez_compressed(local_path, X=X.astype("float64"), y=y.astype("float64"))
    uploaded = await File.from_local(local_path)
    return uploaded.path


@tool(call_handler=right_size)
@env.task
async def engineer_features(dataset_uri: str, degree: int = 2) -> str:
    """Expand features with polynomial + interaction terms.

    This task is both memory- and CPU-bound: ``PolynomialFeatures`` grows the
    column count combinatorially with ``degree``, so a modest input matrix can
    explode into something far larger in memory.

    Args:
        dataset_uri: URI returned by ``generate_dataset``.
        degree: Polynomial degree (2 = pairwise interactions, 3 = triples, ...).

    Returns:
        The remote URI of a new ``.npz`` dataset with the expanded feature matrix.
    """
    import os
    import tempfile

    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures

    local = await File.from_existing_remote(dataset_uri).download()
    data = np.load(local)
    X, y = data["X"], data["y"]
    flyte.logger.info("engineer_features: input shape=%s degree=%d", X.shape, degree)

    X_poly = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
    flyte.logger.info("engineer_features: expanded to shape=%s", X_poly.shape)

    out_path = os.path.join(tempfile.mkdtemp(), "features.npz")
    np.savez_compressed(out_path, X=X_poly.astype("float64"), y=y.astype("float64"))
    uploaded = await File.from_local(out_path)
    return uploaded.path


@tool(call_handler=right_size)
@env.task
async def train_model(
    dataset_uri: str,
    model_type: str = "random_forest",
    n_estimators: int = 200,
    max_depth: int = 12,
) -> dict:
    """Train a regression model and return evaluation metrics.

    This task is CPU-bound: tree ensembles fit each estimator in parallel across
    available cores, so wall-clock time drops as CPU is added. Memory grows with
    the dataset size and the number/depth of trees.

    Args:
        dataset_uri: URI returned by ``generate_dataset`` or ``engineer_features``.
        model_type: One of ``"random_forest"`` or ``"gradient_boosting"``.
        n_estimators: Number of trees in the ensemble.
        max_depth: Maximum depth of each tree.

    Returns:
        A dict with ``model_type``, ``n_samples``, ``n_features``, ``r2``, and ``rmse``.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    local = await File.from_existing_remote(dataset_uri).download()
    data = np.load(local)
    X, y = data["X"], data["y"]
    flyte.logger.info("train_model: %s shape=%s n_estimators=%d", model_type, X.shape, n_estimators)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model_type": model_type,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "rmse": round(float(mean_squared_error(y_test, preds) ** 0.5), 4),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
You are a senior ML engineer agent. You build and evaluate regression models by \
orchestrating Flyte tasks as tools.

Workflow:
1. Call generate_dataset with the requested sizes to create the data.
2. If the user asks for richer features or interactions, call engineer_features on \
the resulting dataset URI.
3. Call train_model on the (possibly engineered) dataset URI.
4. Report the final metrics (r2, rmse) clearly, and mention the dataset/model shape.

Pass dataset URIs returned by one tool as the dataset_uri argument of the next. \
You do not need to think about compute resources — the runtime sizes each task for \
you and retries automatically if it runs out of memory.
"""

agent = Agent(
    name="self-correcting-ds-agent",
    instructions=INSTRUCTIONS,
    model=MODEL,
    tools=[generate_dataset, engineer_features, train_model],
    max_turns=15,
)


@env.task(report=True)
async def run_ds_agent(request: str) -> str:
    """Drive the self-correcting Data Science agent inside a durable Flyte task."""
    result = await agent.run.aio(request, memory=[])
    return result.summary or result.error or ""


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        run_ds_agent,
        request=(
            "Build a regression model on a dataset with 20000 samples and 40 features. "
            "Add degree-2 polynomial features first, then train a random forest with "
            "30 trees and report r2 and rmse."
        ),
    )
    print(f"Run URL: {run.url}")
    run.wait()
    print(run.outputs()[0])
