"""
Bad-spec fast-fail.

A misconfigured clustered task must FAIL FAST with a clear diagnosis instead of looping restarts
forever. There are two distinct guards, at two different layers:

  (A) Runtime / cluster errors (this example, runnable): the pod can't even start — e.g. an image
      that doesn't exist. On the backend, the plugin runs ``DemystifyFailedOrPendingPod`` against
      rank-0 (``<jobset>-workers-0-0``) and, once pod-0 has been Pending/Failed past the grace
      window, surfaces ``BadTaskSpecification`` (a non-retryable error) rather than counting the
      failure against ``max_restarts`` and retrying indefinitely.

  (B) SDK validation errors (compile-time, NOT runnable): caught in
      ``ClusteredTaskEnvironment.__post_init__`` before anything is submitted — e.g.
      ``nproc_per_node`` greater than the GPU count, ``replicas < 1``, an unknown ``runtime``. These
      raise ``ValueError`` / ``TypeError`` locally. See the commented block at the bottom and the
      corresponding asserts in ``tests/flyte/clustered/test_env.py``.

Run (expect a fast BadTaskSpecification, no retry storm):
    uv run python examples/clustered/bad_spec.py
"""

from __future__ import annotations

import flyte
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun

# An image reference that will never pull — the pods go ImagePullBackOff and rank-0 never starts.
# (Use a clearly bogus tag so it can't accidentally resolve to a real image.)
BOGUS_IMAGE = "ghcr.io/unionai/this-image-does-not-exist:never-built-deadbeef"

env = ClusteredTaskEnvironment(
    name="bad_spec_env",
    image=BOGUS_IMAGE,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
    replicas=2,
    nproc_per_node=1,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),
    # Even with a generous restart budget, image-pull failures should NOT consume it — the plugin
    # short-circuits to BadTaskSpecification once pod-0 is stuck.
    failure_policy=ClusterFailurePolicy(max_restarts=3),
)


@env.task
async def never_runs() -> str:
    """This body is never reached — the container image fails to pull first."""
    return "unreachable"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(never_runs)
    print("Run URL:", run.url)
    run.wait()
    # Expect a failed phase with a BadTaskSpecification-style reason, reached quickly (no retry loop).
    print("Final phase:", run.phase)


# --- (B) SDK validation guards — these raise locally at construction time, before any submission. --
# Uncomment any of these to see the immediate error (they never reach the cluster):
#
#   # nproc_per_node (2) exceeds the GPU count (1):
#   ClusteredTaskEnvironment(
#       name="bad", image="x", replicas=1, nproc_per_node=2,
#       resources=flyte.Resources(gpu="A10G:1"),
#   )  # -> ValueError: resources.gpu (1) must be >= nproc_per_node (2)
#
#   # replicas must be >= 1:
#   ClusteredTaskEnvironment(name="bad", image="x", replicas=0, nproc_per_node=1)
#   # -> ValueError: replicas must be >= 1
