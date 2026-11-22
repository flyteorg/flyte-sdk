"""
vLLM with a three-tier KV cache (LMCache) and a Valkey-backed prefix-aware router.

Tiers, per worker:

- L0: GPU memory, vLLM's automatic prefix cache
- L1: pinned CPU memory, LMCache ``local_cpu``
- L2: a shared Valkey, LMCache's remote backend; every worker reads what any worker wrote

Flyte apps have no per-replica address, so each worker is its own single-replica app and a
FastAPI router app (``router.py``) sits in front of them. The router keeps its index
(prefix block -> worker, session -> worker, in-flight load) in the same Valkey.

Setup
-----

Create a secret with the Valkey URL, using LMCache's native ``resp://`` scheme::

    flyte create secret lmcache-valkey-url resp://<valkey-host>:6379

Deploy
------

::

    python examples/genai/vllm_lmcache/serve.py

This prefetches the model into the Flyte object store, then serves the router together with
its workers. See README.md for benchmarking.
"""

import logging
import pathlib

from flyteplugins.vllm import DEFAULT_VLLM_IMAGE, VLLMAppEnvironment
from router import app as router_app

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

MODEL_HF = "Qwen/Qwen3-8B"
MODEL_ID = "qwen3-8b"
NUM_WORKERS = 3
VALKEY_SECRET = "lmcache-valkey-url"
LMCACHE_VERSION = "0.5.5"

# lmcache 0.5.5 ships CUDA 13 wheels, which matches the vLLM 0.26 line in DEFAULT_VLLM_IMAGE.
worker_image = DEFAULT_VLLM_IMAGE.clone(name="vllm-lmcache-image").with_pip_packages(f"lmcache=={LMCACHE_VERSION}")

worker = VLLMAppEnvironment(
    name="lmcache-worker",
    model_hf_path=MODEL_HF,
    model_id=MODEL_ID,
    image=worker_image,
    # L1 lives in pinned host memory, so the pod needs memory well above LMCACHE_MAX_LOCAL_CPU_SIZE.
    resources=flyte.Resources(cpu="8", memory="72Gi", gpu="L40s:1", disk="80Gi", shm="auto"),
    stream_model=True,
    extra_args=[
        "--max-model-len",
        "32768",
        "--enable-prefix-caching",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ],
    env_vars={
        # vLLM seeds the start of its block-hash chain from PYTHONHASHSEED and falls back to
        # random bytes without it. Every worker must use the same seed, or no worker can read
        # another's L2 entries (or its own after a restart).
        "PYTHONHASHSEED": "0",
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "40",  # GB of L1
        "LMCACHE_REMOTE_SERDE": "naive",
        "LMCACHE_PRE_CACHING_HASH_ALGORITHM": "sha256_cbor",
    },
    # For a password-protected Valkey, add a secret mapped to LMCACHE_RESP_PASSWORD.
    secrets=flyte.Secret(VALKEY_SECRET, as_env_var="LMCACHE_REMOTE_URL"),
    # Scaling a worker to zero throws away its L0/L1, so keep each one warm.
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    requires_auth=True,
)

workers = [worker.clone_with(name=f"lmcache-worker-{i}") for i in range(NUM_WORKERS)]

router_image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "fastapi",
    "uvicorn",
    "httpx",
    "valkey",
    "transformers>=5",
    "jinja2",
    "flyteplugins-vllm",
    pre=True,
)


def router_env(workers: list[VLLMAppEnvironment]) -> FastAPIAppEnvironment:
    return FastAPIAppEnvironment(
        name="lmcache-router",
        app=router_app,
        image=router_image,
        resources=flyte.Resources(cpu="4", memory="8Gi"),
        env_vars={
            "WORKER_APPS": ",".join(w.name for w in workers),
            "TOKENIZER": MODEL_HF,
            "ROUTER_NAMESPACE": f"default:{MODEL_ID}",
            "ROUTE_MODE": "prefix",
        },
        secrets=flyte.Secret(VALKEY_SECRET, as_env_var="VALKEY_URL"),
        scaling=flyte.app.Scaling(replicas=(1, 2)),
        depends_on=workers,
        requires_auth=True,
    )


router = router_env(workers)


if __name__ == "__main__":
    import flyte.prefetch

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent, log_level=logging.INFO)

    run = flyte.prefetch.hf_model(repo=MODEL_HF)
    run.wait()
    print(f"prefetched {MODEL_HF}: {run.url}")

    # Every worker must load the model from the same path: LMCache puts the model name into
    # its L2 keys, so workers with different paths would not share L2.
    prefetched = [
        w.clone_with(
            name=w.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            model_hf_path=None,
        )
        for w in workers
    ]
    app = flyte.serve(router_env(prefetched))
    print(f"router: {app.url}")
