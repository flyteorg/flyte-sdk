"""
Serve a GGUF **Model artifact** with llama.cpp as a **native sidecar in a Flyte task pod** --
for batch/pipeline inference against a co-located model, as opposed to the standalone,
scale-to-zero Flyte App in `llamacpp_app_fuse.py`.

The task pod runs two containers:
  * `primary` -- the client task; it calls the local server on `localhost`.
  * `llama`   -- a llama.cpp server (llama-server via the `llama-cpp-fserve` shim) started
                 with the plugin's `build_fserve_command` -- the *same* argv
                 `LlamaCppAppEnvironment` runs, so tuning/flags stay identical.

The server is a **native sidecar**: an init container with `restart_policy="Always"`, so
Kubernetes starts it *before* the primary and SIGTERMs it when the primary exits.

Two TaskEnvironment shapes, one `--reuse` flag
----------------------------------------------
  * **ephemeral** (default): a fresh pod per run. The model is (cold) loaded every run.
  * **reusable actor** (`--reuse`): a warm pool (`flyte.ReusePolicy`) keeps the pod -- with
    the loaded model -- alive across runs, so only the first call pays the cold mount+load.
    Reuse needs the `unionai-reuse` package in the client image (it rewrites the primary
    entrypoint to `unionai-actor-bridge`); it is baked into `CLIENT_IMAGE` unconditionally so
    the same image serves both modes.

Both modes use the **same module-scope `chat` task** so the runtime can resolve it; the per-run
sidecar pod template (and, for `--reuse`, the reuse policy) is injected at submit time via
`chat.override(pod_template=..., reusable=...)` -- `override` takes both.

Model delivery -- artifact over object-store FUSE
-------------------------------------------------
Because the sidecar starts before the primary, it cannot wait for the primary to download a
task input -- the weights must be present at sidecar startup. So the model is served **in
place over object-store FUSE**, the same mechanism as `llamacpp_app_fuse.py`: a versioned
Model **artifact** materialized in the data bucket, read through a read-only PVC that exposes
the data-bucket root. `__main__` resolves the artifact to its object-store URI at submit time
and passes the mounted path to `build_fserve_command` as `model_dir` (the shim finds the
concrete `.gguf`). See `llamacpp_app_fuse.py` for the RO PVC prerequisite and the gcsfuse
(GKE) / Mountpoint-S3 (EKS) manifests.

The same artifact is also bound as the task's `model` input, so the resulting **Run records
Run -> artifact lineage** (which artifact version this inference consumed) -- the input literal
carries the artifact id; the bytes still arrive via the fuse mount, not a download.

Everything is env-configurable (`LLAMACPP_*`), defaulting to a small CPU model. Set
`LLAMACPP_GPU` (e.g. `L4:1`) to give the **sidecar container** a GPU (request==limit) + a GPU
toleration + a CUDA image + `--n-gpu-layers` offload -- so the same file serves a large model
on a GPU. Flyte does NOT inject accelerator affinity for a sidecar-container GPU the way it
does for `Resources.gpu` on the primary, so on some clouds you may also need a node selector
(GKE `gke-accelerator`) -- see the toleration note below.

`flyteplugins.llamacpp` is imported lazily (submit-time helpers / __main__), never at module
scope: the `chat` task runs in the client image (flyte + openai only) and the runtime imports
this module to resolve `chat`, so a module-level plugin import would crash the task container.

Run:
    python examples/genai/llamacpp/llamacpp_sidecar.py --prompt "Write a haiku about GPUs."
    python examples/genai/llamacpp/llamacpp_sidecar.py --reuse --prompt "..."   # warm actor
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import flyte
from flyte.io import Dir

# Model + artifact identity, env-configurable (defaults to a small CPU model). To serve a large
# model, point these at it and set LLAMACPP_GPU below -- e.g. unsloth/Qwen3.8-27B-GGUF / Q8_0.
MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "q4_k_m")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")
MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-0.5b-instruct")
SERVER_PORT = 8080

# Where the read-only, data-bucket-root PVC is mounted into the sidecar. The artifact's key
# under the bucket resolves at `<MODEL_MOUNT>/<key>`.
MODEL_MOUNT = "/tmp/models"
# The pre-provisioned RO PVC over the data-bucket root (dataplane helm release). Required: it
# must name a claim that exists in the task's namespace -- see llamacpp_app_fuse.py / README.md.
MODEL_PVC = os.getenv("LLAMACPP_MODEL_PVC", "flyte-metadata-ro")

# GPU for the sidecar (default CPU). Set e.g. "L4:1" to request a GPU on the llama container,
# build a CUDA image, and offload all layers. IMPORTANT for GPU serving: llama-server defaults
# to CPU, so LLAMACPP_EXTRA_ARGS defaults to `--n-gpu-layers 999` when a GPU is requested.
GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "")

# The client task's own image: an OpenAI client + `unionai-reuse` (the actor bridge, baked in
# unconditionally so one image serves both ephemeral and --reuse runs; unused when ephemeral).
CLIENT_IMAGE = flyte.Image.from_debian_base(
    name="llamacpp-sidecar-client", install_flyte=True
).with_pip_packages("openai", "unionai-reuse")

# Module-scope environment + task so the runtime can resolve `chat`. The sidecar pod template
# (built image URI + resolved model dir, known only at submit time) and, for --reuse, the reuse
# policy are injected per-run via `chat.override(...)` in __main__.
env = flyte.TaskEnvironment(
    name="llamacpp-sidecar",
    image=CLIENT_IMAGE,
    resources=flyte.Resources(cpu="2", memory="6Gi"),
)


def _gpu_count(gpu: str | None) -> int:
    """Parse the `<FAMILY>:<count>` accelerator string to a count (e.g. 'L4:2' -> 2)."""
    if not gpu:
        return 0
    return int(gpu.split(":", 1)[1]) if ":" in gpu else 1


def _pod_template(serve_image_uri: str, model_dir: str) -> flyte.PodTemplate:
    """primary (client) + a llama.cpp server sidecar reading the model from the RO model PVC."""
    from flyteplugins.llamacpp import build_fserve_command
    from kubernetes.client.models import (
        V1Container,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1ResourceRequirements,
        V1Toleration,
        V1Volume,
        V1VolumeMount,
    )

    # The same llama-cpp-fserve argv the App runs; build_fserve_command returns shell-safe
    # tokens meant to be joined into one shell string (how fserve execs llama-server), so
    # wrap them in `sh -c`. model_dir points at the artifact's directory under the fuse mount.
    server_cmd = build_fserve_command(
        model_id=MODEL_ID, port=SERVER_PORT, model_dir=model_dir, extra_args=EXTRA_ARGS.split()
    )

    # GPU goes on the SIDECAR container (not the primary). Extended resources (nvidia.com/gpu)
    # require request == limit, so set both. Flyte only injects accelerator affinity/tolerations
    # for a `Resources.gpu` on the primary -- for a sidecar-container GPU we add the GPU
    # toleration ourselves (Karpenter/NAP then provision a GPU node for the nvidia.com/gpu
    # request). On GKE you may also need a `gke-accelerator` node selector to pin the L4/L40s pool.
    resources = None
    tolerations = None
    if GPU:
        gpu_res = {"nvidia.com/gpu": str(_gpu_count(GPU))}
        resources = V1ResourceRequirements(limits=gpu_res, requests=dict(gpu_res))
        tolerations = [V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]

    llama_sidecar = V1Container(
        name="llama",
        image=serve_image_uri,
        restart_policy="Always",  # native sidecar: started before, torn down with, the primary
        command=["/bin/sh", "-c", " ".join(server_cmd)],
        resources=resources,
        volume_mounts=[V1VolumeMount(name="model", mount_path=MODEL_MOUNT, read_only=True)],
    )
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            init_containers=[llama_sidecar],
            volumes=[
                V1Volume(
                    name="model",
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=MODEL_PVC, read_only=True
                    ),
                )
            ],
            tolerations=tolerations,
        ),
        # Required on gcsfuse (GKE) so the sidecar injector mounts the volume; ignored on
        # Mountpoint-S3 (EKS), so it is harmless to leave in for portability.
        annotations={"gke-gcsfuse/volumes": "true"},
    )


@flyte.trace
def _await_server_ready(base_url: str) -> str:
    """Block until the llama.cpp sidecar answers -- i.e. it has mounted the model over FUSE and
    finished loading it. `@flyte.trace` records this as its own timed sub-action, so the model
    cold-start (fuse first-touch + load) shows up separately from inference in the run timeline.
    """
    import time

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="sk-noauth")
    for _ in range(150):
        try:
            client.models.list()
            return "ready"
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama.cpp sidecar did not become ready")


@flyte.trace
def _complete(base_url: str, prompt: str) -> str:
    """One OpenAI-compatible chat completion. Traced so inference latency is a separate timed
    sub-action from the server cold-start."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="sk-noauth")
    resp = client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": prompt}], max_tokens=256
    )
    return resp.choices[0].message.content or ""


@env.task
def chat(prompt: str, model: Dir) -> str:
    """Call the local llama.cpp sidecar (OpenAI-compatible) once it is ready.

    `model` is bound to the Model artifact purely to record **Run -> artifact lineage**: the
    run's input literal carries the artifact's id, so the platform ties this run to the exact
    artifact version it served. The bytes are not read here -- the sidecar serves them from the
    fuse mount, whose path was baked into the pod template at submit time.

    The two phases are `@flyte.trace`d helpers grouped under `flyte.group("llm-serve")`, so their
    timings surface as sub-actions in the run's UI: sidecar cold-start vs the inference call.
    In --reuse mode the first call pays the cold-start; subsequent calls hit the warm actor.
    """
    base_url = f"http://localhost:{SERVER_PORT}/v1"
    with flyte.group("llm-serve"):
        _await_server_ready(base_url)
        return _complete(base_url, prompt)


def _artifact_model_dir(uri: str) -> str:
    """Map an artifact's object-store URI to its directory under the bucket-root fuse mount.

    The RO PVC exposes the data-bucket root, so `<scheme>://<bucket>/<key>` is read at
    `<MODEL_MOUNT>/<key>`; the shim then finds the concrete `.gguf` in that directory.
    """
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    return f"{MODEL_MOUNT.rstrip('/')}/{key}"


if __name__ == "__main__":
    import argparse
    import asyncio

    import flyte.prefetch
    from flyte.remote import Artifact

    p = argparse.ArgumentParser(description="llama.cpp as a Flyte task-pod sidecar over a Model artifact.")
    p.add_argument("--prompt", default="Write a haiku about GPUs.")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Use a reusable actor (warm pod) instead of an ephemeral pod; fires twice to show the warm speedup.",
    )
    p.add_argument("--config", help="Path to the Flyte config (else UCTL_CONFIG / the default).")
    p.add_argument("--project", help="Project to run in (must route to the dataplane with the RO model PVC).")
    p.add_argument("--domain", help="Domain to run in (default: the config's).")
    args = p.parse_args()

    flyte.init_from_config(path_or_config=args.config, project=args.project, domain=args.domain)

    # 1. A Flyte run creates the Model artifact (one quant, kept by allow_patterns), published
    #    as a versioned artifact in the data bucket -- the same bucket the RO PVC mounts. Sized
    #    for the small default; bump via env (e.g. LLAMACPP_PREFETCH_DISK=60Gi) for a large model.
    run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(
            cpu=os.getenv("LLAMACPP_PREFETCH_CPU", "2"),
            memory=os.getenv("LLAMACPP_PREFETCH_MEMORY", "4Gi"),
            disk=os.getenv("LLAMACPP_PREFETCH_DISK", "10Gi"),
        ),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
    run.wait()

    # 2. Resolve the artifact to its object-store URI, then to the mounted directory the sidecar
    #    reads. Pinning the version here keeps the batch run reproducible.
    artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))
    uri = asyncio.run(artifact.to_python(Dir)).path
    model_dir = _artifact_model_dir(uri)
    print(f"Serving artifact from fuse mount: {uri} -> {model_dir}")

    # 3. Sidecar images must be string URIs, so build the llama.cpp image and use its URI. Build
    #    a CUDA image when a GPU is requested; a CPU image otherwise. The image carries
    #    llama-server + the `llama-cpp-fserve` shim (installs flyteplugins-llamacpp).
    from flyteplugins.llamacpp import build_llama_cpp_image

    serve_image = build_llama_cpp_image(name="llama-cpp-sidecar", cuda=bool(GPU))
    built = asyncio.run(flyte.build.aio(serve_image))
    print(f"llama.cpp sidecar image ({'cuda' if GPU else 'cpu'}): {built.uri}")

    # 4. Inject the per-run sidecar pod template onto the module-scope task and run it. --reuse
    #    also attaches a ReusePolicy (warm actor). The artifact is passed as the `model` input so
    #    the Run records Run -> artifact lineage; the sidecar serves the same artifact from FUSE.
    pod_template = _pod_template(built.uri, model_dir)
    if args.reuse:
        # Warm actor: replicas=1/concurrency=1 = a single warm llama-server, one request at a
        # time. First call cold-loads the model into the actor; later calls reuse it, so fire
        # twice to make the speedup observable.
        task = chat.override(
            pod_template=pod_template,
            reusable=flyte.ReusePolicy(replicas=1, concurrency=1, idle_ttl=600, scaledown_ttl=600),
        )
        for i in range(2):
            run = flyte.run(task, prompt=args.prompt, model=artifact)
            print(f"[reuse call {i + 1}/2] {run.url}")
            run.wait()
            print(run.outputs())
    else:
        task = chat.override(pod_template=pod_template)
        run = flyte.run(task, prompt=args.prompt, model=artifact)
        print(f"Run: {run.url}")
        run.wait()
        print(run.outputs())
