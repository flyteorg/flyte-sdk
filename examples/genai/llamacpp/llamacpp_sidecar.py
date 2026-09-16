"""
Serve a GGUF Model artifact with llama.cpp as a native sidecar in a Flyte task pod -- for
batch/pipeline inference against a co-located model, as opposed to the standalone scale-to-zero
App in `llamacpp_app_fuse.py`.

The pod runs the client (`primary`) plus a llama.cpp server sidecar (`llama`) started with the
plugin's `build_fserve_command` (same argv as the App). The sidecar is a native init container
(`restart_policy="Always"`), so k8s starts it before the primary and stops it after.

Two shapes via `--reuse`:
  * ephemeral (default): fresh pod per run; the model cold-loads each time.
  * reusable actor (`--reuse`): a warm pool keeps the loaded model across runs, so only the
    first call pays the load.
Both use the same module-scope `chat` task; the per-run sidecar pod template (and, for --reuse,
the reuse policy) are injected at submit time via `chat.override(...)`.

Model delivery: the sidecar starts before the primary, so the weights must be present at startup
-- served in place over object-store FUSE (same as `llamacpp_app_fuse.py`), from a read-only PVC
over the data-bucket root. The artifact is also bound as the `model` input so the Run records
Run -> artifact lineage. See `llamacpp_app_fuse.py` for the RO PVC prerequisite + manifests.

Env-configurable (`LLAMACPP_*`), small CPU model by default; set `LLAMACPP_GPU` (e.g. `L4:1`) to
give the sidecar a GPU + CUDA image + layer offload. `flyteplugins.llamacpp` is imported lazily
(never at module scope) so the client task container -- which only has flyte + openai -- doesn't
crash importing it.

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

MODEL_MOUNT = "/tmp/models"  # where the RO PVC is mounted in the sidecar; artifact key resolves under it
# Pre-provisioned RO PVC over the data-bucket root (managed: flyte-metadata-ro). See llamacpp_app_fuse.py.
MODEL_PVC = os.getenv("LLAMACPP_MODEL_PVC", "flyte-metadata-ro")

# GPU for the sidecar (default CPU). Set e.g. "L4:1" for a GPU; EXTRA_ARGS then offloads all layers.
GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "")

# Client image: OpenAI client + unionai-reuse (the actor bridge for --reuse; harmless otherwise).
CLIENT_IMAGE = flyte.Image.from_debian_base(name="llamacpp-sidecar-client", install_flyte=True).with_pip_packages(
    "openai", "unionai-reuse"
)

# Module-scope so the runtime can resolve `chat`; the sidecar pod template + reuse policy are
# injected per-run via `chat.override(...)` in __main__.
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

    # Same argv as the App; build_fserve_command returns shell tokens, so wrap in `sh -c`.
    server_cmd = build_fserve_command(
        model_id=MODEL_ID, port=SERVER_PORT, model_dir=model_dir, extra_args=EXTRA_ARGS.split()
    )

    # GPU goes on the SIDECAR container: nvidia.com/gpu needs request==limit + a GPU toleration
    # (Flyte injects those only for a primary `Resources.gpu`).
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
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=MODEL_PVC, read_only=True),
                )
            ],
            tolerations=tolerations,
        ),
        # gcsfuse (GKE) mounts only with this annotation; Mountpoint-S3 (EKS) ignores it.
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

    def _artifact_exists(name: str) -> bool:
        try:
            Artifact.get(name)
            return True
        except Exception:
            return False

    # Reuse the artifact; prefetch only when missing or LLAMACPP_FORCE_PREFETCH is set
    # (bump LLAMACPP_PREFETCH_DISK for a large model).
    force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
    if force or not _artifact_exists(ARTIFACT_NAME):
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
    else:
        print(f"Reusing artifact {ARTIFACT_NAME!r} (LLAMACPP_FORCE_PREFETCH=1 to re-create)")

    # Resolve the artifact to the mounted dir the sidecar reads (pinning the version keeps the run reproducible).
    artifact: Artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    uri = asyncio.run(artifact.to_python(Dir)).path
    model_dir = _artifact_model_dir(uri)
    print(f"Serving artifact from fuse mount: {uri} -> {model_dir}")

    # Sidecar images are string URIs, so build the llama.cpp image (CUDA if GPU) and use its URI.
    from flyteplugins.llamacpp import build_llama_cpp_image

    serve_image = build_llama_cpp_image(name="llama-cpp-sidecar", cuda=bool(GPU))
    built = asyncio.run(flyte.build.aio(serve_image))  # type: ignore[arg-type, var-annotated]  # ty: ignore[invalid-argument-type]
    print(f"llama.cpp sidecar image ({'cuda' if GPU else 'cpu'}): {built.uri}")

    # Inject the per-run sidecar pod template (+ a ReusePolicy for --reuse); `model` is passed as
    # an input for Run -> artifact lineage.
    pod_template = _pod_template(built.uri, model_dir)
    if args.reuse:
        # Warm actor: first call cold-loads, later calls reuse -- fire twice to show the speedup.
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
