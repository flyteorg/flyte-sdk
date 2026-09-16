"""
Serve a GGUF Model artifact with llama.cpp as a native sidecar in a Flyte task pod, delivering
the model from a Union Volume over the uvol mount broker -- the task-pod counterpart of the
standalone App in `llamacpp_app_union_volume.py`, and the Union-Volume counterpart of the
object-store-PVC sidecar in `llamacpp_sidecar.py`.

Steps -- prefetch Artifact -> build Union Volume -> serve alongside a client:
  1. `flyte.prefetch.hf_model` publishes the weights as a versioned Model artifact.
  2. `build_model_volume` copies that artifact into a fresh Volume and commits it to an immutable
     ROVolume locator (one build fans out to many read-only readers).
  3. The `chat` task pod runs a client (`primary`) plus a llama.cpp server sidecar (`llama`). The
     sidecar runs `union-volume-exec`, which mounts the ROVolume read-only through the broker and
     runs `build_fserve_command` against it; the client calls it over localhost.

Delivery contrast with `llamacpp_sidecar.py`: that mounts the weights in place over an object-store
RO PVC; this pulls them from a committed Union Volume, brokered zero-privilege (no CAP_SYS_ADMIN,
no /dev/fuse, no hostPath, no pre-provisioned PVC). And unlike the App variant, a task pod never
goes through Knative, so this needs none of the Knative volume feature flags.

Two shapes via `--reuse`:
  * ephemeral (default): fresh pod per run; the model cold-loads each time.
  * reusable actor (`--reuse`): a warm pool keeps the loaded model across runs, so only the first
    call pays the mount + load.

Env-configurable (`LLAMACPP_*`), small CPU model by default; set `LLAMACPP_GPU` (e.g. `L4:1`) to
give the sidecar a GPU + CUDA image + layer offload. `LLAMACPP_PREBUILT=1` uses the official
prebuilt llama.cpp image instead of compiling from source (much faster/lighter image build).
`flyteplugins.*` is imported lazily (never at module scope) so the client task container -- flyte +
openai only -- never imports it.

Run:
    python examples/genai/llamacpp/llamacpp_sidecar_union_volume.py --prompt "Write a haiku about GPUs."
    python examples/genai/llamacpp/llamacpp_sidecar_union_volume.py --reuse --prompt "..."   # warm actor
"""

from __future__ import annotations

import os

import flyte
from flyte.io import Dir

MODEL_REPO = os.getenv("LLAMACPP_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
QUANT = os.getenv("LLAMACPP_QUANT", "q4_k_m")
ARTIFACT_NAME = os.getenv("LLAMACPP_ARTIFACT_NAME", "qwen2-5-0-5b-instruct-q4-k-m")
MODEL_ID = os.getenv("LLAMACPP_MODEL_ID", "qwen2.5-0.5b-instruct")
SERVER_PORT = 8080
MODEL_MOUNT = "/tmp/models"  # where union-volume-exec mounts the volume in the sidecar; --model-dir points here
SIDECAR_NAME = "llama"

# GPU for the sidecar (default CPU). Set e.g. "L4:1" for a GPU; EXTRA_ARGS then offloads all layers.
GPU = os.getenv("LLAMACPP_GPU", "") or None
EXTRA_ARGS = os.getenv("LLAMACPP_EXTRA_ARGS", "--n-gpu-layers 999 --flash-attn on" if GPU else "--ctx-size 8192")

# Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
FLYTEPLUGINS_UNION = "flyteplugins-union>=0.11.0b0"

# Client image: OpenAI client + unionai-reuse (the actor bridge for --reuse; harmless otherwise).
# Deliberately no flyteplugins-union -- the client only talks HTTP to the sidecar.
CLIENT_IMAGE = flyte.Image.from_debian_base(name="llamacpp-sidecar-client", install_flyte=True).with_pip_packages(
    "openai", "unionai-reuse"
)

# Builder needs the Volume client; its pod template (a brokered mount) is applied per-run in __main__.
builder_image = flyte.Image.from_debian_base(name="llamacpp-volume-builder", install_flyte=True).with_pip_packages(
    FLYTEPLUGINS_UNION
)
builder_env = flyte.TaskEnvironment(
    name="llamacpp-volume-builder",
    image=builder_image,
    resources=flyte.Resources(cpu="2", memory="8Gi", disk="30Gi"),
)

# Client task pod; the sidecar pod template (and, for --reuse, the reuse policy) are injected per-run.
env = flyte.TaskEnvironment(
    name="llamacpp-sidecar-volume",
    image=CLIENT_IMAGE,
    resources=flyte.Resources(cpu="2", memory="2Gi"),
)


def _volume_name(artifact_name: str, artifact_version: str) -> str:
    """Volume name derived from the artifact identity -- one volume per (name, version).

    A digest of the full name@version keeps it unique and bounded even when the name is long or
    unusual; the sanitized name is a best-effort readable prefix.
    """
    import hashlib
    import re

    digest = hashlib.sha256(f"{artifact_name}@{artifact_version}".encode()).hexdigest()[:12]
    prefix = re.sub(r"[^a-z0-9-]", "-", artifact_name.lower()).strip("-")[: 63 - len(digest) - 1].strip("-")
    return f"{prefix or 'vol'}-{digest}"


def _gpu_count(gpu: str | None) -> int:
    """Parse the `<FAMILY>:<count>` accelerator string to a count (e.g. 'L4:2' -> 2)."""
    if not gpu:
        return 0
    return int(gpu.split(":", 1)[1]) if ":" in gpu else 1


def _prebuilt_serve_image(cuda: bool) -> flyte.Image:
    """Serve image from the official prebuilt llama.cpp binaries (ggml-org) instead of compiling.

    `build_llama_cpp_image` compiles llama.cpp from source -- correct, but a heavy cmake/CUDA build
    that can be slow and taxing on the remote image builder. `LLAMACPP_PREBUILT=1` opts into this
    lighter alternative: `from_base` the official image (which ships `llama-server` at `/app`) and
    only layer Python + the plugins on top (seconds, no compile). `from_base` images are unnamed and
    non-extendable by default, hence the `.clone(name=..., extendable=True)`. `flyte` installs
    transitively as a plugin dependency.
    """
    base = "ghcr.io/ggml-org/llama.cpp:server-cuda" if cuda else "ghcr.io/ggml-org/llama.cpp:server"
    return (
        flyte.Image.from_base(base)
        .clone(name="llama-cpp-prebuilt", extendable=True)
        .with_apt_packages("python3", "python3-pip", "python3-venv")
        .with_pip_packages("flyteplugins-llamacpp", FLYTEPLUGINS_UNION, pre=True)
        .with_env_vars({"PATH": "/app:/usr/local/bin:/usr/bin:/bin"})  # /app holds the prebuilt llama-server
    )


@builder_env.task(cache="auto")
async def build_model_volume(model: Dir, volume_name: str) -> str:
    """Copy the prefetched Model artifact into a fresh Union Volume; return its ROVolume locator.

    `cache="auto"` keys on (model, volume_name), and `model` is bound as a task input so the Run
    records Run -> artifact lineage.
    """
    from flyteplugins.union.io import Volume

    vol = Volume.new(name=volume_name)
    mount_path = await vol.mount()
    await model.download(str(mount_path))
    ro = await vol.commit()
    if not ro.locator:
        raise RuntimeError("committed volume has no locator")
    flyte.logger.info("Built model volume %r -> locator=%s", volume_name, ro.locator)
    return ro.locator


def _volume_sidecar(serve_image_uri: str, locator: str, serve_cmd: list[str]) -> flyte.PodTemplate:
    """primary (client) + a llama.cpp server sidecar that serves the model straight from a Volume.

    The sidecar runs `union-volume-exec`, which mounts the read-only Volume through the broker and
    then runs the llama.cpp server against it. `allow_volumes()` supplies exactly the pod/container
    plumbing a brokered mount needs; we take that wired container and attach it as a native sidecar
    (`restart_policy="Always"`) so k8s starts it before, and tears it down with, the primary.
    """
    from flyteplugins.union.io import allow_volumes
    from kubernetes.client.models import V1Container, V1PodSpec, V1ResourceRequirements, V1Toleration

    brokered = allow_volumes(flyte.PodTemplate(primary_container_name=SIDECAR_NAME))
    brokered_spec = brokered.pod_spec
    assert brokered_spec is not None  # allow_volumes always populates the pod spec
    sidecar = brokered_spec.containers[0]  # the broker-wired container, named SIDECAR_NAME
    sidecar.image = serve_image_uri
    sidecar.restart_policy = "Always"
    # union-volume-exec mounts the ROVolume at MODEL_MOUNT, then runs the llama.cpp serve argv.
    sidecar.command = ["union-volume-exec", "--locator", str(locator), "--mount-at", MODEL_MOUNT, "--", *serve_cmd]

    # GPU goes on the SIDECAR: nvidia.com/gpu needs request==limit. Flyte injects GPU tolerations for
    # a *primary* GPU request, but this GPU is on a sidecar, so tolerate the GPU node taints ourselves:
    # `nvidia.com/gpu` (GKE) and `k8s.amazonaws.com/accelerator` (managed EKS); each a no-op elsewhere.
    tolerations = None
    if GPU:
        gpu_res = {"nvidia.com/gpu": str(_gpu_count(GPU))}
        sidecar.resources = V1ResourceRequirements(limits=gpu_res, requests=dict(gpu_res))
        tolerations = [
            V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"),
            V1Toleration(key="k8s.amazonaws.com/accelerator", operator="Exists", effect="NoSchedule"),
        ]

    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            init_containers=[sidecar],
            volumes=brokered_spec.volumes,  # broker CSI channel + staging
            tolerations=tolerations,
        ),
        annotations=brokered.annotations,  # carries the broker capability gate
    )


@flyte.trace
def _await_server_ready(base_url: str) -> str:
    """Block until the sidecar answers -- i.e. it has mounted the Volume and finished loading the
    model. `@flyte.trace` records this as its own timed sub-action, so the mount + cold-load shows
    up separately from inference in the run timeline.
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
    """One OpenAI-compatible chat completion, traced so inference latency is separate from cold-start."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="sk-noauth")
    resp = client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": prompt}], max_tokens=256
    )
    return resp.choices[0].message.content or ""


@env.task
def chat(prompt: str, model: Dir) -> str:
    """Call the local llama.cpp sidecar (OpenAI-compatible) once it is ready.

    `model` is bound to the Model artifact purely to record Run -> artifact lineage; the bytes are
    not read here -- the sidecar serves them from the Volume it mounted via the locator baked into
    the pod template at submit time. The two phases are `@flyte.trace`d under `flyte.group`, so the
    run UI shows mount + cold-start separately from the inference call. In --reuse mode the first
    call pays the cold-start; later calls hit the warm actor.
    """
    base_url = f"http://localhost:{SERVER_PORT}/v1"
    with flyte.group("llm-serve"):
        _await_server_ready(base_url)
        return _complete(base_url, prompt)


if __name__ == "__main__":
    import argparse
    import asyncio

    from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image
    from flyteplugins.union.io import allow_volumes

    import flyte.prefetch
    from flyte.remote import Artifact

    p = argparse.ArgumentParser(description="llama.cpp task-pod sidecar serving a Model from a Union Volume.")
    p.add_argument("--prompt", default="Write a haiku about GPUs.")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Use a reusable actor (warm pod) instead of an ephemeral pod; fires twice to show the warm speedup.",
    )
    p.add_argument("--config", help="Path to the Flyte config (else UCTL_CONFIG / the default).")
    p.add_argument("--project", help="Project to run in.")
    p.add_argument("--domain", help="Domain to run in (default: the config's).")
    args = p.parse_args()

    flyte.init_from_config(path_or_config=args.config, project=args.project, domain=args.domain)

    def _artifact_exists(name: str) -> bool:
        try:
            Artifact.get(name)
            return True
        except Exception:
            return False

    # Reuse the artifact if present; prefetch only when missing or LLAMACPP_FORCE_PREFETCH is set.
    force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
    if force or not _artifact_exists(ARTIFACT_NAME):
        run = flyte.prefetch.hf_model(
            repo=MODEL_REPO,
            artifact_name=ARTIFACT_NAME,
            allow_patterns=[f"*{QUANT}*"],
            hf_token_key=None,  # public repo
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
    artifact: Artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    # Build the Volume from the artifact (name derived from its identity); its locator is the serve handle.
    volume_name = _volume_name(ARTIFACT_NAME, artifact.version)
    build_run = flyte.run(
        build_model_volume.override(pod_template=allow_volumes()),  # zero-privilege brokered mount for the build
        model=artifact,
        volume_name=volume_name,
    )
    print(f"Building model volume {volume_name!r} (artifact {ARTIFACT_NAME}@{artifact.version}): {build_run.url}")
    build_run.wait()
    locator = build_run.outputs()[0]
    print(f"Model volume locator: {locator}")

    # Sidecar images are string URIs, so build the serve image and use its URI. Default: compile
    # llama.cpp from source; `LLAMACPP_PREBUILT=1` uses the lighter prebuilt-binary image instead.
    if os.getenv("LLAMACPP_PREBUILT", "").lower() in ("1", "true", "yes"):
        serve_image = _prebuilt_serve_image(cuda=bool(GPU))
    else:
        serve_image = build_llama_cpp_image(name="llama-cpp-sidecar-volume", cuda=bool(GPU)).with_pip_packages(
            FLYTEPLUGINS_UNION
        )
    built = asyncio.run(flyte.build.aio(serve_image))  # type: ignore[arg-type, var-annotated]  # ty: ignore[invalid-argument-type]
    print(f"llama.cpp sidecar image ({'cuda' if GPU else 'cpu'}): {built.uri}")

    serve_cmd = build_fserve_command(
        model_id=MODEL_ID, port=SERVER_PORT, model_dir=MODEL_MOUNT, extra_args=EXTRA_ARGS.split()
    )
    pod_template = _volume_sidecar(str(built.uri), locator, serve_cmd)

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
