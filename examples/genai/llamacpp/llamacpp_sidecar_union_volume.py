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

Which model to serve is chosen from the `MODELS` registry below via `LLAMACPP_MODEL=<id>` (default:
the first entry). Two shapes via `--reuse`:
  * ephemeral (default): fresh pod per run; the model cold-loads each time.
  * reusable actor (`--reuse`): a warm pool keeps the loaded model across runs, so only the first
    call pays the mount + load.

All `flyteplugins.*` imports are lazy so the client task container -- flyte + openai only -- never
imports them.

Run:
    python examples/genai/llamacpp/llamacpp_sidecar_union_volume.py --prompt "Write a haiku about GPUs."
    LLAMACPP_MODEL=qwen3-27b python examples/genai/llamacpp/llamacpp_sidecar_union_volume.py --reuse
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import flyte
from flyte.io import Dir


@dataclass(frozen=True)
class Model:
    """A GGUF model this example can serve with llama.cpp."""

    id: str  # served model id + selector handle (LLAMACPP_MODEL)
    repo: str  # HuggingFace GGUF repo
    quant: str  # quant tag; selects the .gguf via allow_patterns="*<quant>*"
    gpu: str | None = None  # accelerator on the server sidecar, e.g. "L4:1"; None serves on CPU
    server_args: str = "--ctx-size 8192"  # extra llama-server flags
    cpu: str = "2"  # server sidecar cpu
    memory: str = "8Gi"  # server sidecar memory
    disk: str = "20Gi"  # prefetch + volume-build scratch


# The models this example knows how to serve. Pick one with LLAMACPP_MODEL=<id> (default: the first).
MODELS: list[Model] = [
    Model(id="qwen2.5-0.5b-instruct", repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF", quant="q4_k_m"),
    Model(
        id="qwen3-27b",
        repo="unsloth/Qwen3.8-27B-GGUF",
        quant="Q8_0",
        gpu="L4:2",
        server_args="--ctx-size 16384 --n-gpu-layers 999 --flash-attn on",
        cpu="8",
        memory="48Gi",
        disk="60Gi",
    ),
]


def _select(models: list[Model]) -> Model:
    want = os.getenv("LLAMACPP_MODEL")
    if want is None:
        return models[0]
    chosen = next((m for m in models if m.id == want), None)
    if chosen is None:
        raise SystemExit(f"LLAMACPP_MODEL={want!r} not found; choose one of {[m.id for m in models]}")
    return chosen


def _slug(*parts: str) -> str:
    """Lowercase [a-z0-9-] slug (the Artifact/Volume name charset) from the given parts."""
    import re

    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", "-".join(p for p in parts if p).lower())).strip("-")


MODEL = _select(MODELS)
# One artifact (and, downstream, one Volume) per model + quant, so different models/quants never collide.
ARTIFACT_NAME = _slug(MODEL.repo.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf"), MODEL.quant)

SERVER_PORT = 8080
MODEL_MOUNT = "/tmp/models"  # where union-volume-exec mounts the volume in the sidecar; --model-dir points here
SIDECAR_NAME = "llama"

# Pinned to the beta that first ships `union-volume-exec`; relax once a stable 0.11.x is released.
FLYTEPLUGINS_UNION = "flyteplugins-union>=0.11.0b0"

# Client image: OpenAI client + unionai-reuse (the actor bridge for --reuse; harmless otherwise).
# Deliberately no flyteplugins-union -- the client only talks HTTP to the sidecar.
CLIENT_IMAGE = flyte.Image.from_debian_base(name="llamacpp-sidecar-client", install_flyte=True).with_pip_packages(
    "openai", "unionai-reuse"
)

# Builder needs the Volume client; its pod template (a brokered mount) is applied per-run in _main.
builder_image = flyte.Image.from_debian_base(name="llamacpp-volume-builder", install_flyte=True).with_pip_packages(
    FLYTEPLUGINS_UNION
)
builder_env = flyte.TaskEnvironment(
    name="llamacpp-volume-builder",
    image=builder_image,
    resources=flyte.Resources(cpu="2", memory="8Gi", disk=MODEL.disk),
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

    digest = hashlib.sha256(f"{artifact_name}@{artifact_version}".encode()).hexdigest()[:12]
    prefix = _slug(artifact_name)[: 63 - len(digest) - 1].strip("-")
    return f"{prefix or 'vol'}-{digest}"


def _gpu_count(gpu: str | None) -> int:
    """Parse the `<FAMILY>:<count>` accelerator string to a count (e.g. 'L4:2' -> 2)."""
    if not gpu:
        return 0
    return int(gpu.split(":", 1)[1]) if ":" in gpu else 1


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

    # Size the server sidecar. request==limit on every resource: a sidecar isn't Flyte-managed, so
    # nothing mirrors requests into limits for us, and a request without a matching limit is rejected
    # where a LimitRange is enforced. nvidia.com/gpu also requires request==limit + a GPU toleration.
    resources = {"cpu": MODEL.cpu, "memory": MODEL.memory}
    tolerations = None
    if MODEL.gpu:
        resources["nvidia.com/gpu"] = str(_gpu_count(MODEL.gpu))
        # Flyte injects GPU node tolerations for a *primary* GPU request, but this GPU is on a
        # sidecar, so tolerate the GPU node taints ourselves: `nvidia.com/gpu` (GKE convention) and
        # `k8s.amazonaws.com/accelerator` (managed EKS). Each is a no-op where that taint is absent.
        tolerations = [
            V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule"),
            V1Toleration(key="k8s.amazonaws.com/accelerator", operator="Exists", effect="NoSchedule"),
        ]
    sidecar.resources = V1ResourceRequirements(requests=dict(resources), limits=dict(resources))

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
        model=MODEL.id, messages=[{"role": "user", "content": prompt}], max_tokens=256
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


async def _main(prompt: str, reuse: bool, config: str | None, project: str | None, domain: str | None) -> None:
    from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image
    from flyteplugins.union.io import allow_volumes

    import flyte.prefetch
    from flyte.remote import Artifact

    flyte.init_from_config(path_or_config=config, project=project, domain=domain)

    async def _serve_image_uri() -> str:
        """Build the llama.cpp serve image (+ Volume client). Independent of the volume, so it runs
        concurrently with the prefetch + volume build below."""
        serve_image = build_llama_cpp_image(name="llama-cpp-sidecar-volume", cuda=bool(MODEL.gpu)).with_pip_packages(
            FLYTEPLUGINS_UNION
        )
        built = await flyte.build.aio(serve_image)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert built.uri is not None
        print(f"llama.cpp sidecar image ({'cuda' if MODEL.gpu else 'cpu'}): {built.uri}")
        return built.uri

    async def _model_and_locator() -> "tuple[Artifact, str]":
        """Prefetch (if needed) -> build the Union Volume -> return (artifact, ROVolume locator).
        Serial by necessity: the volume build consumes the artifact."""

        def _artifact_exists(name: str) -> bool:
            try:
                Artifact.get(name)
                return True
            except Exception:
                return False

        force = os.getenv("LLAMACPP_FORCE_PREFETCH", "").lower() in ("1", "true", "yes")
        if force or not _artifact_exists(ARTIFACT_NAME):
            run = flyte.prefetch.hf_model(
                repo=MODEL.repo,
                artifact_name=ARTIFACT_NAME,
                allow_patterns=[f"*{MODEL.quant}*"],
                hf_token_key=None,  # public repo
                resources=flyte.Resources(cpu="2", memory="4Gi", disk=MODEL.disk),
            )
            print(f"Prefetching {MODEL.repo} ({MODEL.quant}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
            await run.wait.aio()
        else:
            print(f"Reusing artifact {ARTIFACT_NAME!r} (LLAMACPP_FORCE_PREFETCH=1 to re-create)")
        artifact = await Artifact.get.aio(ARTIFACT_NAME, "latest")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        volume_name = _volume_name(ARTIFACT_NAME, artifact.version)
        build_run = await flyte.run.aio(
            build_model_volume.override(pod_template=allow_volumes()),  # zero-privilege brokered mount for the build
            model=artifact,
            volume_name=volume_name,
        )
        print(f"Building model volume {volume_name!r} (artifact {ARTIFACT_NAME}@{artifact.version}): {build_run.url}")
        await build_run.wait.aio()
        locator = build_run.outputs()[0]
        print(f"Model volume locator: {locator}")
        return artifact, locator

    # The serve image and the model volume are independent -> build them concurrently.
    (artifact, locator), serve_image_uri = await asyncio.gather(_model_and_locator(), _serve_image_uri())

    serve_cmd = build_fserve_command(
        model_id=MODEL.id, port=SERVER_PORT, model_dir=MODEL_MOUNT, extra_args=MODEL.server_args.split()
    )
    pod_template = _volume_sidecar(serve_image_uri, locator, serve_cmd)

    if reuse:
        # Warm actor: first call cold-loads, later calls reuse -- fire twice to show the speedup. Serial
        # on purpose: the second call must see the warm pod the first one primed.
        task = chat.override(
            pod_template=pod_template,
            reusable=flyte.ReusePolicy(replicas=1, concurrency=1, idle_ttl=600, scaledown_ttl=600),
        )
        for i in range(2):
            run = await flyte.run.aio(task, prompt=prompt, model=artifact)
            print(f"[reuse call {i + 1}/2] {run.url}")
            await run.wait.aio()
            print(run.outputs())
    else:
        run = await flyte.run.aio(chat.override(pod_template=pod_template), prompt=prompt, model=artifact)
        print(f"Run: {run.url}")
        await run.wait.aio()
        print(run.outputs())


if __name__ == "__main__":
    import argparse
    import asyncio

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

    asyncio.run(_main(args.prompt, args.reuse, args.config, args.project, args.domain))
