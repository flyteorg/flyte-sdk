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

Structure: `chat` is a **module-scope `@env.task`** so the runtime can resolve it. The sidecar
pod template needs values known only at submit time (the built llama.cpp image URI + the
resolved model directory), so it is injected per-run via `chat.override(pod_template=...)`
rather than baked into a dynamically-built environment (a dynamic env has no module-scope task
for the runtime to reconstruct).

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

`flyteplugins.llamacpp` is imported lazily (submit-time helpers / __main__), never at module
scope: the `chat` task runs in the client image (flyte + openai only) and the runtime imports
this module to resolve `chat`, so a module-level plugin import would crash the task container.

Run:
    python examples/genai/llamacpp/llamacpp_sidecar.py --prompt "Write a haiku about GPUs."
"""

from __future__ import annotations

from urllib.parse import urlparse

import flyte
from flyte.io import Dir

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"  # public repo, small enough for CPU
QUANT = "q4_k_m"
ARTIFACT_NAME = "qwen2-5-0-5b-instruct-q4-k-m"
MODEL_ID = "qwen2.5-0.5b-instruct"
SERVER_PORT = 8080

# Where the read-only, data-bucket-root PVC is mounted into the sidecar. The artifact's key
# under the bucket resolves at `<MODEL_MOUNT>/<key>`.
MODEL_MOUNT = "/tmp/models"
# The pre-provisioned RO PVC over the data-bucket root (dataplane helm release). Required: it
# must name a claim that exists in the task's namespace -- see llamacpp_app_fuse.py / README.md.
MODEL_PVC = "flyte-metadata-ro"

# The client task's own image (Flyte-built): just needs an OpenAI client.
CLIENT_IMAGE = flyte.Image.from_debian_base(name="llamacpp-sidecar-client", install_flyte=True).with_pip_packages(
    "openai"
)

# Module-scope environment + task so the runtime can resolve `chat`. The sidecar pod template
# (built image URI + resolved model dir, known only at submit time) is injected per-run via
# `chat.override(pod_template=...)` in __main__.
env = flyte.TaskEnvironment(
    name="llamacpp-sidecar",
    image=CLIENT_IMAGE,
    resources=flyte.Resources(cpu="2", memory="6Gi"),
)


def _pod_template(serve_image_uri: str, model_dir: str) -> flyte.PodTemplate:
    """primary (client) + a llama.cpp server sidecar reading the model from the RO model PVC."""
    from flyteplugins.llamacpp import build_fserve_command
    from kubernetes.client.models import (
        V1Container,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1Volume,
        V1VolumeMount,
    )

    # The same llama-cpp-fserve argv the App runs; build_fserve_command returns shell-safe
    # tokens meant to be joined into one shell string (how fserve execs llama-server), so
    # wrap them in `sh -c`. model_dir points at the artifact's directory under the fuse mount.
    server_cmd = build_fserve_command(model_id=MODEL_ID, port=SERVER_PORT, model_dir=model_dir)
    llama_sidecar = V1Container(
        name="llama",
        image=serve_image_uri,
        restart_policy="Always",  # native sidecar: started before, torn down with, the primary
        command=["/bin/sh", "-c", " ".join(server_cmd)],
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
    args = p.parse_args()

    flyte.init_from_config()

    # 1. A Flyte run creates the Model artifact (one quant, kept by allow_patterns), published
    #    as a versioned artifact in the data bucket -- the same bucket the RO PVC mounts.
    run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        artifact_name=ARTIFACT_NAME,
        allow_patterns=[f"*{QUANT}*"],
        hf_token_key=None,  # public repo: prefetch anonymously
        resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    )
    print(f"Prefetching {MODEL_REPO} ({QUANT}) -> artifact {ARTIFACT_NAME!r}: {run.url}")
    run.wait()

    # 2. Resolve the artifact to its object-store URI, then to the mounted directory the sidecar
    #    reads. Pinning the version here keeps the batch run reproducible.
    artifact = asyncio.run(Artifact.get.aio(ARTIFACT_NAME, "latest"))
    uri = asyncio.run(artifact.to_python(Dir)).path
    model_dir = _artifact_model_dir(uri)
    print(f"Serving artifact from fuse mount: {uri} -> {model_dir}")

    # 3. Sidecar images must be string URIs, so build the llama.cpp image and use its URI. The
    #    image carries llama-server + the `llama-cpp-fserve` shim (installs flyteplugins-llamacpp).
    from flyteplugins.llamacpp import build_llama_cpp_image

    serve_image = build_llama_cpp_image(name="llama-cpp-sidecar", cuda=False)
    built = asyncio.run(flyte.build.aio(serve_image))
    print(f"llama.cpp sidecar image: {built.uri}")

    # 4. Inject the per-run sidecar pod template onto the module-scope task and run it. The
    #    artifact is passed as the `model` input so the Run records Run -> artifact lineage; the
    #    sidecar serves the same artifact's bytes from the fuse mount.
    task = chat.override(pod_template=_pod_template(built.uri, model_dir))
    run = flyte.run(task, prompt=args.prompt, model=artifact)
    print(f"Run: {run.url}")
    run.wait()
    print(run.outputs())
