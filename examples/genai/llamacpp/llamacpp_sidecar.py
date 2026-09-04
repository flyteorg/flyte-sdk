"""
Serve a GGUF model with llama.cpp as a **native sidecar in a Flyte task pod** -- for
batch/pipeline inference against a co-located model, as opposed to the standalone,
scale-to-zero Flyte App in `llamacpp_app.py`.

The task pod runs two containers:
  * `primary` -- the client task; it calls the local server on `localhost`.
  * `llama`   -- a llama.cpp server (llama-server via the `llama-cpp-fserve` shim) started
                 with the plugin's `build_fserve_command` -- the *same* argv
                 `LlamaCppAppEnvironment` runs. Both serving shapes assemble the command the
                 one way, so tuning/flags stay identical.

The server is a **native sidecar**: an init container with `restart_policy="Always"`, so
Kubernetes starts it before the primary and SIGTERMs it when the primary exits (the pod
reaches a terminal state instead of hanging with a forever-running plain container).

Self-contained: the sidecar loads the model straight from HuggingFace (`--hf-repo` via
`model_hf_path`), so there is no model volume to provision. To serve a mounted model instead
(object-store FUSE), pass `model_dir=<mount>/<subpath>` to `build_fserve_command` and attach
the PVC to the pod (see `llamacpp_app_gcsfuse.py`).

Sidecar container images must be string image URIs (`V1Container.image`), not `flyte.Image`
objects (only the task environment's own image is Flyte-built), so `__main__` builds the
llama.cpp image first with `flyte.build` and passes its URI.

Run:
    python examples/genai/llamacpp/llamacpp_sidecar.py --prompt "Write a haiku about GPUs."
"""

from __future__ import annotations

from flyteplugins.llamacpp import build_fserve_command, build_llama_cpp_image

import flyte

MODEL_HF = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m"  # public repo, small enough for CPU
MODEL_ID = "qwen2.5-0.5b-instruct"
SERVER_PORT = 8080

# CPU-only so the example needs no GPU. The image carries llama-server + the
# `llama-cpp-fserve` shim (it installs `flyteplugins-llamacpp`).
SERVE_IMAGE = build_llama_cpp_image(name="llama-cpp-sidecar", cuda=False)

# The client task's own image (Flyte-built): just needs an OpenAI client.
CLIENT_IMAGE = flyte.Image.from_debian_base(name="llamacpp-sidecar-client", install_flyte=True).with_pip_packages(
    "openai"
)


def _pod_template(serve_image_uri: str) -> flyte.PodTemplate:
    """primary (client) + a llama.cpp server as a native sidecar."""
    from kubernetes.client.models import V1Container, V1PodSpec

    # The same llama-cpp-fserve argv the App runs; build_fserve_command returns shell-safe
    # tokens meant to be joined into one shell string (how fserve execs llama-server), so
    # wrap them in `sh -c`.
    server_cmd = build_fserve_command(model_id=MODEL_ID, port=SERVER_PORT, model_hf_path=MODEL_HF)
    llama_sidecar = V1Container(
        name="llama",
        image=serve_image_uri,
        restart_policy="Always",  # native sidecar: started before, torn down with, the primary
        command=["/bin/sh", "-c", " ".join(server_cmd)],
    )
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            init_containers=[llama_sidecar],
        ),
    )


def _build_env(serve_image_uri: str) -> flyte.TaskEnvironment:
    return flyte.TaskEnvironment(
        name="llamacpp-sidecar",
        image=CLIENT_IMAGE,
        pod_template=_pod_template(serve_image_uri),
        resources=flyte.Resources(cpu="2", memory="6Gi"),
    )


def chat(prompt: str) -> str:
    """Call the local llama.cpp sidecar (OpenAI-compatible) once it is ready."""
    import time

    from openai import OpenAI

    client = OpenAI(base_url=f"http://localhost:{SERVER_PORT}/v1", api_key="sk-noauth")
    # Wait for the sidecar to finish loading the model (cold HF download on first start).
    for _ in range(150):
        try:
            client.models.list()
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama.cpp sidecar did not become ready")

    resp = client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": prompt}], max_tokens=256
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import argparse
    import asyncio

    p = argparse.ArgumentParser(description="llama.cpp as a Flyte task-pod sidecar.")
    p.add_argument("--prompt", default="Write a haiku about GPUs.")
    args = p.parse_args()

    flyte.init_from_config()

    # Sidecar images must be string URIs, so build the llama.cpp image and use its URI.
    built = asyncio.run(flyte.build.aio(SERVE_IMAGE))
    print(f"llama.cpp sidecar image: {built.uri}")

    env = _build_env(built.uri)
    run = flyte.run(env.task(chat), prompt=args.prompt)
    print(f"Run: {run.url}")
    run.wait()
    print(run.outputs())
