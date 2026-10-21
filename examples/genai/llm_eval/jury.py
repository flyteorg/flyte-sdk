"""
**LLM-as-Jury** — grade candidate answers with a *panel* of diverse LLM jurors. A **parent task** fans
the panel out, ensembles the verdicts into a score + an **agreement = confidence** signal, and emits an
**HTML comparison report** (`flyte.report`) so you can eyeball where the jurors agreed and diverged.

A single judge is a point estimate; an independent, diverse panel turns the spread of verdicts into a
signal — the **mean score** is the grade, the jurors' **agreement** is a **confidence** (unanimous →
trust it; split → route to a human). Judging is **fail-soft**: a juror that errors (throttle, auth,
cold sidecar) contributes nothing; the panel just shrinks. Jurors are **reusable actors** (a warm pod
per juror); the orchestrator is a separate non-reusable parent. Every verdict is **schema-constrained**
to strict `{"score": 1-5, "rationale": ...}`.

Three juror archetypes — mix and match
---------------------------------------
Defaults to **3 self-hosted jurors** (in-perimeter, no external credentials). All three route through
`litellm`, so a juror is just `(model, auth)` and auth is the only difference:

  1. **LOCAL** (self-hosted, 3 by default) — a `llama.cpp` sidecar serving a small instruct GGUF. Two
     delivery modes (default = download):
       * `download` — the sidecar pulls the GGUF at startup (`llama-server --hf-repo`); no prereqs.
       * `fuse` — a prefetched Model **artifact** read in place over the RO data-bucket-root PVC, like
         `../llamacpp/llamacpp_sidecar.py` (needs the PVC). Set `LLM_JURY_LOCAL_DELIVERY=fuse`.
     Set `LLM_JURY_LOCAL_GPU=<FAMILY>:<n>` to run each juror on an accelerator.
  2. **SECRET-API-KEY** (cloud-agnostic) — a hosted model reached with an API key stored as a **Flyte
     Secret**; `litellm` reads the injected env var — no key in code. Uncomment in `SECRET_API_JURORS`.
  3. **WORKLOAD IDENTITY** (cloud-specific, keyless) — a cloud-managed model via the pod's federated IAM
     identity: Vertex on **GKE WI** / Bedrock on **EKS IRSA**, expressed as the task's `service_account`.
     Uncomment in `WI_JURORS`.

Testing across clouds — env only, no code change (point `--config`/`--project` at your deployment):
  * **GKE (GCP):**  `LLM_JURY_LOCAL_GPU=L4:1`   (+ an optional Vertex Workload-Identity juror; see `WI_JURORS`)
  * **EKS (AWS):**  `LLM_JURY_LOCAL_GPU=L40S:1` (+ an optional Bedrock IRSA juror; see `WI_JURORS`)

Run (defaults to 3 local jurors, download delivery; the report is on the parent run in the UI):
    python examples/genai/llm_eval/jury.py
    LLM_JURY_LOCAL_GPU=L4:1 python examples/genai/llm_eval/jury.py
    python examples/genai/llm_eval/jury.py --question "..." --answer "..."
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Awaitable, Callable
from typing import Any

import flyte
import flyte.report

SERVER_PORT = 8080  # each self-hosted juror's local llama.cpp endpoint

# =================================================================================================
# Juror registries — three archetypes. Add/remove entries freely; diversity strengthens the signal.
# =================================================================================================

# 1) LOCAL (self-hosted) — DEFAULT panel: (name, hf_repo, quant). Three diverse families. `name` is the
#    served alias + artifact stem, so keep it [a-z0-9-].
LOCAL_JURORS: list[tuple[str, str, str]] = [
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct-GGUF", "q4_k_m"),
    ("llama31-8b", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Q4_K_M"),
    ("gemma2-9b", "bartowski/gemma-2-9b-it-GGUF", "Q4_K_M"),
]

# 2) SECRET-API-KEY (cloud-agnostic) — (name, litellm_model, flyte.Secret). Uncomment to include:
SECRET_API_JURORS: list[tuple[str, str, flyte.Secret]] = [
    # ("openai-4o-mini", "openai/gpt-4o-mini",
    #     flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")),
]

# 3) WORKLOAD IDENTITY (cloud-specific, keyless) — (name, litellm_model, ksa). Uncomment to include:
WI_JURORS: list[tuple[str, str, str]] = [
    # ("vertex-gemini-flash", "vertex_ai/gemini-2.0-flash", "vertex-jury"),   # GKE Workload Identity KSA
    # ("bedrock-claude", "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0", "bedrock-jury"),  # EKS IRSA KSA
]

# ---- Self-hosted delivery config -----------------------------------------------------------------
# "download" (default): the sidecar pulls the GGUF at startup (--hf-repo); no prereqs.
# "fuse": serve a prefetched artifact in place over the RO, data-bucket-root PVC (needs the PVC).
LOCAL_DELIVERY = os.getenv("LLM_JURY_LOCAL_DELIVERY", "download").lower()
LOCAL_GPU = os.getenv("LLM_JURY_LOCAL_GPU", "") or None  # e.g. "L4:1" (GCP) / "L40S:1" (AWS); unset => CPU
MODEL_MOUNT = "/tmp/models"  # fuse mode: where the RO data-bucket-root PVC is mounted into the sidecar
MODEL_PVC = os.getenv("LLM_JURY_LOCAL_PVC", "flyte-metadata-ro")  # fuse mode; see llamacpp_app_fuse.py

RUBRIC = os.getenv(
    "LLM_JURY_RUBRIC",
    "Grade the ANSWER to the QUESTION on a 1-5 integer scale for correctness and quality of reasoning: "
    "5 = fully correct and well-justified; 3 = partially correct or thinly justified; 1 = incorrect or "
    "irrelevant. Judge only the answer's merit, not its length or style.",
)

# Self-contained grading set (answers of deliberately varied quality). Replace with your own.
_EXAMPLES = [
    {
        "question": "Why does adding an index to a database column speed up equality lookups?",
        "answer": "An index keeps the column's values in a sorted structure (e.g. a B-tree), so the "
        "engine binary-searches to matching rows in O(log n) instead of scanning every row (O(n)).",
    },
    {
        "question": "What is the time complexity of appending N items to a Python list one at a time?",
        "answer": "It's O(N^2) because each append copies the whole list.",
    },
]

# OpenAI-compatible structured-output schema every juror is constrained to.
_VERDICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
        },
    },
}

# Juror image: litellm (routes openai / anthropic / vertex_ai / bedrock uniformly on ambient creds —
# a Secret env var OR a WI/IRSA identity) + provider SDKs + the reuse actor bridge.
JUROR_IMAGE = flyte.Image.from_debian_base(name="llm-jury-juror", install_flyte=True).with_pip_packages(
    "litellm", "google-cloud-aiplatform", "boto3", "unionai-reuse"
)
# Orchestrator image: builds the juror sidecar pod templates (kubernetes) and the llama.cpp argv
# (flyteplugins-llamacpp); does no inference itself. Non-reusable (a reusable parent would pin a slot).
ORCH_IMAGE = flyte.Image.from_debian_base(name="llm-jury-orch", install_flyte=True).with_pip_packages(
    "flyteplugins-llamacpp", "kubernetes"
)

# Jurors are reusable actors (warm pod each); the orchestrator is a separate short-lived parent.
juror_env = flyte.TaskEnvironment(
    name="llm-jury-juror",
    image=JUROR_IMAGE,
    # The llama.cpp sidecar shares this pod (download mode: model + KV cache live in RAM on CPU); size
    # for the model. On GPU the weights are in VRAM, so this is just host overhead.
    resources=flyte.Resources(cpu="2", memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=4, idle_ttl=900, scaledown_ttl=900),
)
orch_env = flyte.TaskEnvironment(
    name="llm-jury",
    image=ORCH_IMAGE,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    depends_on=[juror_env],
)


def _prompt(question: str, answer: str, rubric: str) -> str:
    return f"{rubric}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\nRespond ONLY with the JSON verdict."


def _parse(raw: str, model: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return {"model": model, "score": int(v["score"]), "rationale": v.get("rationale", "")}
    except Exception as e:
        return {"model": model, "score": None, "rationale": f"(bad verdict: {str(e)[:100]})"}


@flyte.trace
def _complete(model: str, prompt: str, api_base: str | None = None) -> str:
    """One structured, schema-constrained completion via litellm. Credentials are ambient: a Flyte
    Secret's env var, the pod's WI/IRSA identity, or none for the self-hosted sidecar (`api_base`)."""
    import litellm

    kw: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict, fair grader. Output only the JSON verdict."},
            {"role": "user", "content": prompt},
        ],
        "response_format": _VERDICT_SCHEMA,
        "temperature": 0.0,
        "max_tokens": 512,
        "drop_params": True,
    }
    if api_base:
        kw.update(api_base=api_base, api_key="sk-noauth")
    return litellm.completion(**kw).choices[0].message.content or ""


@flyte.trace
def _await_ready(api_base: str, model_id: str) -> str:
    """Block until a self-hosted juror's sidecar answers (model pulled/mounted + loaded)."""
    import time

    import litellm

    for _ in range(180):
        try:
            litellm.completion(
                model=f"openai/{model_id}",
                api_base=api_base,
                api_key="sk-noauth",
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            return "ready"
        except Exception:
            time.sleep(2)
    raise RuntimeError("self-hosted juror sidecar did not become ready")


@juror_env.task
async def grade(question: str, answer: str, rubric: str, model: str) -> dict:
    """A hosted-model juror (SECRET or WI archetype). Auth is ambient (Secret env var / WI-IRSA
    identity); code is identical. Fail-soft: errors return score=None and drop out of the ensemble."""
    with flyte.group("jury"):
        try:
            return _parse(_complete(model, _prompt(question, answer, rubric)), model)
        except Exception as e:
            return {"model": model, "score": None, "rationale": f"(juror failed: {str(e)[:120]})"}


@juror_env.task
async def grade_local(question: str, answer: str, rubric: str, model_id: str) -> dict:
    """A self-hosted juror, from its co-located llama.cpp sidecar (served alias `model_id`); the sidecar
    has already delivered the model (download via --hf-repo, or FUSE mount). Fail-soft."""
    base_url = f"http://localhost:{SERVER_PORT}/v1"
    with flyte.group("jury"):
        try:
            _await_ready(base_url, model_id)
            return _parse(_complete(f"openai/{model_id}", _prompt(question, answer, rubric), base_url), model_id)
        except Exception as e:
            return {"model": model_id, "score": None, "rationale": f"(juror failed: {str(e)[:120]})"}


def _gpu_count(gpu: str | None) -> int:
    if not gpu:
        return 0
    return int(gpu.split(":", 1)[1]) if ":" in gpu else 1


def _local_pod_template(
    serve_image_uri: str,
    model_id: str,
    *,
    hf_repo: str | None = None,
    model_dir: str | None = None,
    local_gpu: str | None = None,
) -> flyte.PodTemplate:
    """primary (client) + a llama.cpp sidecar (alias `model_id`). Exactly one of `hf_repo` (download) or
    `model_dir` (fuse, from the RO model PVC) is given. `local_gpu` (e.g. "L4:1") goes on the sidecar.

    `local_gpu` is an explicit argument, NOT the module-level `LOCAL_GPU` env read: this function runs in
    the remote orchestrator pod, where the launcher's `LLM_JURY_LOCAL_GPU` env var is absent. The serve
    image's CUDA build (`cuda=bool(LOCAL_GPU)` in `__main__`) and this GPU request must be derived from the
    *same* value — threading it through as an input is what keeps a CUDA binary from landing on a CPU node."""
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

    extra = ["--n-gpu-layers", "999", "--flash-attn", "on"] if local_gpu else []
    server_cmd = build_fserve_command(
        model_id=model_id, port=SERVER_PORT, model_hf_path=hf_repo, model_dir=model_dir, extra_args=extra
    )

    resources = tolerations = None
    if local_gpu:  # GPU on the sidecar; extended resources need request == limit
        gpu_res = {"nvidia.com/gpu": str(_gpu_count(local_gpu))}
        resources = V1ResourceRequirements(limits=gpu_res, requests=dict(gpu_res))
        tolerations = [V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]

    volume_mounts = volumes = None
    if model_dir is not None:  # fuse mode mounts the RO model PVC; download needs no volume
        volume_mounts = [V1VolumeMount(name="model", mount_path=MODEL_MOUNT, read_only=True)]
        volumes = [
            V1Volume(
                name="model",
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=MODEL_PVC, read_only=True),
            )
        ]

    llama_sidecar = V1Container(
        name="llama",
        image=serve_image_uri,
        restart_policy="Always",  # native sidecar: started before, torn down with, the primary
        command=["/bin/sh", "-c", " ".join(server_cmd)],
        resources=resources,
        volume_mounts=volume_mounts,
    )
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            init_containers=[llama_sidecar],
            volumes=volumes,
            tolerations=tolerations,
        ),
        annotations={"gke-gcsfuse/volumes": "true"} if model_dir is not None else None,
    )


def _ensemble(verdicts: list[dict]) -> dict:
    """Grade + confidence. `grade` = mean score; `agreement` = 1 - min(stdev/2, 1) on [1,5] (1.0
    unanimous, ~0 maximally split). Dead jurors (score=None) are excluded."""
    scores = [v["score"] for v in verdicts if v.get("score") is not None]
    if not scores:
        return {"grade": None, "agreement": None, "n_jurors": 0, "verdicts": verdicts}
    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return {
        "grade": round(statistics.mean(scores), 2),
        "agreement": round(1.0 - min(spread / 2.0, 1.0), 3),
        "n_jurors": len(scores),
        "verdicts": verdicts,
    }


def _report_html(rows: list[dict], juror_names: list[str]) -> str:
    """A self-contained comparison report: one row per graded item, one column per juror (score),
    plus the ensemble grade + agreement, and the per-juror rationales beneath."""
    import html as _h

    def esc(x: object) -> str:
        return _h.escape(str(x))

    head = "".join(f"<th>{esc(n)}</th>" for n in juror_names)
    body = []
    for r in rows:
        by = {v["model"]: v for v in r["ensemble"]["verdicts"]}
        cells = "".join(
            f"<td style='text-align:center'>{esc(by.get(n, {}).get('score', '—'))}</td>" for n in juror_names
        )
        ens = r["ensemble"]
        conf = ens["agreement"]
        color = "#1a7f37" if (conf or 0) >= 0.75 else ("#9a6700" if (conf or 0) >= 0.5 else "#cf222e")
        rationales = "".join(
            f"<div><b>{esc(v['model'])}</b> ({esc(v['score'])}): {esc((v.get('rationale') or '')[:240])}</div>"
            for v in ens["verdicts"]
        )
        body.append(
            f"<tr><td style='max-width:40ch'>{esc(r['question'])}</td>{cells}"
            f"<td style='text-align:center'><b>{esc(ens['grade'])}</b></td>"
            f"<td style='text-align:center;color:{color}'><b>{esc(conf)}</b></td></tr>"
            f"<tr><td colspan='{len(juror_names) + 3}' style='font-size:0.85em;color:#57606a;"
            f"padding:2px 8px 10px'>{rationales}</td></tr>"
        )
    return (
        "<style>table{border-collapse:collapse;font:14px system-ui}"
        "th,td{border:1px solid #d0d7de;padding:6px 10px;vertical-align:top}"
        "th{background:#f6f8fa}</style>"
        "<h2>LLM-as-Jury — comparison report</h2>"
        f"<p>{len(rows)} item(s) · {len(juror_names)} jurors: {esc(', '.join(juror_names))}. "
        "<b>grade</b> = mean score; <b>agreement</b> = 1 - stdev/2 (confidence).</p>"
        f"<table><thead><tr><th>Question</th>{head}<th>grade</th><th>agreement</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _submitter(task: Any, model: str) -> Callable[[str, str], Awaitable[dict]]:
    """Bind a juror `task` + its model arg into a `(question, answer) -> verdict` coroutine fn. A named
    factory (not an inline lambda) so each submitter captures its own `task`/`model` by value — the usual
    loop-variable-capture trap — and carries a precise type the panel's `asyncio.gather` can check."""

    async def submit(question: str, answer: str) -> dict:
        return await task.aio(question, answer, RUBRIC, model)

    return submit


@orch_env.task(report=True)
async def run_jury(
    items: list[dict], serve_image_uri: str, delivery: str, model_dirs: dict[str, str], local_gpu: str = ""
) -> list[dict]:
    """Parent: fan the panel out over every item, ensemble the verdicts, and emit an HTML comparison
    report (`flyte.report`). Returns the per-item ensembles. Jurors run as reusable-actor sub-tasks."""
    import asyncio

    # Build a submitter per juror: local (its sidecar pod template) + Secret + WI archetypes.
    submitters: list[tuple[str, Callable[[str, str], Awaitable[dict]]]] = []
    for name, repo, quant in LOCAL_JURORS:
        pod = (
            _local_pod_template(serve_image_uri, name, model_dir=model_dirs[name], local_gpu=local_gpu)
            if delivery == "fuse"
            else _local_pod_template(serve_image_uri, name, hf_repo=f"{repo}:{quant}", local_gpu=local_gpu)
        )
        t = grade_local.override(short_name=f"juror:{name}", pod_template=pod)
        submitters.append((name, _submitter(t, name)))
    for name, model, secret in SECRET_API_JURORS:
        t = grade.override(short_name=f"juror:{name}", secrets=[secret])
        submitters.append((name, _submitter(t, model)))
    for name, model, ksa in WI_JURORS:
        t = grade.override(short_name=f"juror:{name}", service_account=ksa)
        submitters.append((name, _submitter(t, model)))

    juror_names = [n for n, _ in submitters]
    rows = []
    for item in items:
        verdicts = await asyncio.gather(*(submit(item["question"], item["answer"]) for _, submit in submitters))
        rows.append({"question": item["question"], "answer": item["answer"], "ensemble": _ensemble(list(verdicts))})

    await flyte.report.replace.aio(_report_html(rows, juror_names))
    await flyte.report.flush.aio()
    return [r["ensemble"] for r in rows]


if __name__ == "__main__":
    import argparse
    import asyncio

    p = argparse.ArgumentParser(description="LLM-as-Jury: parent task fans out a panel + emits a comparison report.")
    p.add_argument("--question")
    p.add_argument("--answer")
    p.add_argument("--config", help="Path to the Flyte config for the target deployment (else UCTL_CONFIG / default).")
    p.add_argument("--project")
    p.add_argument("--domain")
    args = p.parse_args()

    flyte.init_from_config(path_or_config=args.config, project=args.project, domain=args.domain)

    # Build the shared llama.cpp serve image once; fuse mode also prefetches each model as an artifact.
    from flyteplugins.llamacpp import build_llama_cpp_image

    serve_uri = flyte.build(build_llama_cpp_image(name="llama-cpp-jury", cuda=bool(LOCAL_GPU))).uri  # type: ignore[arg-type]
    print(f"juror sidecar image ({'cuda' if LOCAL_GPU else 'cpu'}, delivery={LOCAL_DELIVERY}): {serve_uri}")

    model_dirs: dict[str, str] = {}
    if LOCAL_DELIVERY == "fuse":
        from urllib.parse import urlparse

        import flyte.prefetch
        from flyte.io import Dir
        from flyte.remote import Artifact

        # Submit every prefetch up front, THEN wait: `hf_model` returns a non-blocking run handle, so
        # the models download concurrently on the cluster instead of one-at-a-time (waiting inside the
        # loop would serialize them). Wall-clock becomes the slowest single download, not their sum.
        pending = []
        for name, repo, quant in LOCAL_JURORS:
            artifact_name = f"llm-jury-{name}"
            run = flyte.prefetch.hf_model(
                repo=repo,
                artifact_name=artifact_name,
                allow_patterns=[f"*{quant}*"],
                hf_token_key=None,
                resources=flyte.Resources(cpu="2", memory="8Gi", disk="30Gi"),
            )
            print(f"prefetch {repo} ({quant}) -> {artifact_name}: {run.url}")
            pending.append((name, artifact_name, run))
        for name, artifact_name, run in pending:
            run.wait()
            art: Artifact = asyncio.run(Artifact.get.aio(artifact_name, "latest"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            uri = asyncio.run(art.to_python(Dir)).path
            model_dirs[name] = f"{MODEL_MOUNT.rstrip('/')}/{urlparse(uri).path.lstrip('/')}"

    items = [{"question": args.question, "answer": args.answer}] if (args.question and args.answer) else _EXAMPLES
    run = flyte.run(
        run_jury,
        items=items,
        serve_image_uri=serve_uri,
        delivery=LOCAL_DELIVERY,
        model_dirs=model_dirs,
        local_gpu=LOCAL_GPU or "",
    )
    print(f"jury run (report in the UI): {run.url}")
    run.wait()
    for ens in run.outputs()[0]:
        print(f"grade={ens['grade']} agreement={ens['agreement']} (n={ens['n_jurors']})")

# =================================================================================================
# Prerequisites by archetype (provisioned out of band; code references only a Secret / KSA / PVC)
# -------------------------------------------------------------------------------------------------
# LOCAL download (default): none. Set LLM_JURY_LOCAL_GPU=<FAMILY>:<n> (GCP L4:1 / AWS L40S:1) for GPUs.
# LOCAL fuse (LLM_JURY_LOCAL_DELIVERY=fuse): the RO, data-bucket-root model PVC (LLM_JURY_LOCAL_PVC) —
#   same prereq + gcsfuse (GKE) / Mountpoint-S3 (EKS) manifests as ../llamacpp/llamacpp_app_fuse.py.
# SECRET-API-KEY (cloud-agnostic): create the Flyte Secret (`flyte create secret openai-api-key`), then
#   reference it in SECRET_API_JURORS. litellm reads the injected env var (OPENAI_API_KEY, etc.).
# WORKLOAD IDENTITY (keyless):
#   GCP/GKE (Vertex): GSA with roles/aiplatform.user; bind the KSA (roles/iam.workloadIdentityUser) +
#     annotate iam.gke.io/gcp-service-account; set VERTEX_PROJECT / VERTEX_LOCATION.
#   AWS/EKS (Bedrock): IAM role (OIDC trust, bedrock:InvokeModel); annotate the KSA
#     eks.amazonaws.com/role-arn; set AWS_REGION.
# =================================================================================================
