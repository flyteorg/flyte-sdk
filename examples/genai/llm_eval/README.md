# LLM-as-Jury

Grade candidate answers with a **panel of diverse LLM jurors**. A **parent task** fans the panel out,
ensembles the verdicts, and emits an **HTML comparison report** (`flyte.report`, visible on the parent
run in the UI): the **mean score** is the grade, and the jurors' **agreement** is a **confidence**
(unanimous → trust it; split → route to a human). Judging is **fail-soft** — a juror that errors just
drops out. Jurors are **reusable actors**; verdicts are **schema-constrained** to `{"score": 1-5, "rationale": "..."}`.

## Three juror archetypes — mix and match

Defaults to **3 self-hosted jurors** (in-perimeter, no external credentials). All three route through
`litellm`, so a juror is just `(model, auth)` — auth is the only difference:

| Archetype                                                | Auth                                                 | Add it                            |
| -------------------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| **1. Local** (default ×3)                         | none — in-perimeter`llama.cpp` sidecar            | edit`LOCAL_JURORS`              |
| **2. Secret API-key** (cloud-agnostic)             | a**Flyte Secret** injects the provider API key | uncomment in`SECRET_API_JURORS` |
| **3. Workload Identity** (cloud-specific, keyless) | the pod's**WI/IRSA `service_account`**       | uncomment in`WI_JURORS`         |

**Local delivery has two modes** (mirrors `../llamacpp/llamacpp_app.py` vs `llamacpp_app_fuse.py`):

- **`download`** (default) — the sidecar pulls the GGUF at startup (`llama-server --hf-repo`); **no
  prerequisites**. Just works.
- **`fuse`** (`LLM_JURY_LOCAL_DELIVERY=fuse`) — a prefetched Model **artifact** read in place over the
  RO data-bucket-root PVC (the stack's `llamacpp_sidecar.py` delivery; needs the PVC).

Set `LLM_JURY_LOCAL_GPU=<FAMILY>:<n>` to run each juror on an **accelerator**.

## Run

```bash
# default: 3 local jurors, download delivery (CPU unless LLM_JURY_LOCAL_GPU is set)
python examples/genai/llm_eval/jury.py

# grade a single ad-hoc pair
python examples/genai/llm_eval/jury.py --question "..." --answer "..."
```

The parent `run_jury` run carries the comparison **report** in the UI.

### Testing across clouds (env only, no code change)

Point `--config`/`--project` at your deployment; the only per-cloud differences are the accelerator
string and the (optional) keyless WI juror:

| Cloud | Accelerator | Optional keyless WI juror |
|---|---|---|
| **GKE (GCP)** | `LLM_JURY_LOCAL_GPU=L4:1` | Vertex: `LLM_JURY_VERTEX_KSA`, `VERTEX_PROJECT`, `VERTEX_LOCATION` |
| **EKS (AWS)** | `LLM_JURY_LOCAL_GPU=L40S:1` | Bedrock: `LLM_JURY_BEDROCK_KSA`, `AWS_REGION` |

```bash
LLM_JURY_LOCAL_GPU=L4:1   python examples/genai/llm_eval/jury.py --config <your-config> --project <your-project>
LLM_JURY_LOCAL_GPU=L40S:1 python examples/genai/llm_eval/jury.py --config <your-config> --project <your-project>
```

## Prerequisites by archetype

Provisioned out of band; the code references only a Secret / KSA / PVC (full steps in [`jury.py`](jury.py)).

- **Local download (default):** none. `LLM_JURY_LOCAL_GPU` for GPUs.
- **Local fuse:** a read-only, data-bucket-root **model PVC** (`LLM_JURY_LOCAL_PVC`, default
  `flyte-metadata-ro`) — same prereq + manifests as [`../llamacpp/llamacpp_app_fuse.py`](../llamacpp/llamacpp_app_fuse.py).
- **Secret API-key:** create the Flyte Secret (e.g. `flyte create secret openai-api-key`) and reference it.
- **Workload Identity:** GKE — a GSA with `roles/aiplatform.user` bound to the KSA + the
  `iam.gke.io/gcp-service-account` annotation, `VERTEX_PROJECT`/`VERTEX_LOCATION`. EKS — an IAM role
  (OIDC trust, `bedrock:InvokeModel`) wired via `eks.amazonaws.com/role-arn`, `AWS_REGION`.

## Architecture / relationship to the other GenAI examples

- **Parent + reusable jurors:** a non-reusable `run_jury` parent (`report=True`) fans out to
  reusable-actor `grade`/`grade_local` sub-tasks — the [annotate-style](../llamacpp) orchestrator +
  warm-actor panel.
- **Local juror** reuses the sidecar delivery of [`../llamacpp/llamacpp_sidecar.py`](../llamacpp/llamacpp_sidecar.py)
  (`build_fserve_command`, `build_llama_cpp_image(cuda=…)`, and the download/fuse duality of the serving Apps).
- A **pipeline/batch** shape (jurors called from tasks), not a scale-to-zero endpoint like the serving Apps.
