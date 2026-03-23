"""Autoresearch-style self-healing agent (PyTorch LM on climbmix shards).

Inspired by `karpathy/autoresearch` — short experiments, one metric (**val_bpb**,
lower is better), iterate — on Flyte with explicit **self-healing** hooks:

1. **Provisioning** — LLM proposes sandbox CPU/memory (and notes for GPU runs);
   OOM triggers a bump and retry.
2. **Training code** — Agent rewrites ``train.py``; execution runs in
   ``flyte.sandbox`` with a **prepared bundle** (parquet shards + BPE tokenizer +
   ``autoresearch_runtime`` from the same ``prepare.py`` as upstream).
3. **Literature** — arXiv Atom API over HTTP with retries/backoff.

Dataset source matches upstream ``prepare.py`` (``karpathy/climbmix-400b-shuffle``
parquet shards on HuggingFace). See:
https://github.com/karpathy/autoresearch/blob/master/prepare.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import flyte
import flyte.errors
import flyte.report
from flyte.io import File

# PyTorch + same stack as ``prepare.py`` (BPE via rustbpe, parquet via pyarrow).
_AR_PIP = (
    "torch",
    "numpy",
    "pyarrow",
    "requests",
    "tiktoken",
    "rustbpe",
)

experiment_env = flyte.TaskEnvironment(
    "autoresearch-experiment",
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    image=(
        flyte.Image.from_debian_base(name="autoresearch-experiment-image").with_pip_packages(
            *_AR_PIP,
        )
    ),
)

agent_env = flyte.TaskEnvironment(
    "autoresearch-agent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="niels-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="autoresearch-agent-image").with_pip_packages(
            "httpx",
            "pydantic-monty",
        )
    ),
    depends_on=[experiment_env],
)

TRAINING_CODE_SYSTEM = """\
You are a **deep learning researcher** agent. You edit an existing ``train.py``
that trains a **PyTorch** causal language model on the **climbmix** text corpus.

The training job runs as a **verbatim** Flyte sandbox (auto_io=False).

**Data bundle** (read-only):
- Input path: ``/var/inputs/bundle`` — a ``.tar.gz`` whose root contains:
  - ``data/`` — parquet shards (column ``text``), same layout as karpathy autoresearch
  - ``tokenizer/`` — ``tokenizer.pkl`` and ``token_bytes.pt`` from BPE training
  - ``autoresearch_runtime.py`` — copy of ``prepare.py``; import as module
    ``autoresearch_runtime`` after setting ``AUTORESEARCH_CACHE`` to the extract root

**Required behavior of ``train.py``:**
1. Extract the tarball to a temp directory; set ``os.environ["AUTORESEARCH_CACHE"]``
   to that directory; ``sys.path.insert(0, that_dir)``; ``import autoresearch_runtime as ar``.
2. Use ``ar.Tokenizer``, ``ar.make_dataloader``, ``ar.evaluate_bpb``, ``ar.MAX_SEQ_LEN``
   so the metric matches upstream **bits-per-byte** semantics.
3. Explore **architecture / optimization** (depth, width, heads, batch, LR, etc.).
   Keep the script single-file and self-contained.
4. Write **one** UTF-8 file ``/var/outputs/metrics_json`` with a single JSON object:
   - ``val_metric`` (float): validation **val_bpb** from ``ar.evaluate_bpb`` — **lower is better**
   - ``model_name`` (str)
   - ``notes`` (str): brief rationale (params, device, steps, etc.)

**Rules:**
- Modeling code: PyTorch (+ stdlib). The sandbox also installs the same pip deps as
  ``prepare.py`` (``torch``, ``numpy``, ``pyarrow``, ``requests``, ``tiktoken``, ``rustbpe``)
  so ``import autoresearch_runtime`` works after extraction.
- No network in the training sandbox.
- Prefer changes that stay compatible with ``evaluate_bpb``'s call signature:
  ``model(x, y, reduction="none")`` returns per-token losses shaped like ``y``.
- For CPU-only sandboxes, keep batch sizes modest; if CUDA is available you may scale up.

Optional env tuning (document in ``notes`` if you rely on it):
- ``AUTORESEARCH_TIME_BUDGET`` — training wall time
- ``AUTORESEARCH_EVAL_TOKENS`` — validation token budget inside ``evaluate_bpb``

Literature context (may be empty):
{literature}

Prepared bundle profile (JSON):
{dataset_profile}

Return **only** a ```python fenced block with the full ``train.py`` script.
"""

PROVISION_SYSTEM = """\
You size a Flyte sandbox for **one short PyTorch language-model training run**
on a prepared text bundle (parquet + tokenizer on disk).

Return **only** a ```json block with exactly:
{{"cpu": <int >= 1>, "memory": "<string like 2Gi, 4Gi, 8Gi, 16Gi>"}}

Use **more CPU** (2-8) and **RAM** (4-16Gi) for deeper/wider Transformers and
larger micro-batches. If the intent mentions GPU, still output CPU/RAM suitable
for a **CPU** sandbox unless the platform provides GPU workers (this example
defaults to CPU PyTorch wheels).

Prepared bundle profile (JSON):
{dataset_profile}

Optional notes from the researcher about the next experiment:
{intent}

Previous resource JSON (may be empty):
{previous_resources}

Last error message (may be empty):
{error}
"""


async def _call_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    import httpx

    api_key = os.environ["ANTHROPIC_API_KEY"]
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 8192,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["content"][0]["text"]


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_resources_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "{}"


def _default_resources() -> flyte.Resources:
    return flyte.Resources(cpu=2, memory="4Gi")


def _read_baseline_train_script() -> str:
    train_path = Path(__file__).with_name("train.py")
    return train_path.read_text(encoding="utf-8")


def _load_prepare_module():
    """Load local ``prepare.py`` without requiring package install."""
    prep_path = Path(__file__).resolve().parent / "prepare.py"
    spec = importlib.util.spec_from_file_location("autoresearch_prepare", prep_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prepare module from {prep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@experiment_env.task
async def build_autoresearch_bundle(
    num_shards: int = 4,
    download_workers: int = 4,
) -> File:
    """Download climbmix shards + train BPE; return a tarball for training sandboxes."""
    cache = tempfile.mkdtemp(prefix="autoresearch-cache-")
    os.environ["AUTORESEARCH_CACHE"] = cache
    try:
        prep = _load_prepare_module()
        prep.download_data(num_shards, download_workers=download_workers)
        prep.train_tokenizer()

        prep_path = Path(__file__).resolve().parent / "prepare.py"
        shutil.copy(prep_path, Path(cache) / "autoresearch_runtime.py")

        fd, tar_path = tempfile.mkstemp(prefix="autoresearch-bundle-", suffix=".tar.gz")
        os.close(fd)
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in sorted(os.listdir(cache)):
                tar.add(os.path.join(cache, name), arcname=name)
        return await File.from_local(tar_path)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


@experiment_env.task
async def profile_autoresearch_bundle(bundle: File) -> str:
    """Summarize tarball contents for the LLM (no full extract)."""
    path = await bundle.download()
    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        members = tar.getmembers()
    parquet = sorted(n for n in names if n.endswith(".parquet"))
    uncompressed = sum(m.size for m in members if m.isfile())
    profile = {
        "n_parquet_files": len(parquet),
        "parquet_examples": parquet[:8],
        "has_tokenizer_pkl": any(n.endswith("tokenizer.pkl") for n in names),
        "has_token_bytes": any(n.endswith("token_bytes.pt") for n in names),
        "has_autoresearch_runtime": any(n.endswith("autoresearch_runtime.py") for n in names),
        "approx_uncompressed_bytes": uncompressed,
    }
    return json.dumps(profile, indent=2)


@flyte.trace
async def write_training_code(
    user_intent: str,
    dataset_profile: str,
    literature: str,
    previous_code: str = "",
    error: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> str:
    system = TRAINING_CODE_SYSTEM.format(
        literature=literature or "(none)",
        dataset_profile=dataset_profile,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": user_intent}]
    if previous_code:
        messages.append({"role": "user", "content": f"Previous code:\n```python\n{previous_code}\n```"})
    if error:
        messages.append({"role": "user", "content": f"Execution error:\n{error}"})
    raw = await _call_llm(model, system, messages)
    return _extract_code(raw)


@flyte.trace
async def provision_resources(
    dataset_profile: str,
    intent: str = "",
    previous_resources: str = "",
    error: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> str:
    system = PROVISION_SYSTEM.format(
        dataset_profile=dataset_profile,
        intent=intent or "(none)",
        previous_resources=previous_resources or "(none)",
        error=error or "(none)",
    )
    raw = await _call_llm(
        model,
        system,
        [{"role": "user", "content": "Propose sandbox resources as specified."}],
    )
    return _extract_resources_json(raw)


@flyte.trace
async def search_arxiv_with_retry(
    query: str,
    max_results: int = 4,
    max_attempts: int = 4,
    workshop_demo_flaky_network: bool = False,
) -> str:
    """Fetch titles/snippets from arXiv Atom API with exponential backoff."""
    import asyncio

    import httpx

    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        if workshop_demo_flaky_network and attempt == 1:
            last_error = "simulated_connect_error"
            await asyncio.sleep(0.2 * (2 ** (attempt - 1)))
            continue

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            lines: list[str] = []
            for entry in root.findall("atom:entry", ns)[:max_results]:
                title = (entry.find("atom:title", ns) or ET.Element("t")).text
                title = " ".join(title.split()) if title else ""
                summary_el = entry.find("atom:summary", ns)
                summary = summary_el.text if summary_el is not None and summary_el.text else ""
                summary = " ".join(summary.split())[:400]
                lines.append(f"- {title}\n  {summary}")
            return "\n".join(lines) if lines else "(no arXiv results; proceed without external context)"
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_error = str(exc)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                last_error = str(exc)
            else:
                raise
        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

    return f"(literature search failed after retries: {last_error})"


@experiment_env.task
async def run_training_subjob(
    code: str,
    bundle: File,
    resources_json: str,
    packages: list[str],
) -> str:
    """Execute one training experiment inside an isolated sandbox (sub-job)."""
    try:
        resources = flyte.Resources(**json.loads(resources_json))
    except (json.JSONDecodeError, TypeError, ValueError):
        resources = _default_resources()

    sandbox = flyte.sandbox.create(
        name="autoresearch-train",
        code=code,
        inputs={"bundle": File},
        outputs={"metrics_json": str},
        packages=packages,
        resources=resources,
        auto_io=False,
    )
    (metrics_json,) = await sandbox.run.aio(bundle=bundle)
    return metrics_json


@agent_env.task(retries=2, report=True)
async def autoresearch_agent(
    research_topic: str,
    user_intent: str = (
        "Explore small Transformer LM architectures on the bundled text data; "
        "minimize val_bpb (bits per byte) within the time budget."
    ),
    num_prepare_shards: int = 4,
    prepare_download_workers: int = 4,
    max_experiment_rounds: int = 4,
    workshop_demo_flaky_network: bool = False,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Orchestrate bundle prep, literature search, provisioning, and training loops."""
    bundle = await build_autoresearch_bundle(num_prepare_shards, prepare_download_workers)
    dataset_profile = await profile_autoresearch_bundle(bundle)

    literature = await search_arxiv_with_retry(
        research_topic,
        workshop_demo_flaky_network=workshop_demo_flaky_network,
    )

    resources_json = await provision_resources(dataset_profile, intent=user_intent)
    packages = list(_AR_PIP)

    code = await write_training_code(
        user_intent=user_intent,
        dataset_profile=dataset_profile,
        literature=literature,
        previous_code=_read_baseline_train_script(),
        model=model,
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for round_idx in range(1, max_experiment_rounds + 1):
        tab = flyte.report.get_tab(f"Experiment {round_idx}")
        tab.replace(f"<pre>Round {round_idx}\n\n<code>{code[:8000]}</code>\n\nresources: {resources_json}</pre>")
        await flyte.report.flush.aio()

        try:
            raw_metrics = await run_training_subjob(code, bundle, resources_json, packages)
            metrics = json.loads(raw_metrics)
            val = float(metrics.get("val_metric", float("inf")))
            record = {"round": round_idx, "metrics": metrics, "resources": resources_json, "code": code}
            history.append(record)
            if best is None or val < float(best["metrics"].get("val_metric", float("inf"))):
                best = record
            break
        except flyte.errors.OOMError as exc:
            err = str(exc)
            history.append({"round": round_idx, "oom": True, "error": err, "code": code})
            resources_json = await provision_resources(
                dataset_profile,
                intent=user_intent,
                previous_resources=resources_json,
                error=err,
                model=model,
            )
            if round_idx == max_experiment_rounds:
                raise RuntimeError(f"OOM persists after {max_experiment_rounds} rounds: {err}") from exc
        except Exception as exc:
            err = str(exc)
            history.append({"round": round_idx, "error": err, "code": code})
            if round_idx >= max_experiment_rounds:
                raise RuntimeError(f"Training failed after {max_experiment_rounds} rounds: {err}") from exc
            code = await write_training_code(
                user_intent=user_intent,
                dataset_profile=dataset_profile,
                literature=literature,
                previous_code=code,
                error=err,
                model=model,
            )

    summary_html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Autoresearch summary</title></head>
<body>
<h1>Autoresearch run</h1>
<p><b>Topic</b>: {research_topic}</p>
<h2>Literature snippet</h2>
<pre>{literature[:4000]}</pre>
<h2>Best result</h2>
<pre>{json.dumps(best, indent=2) if best else "none"}</pre>
<h2>History</h2>
<pre>{json.dumps(history, indent=2)[:12000]}</pre>
</body></html>
"""
    await flyte.report.replace.aio(summary_html)
    await flyte.report.flush.aio()

    return {
        "research_topic": research_topic,
        "literature_excerpt": literature[:2000],
        "dataset_profile": json.loads(dataset_profile),
        "best": best,
        "history": history,
    }


if __name__ == "__main__":
    import asyncio

    flyte.init_from_config()

    async def main():
        run = flyte.run(
            autoresearch_agent,
            research_topic="transformer language modeling bits per byte",
            user_intent=("Try a small but reasonable GPT baseline; tune depth/width if time allows. Minimize val_bpb."),
            num_prepare_shards=4,
            max_experiment_rounds=4,
            workshop_demo_flaky_network=True,
        )
        print(f"View at: {run.url}")
        run.wait()
        print(f"Result: {run.outputs()}")

    asyncio.run(main())
