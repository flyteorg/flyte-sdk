"""Autoresearch-style self-healing agent (PyTorch LM on climbmix shards).

Inspired by `karpathy/autoresearch` — short experiments, one metric (**val_bpb**,
lower is better), iterate — on Flyte with explicit **self-healing** hooks:

1. **Provisioning** — LLM proposes sandbox CPU/memory (and notes for GPU runs);
   OOM triggers a bump and retry.
2. **Training code** — Agent rewrites ``train.py``; execution runs in
   ``flyte.sandbox`` with a **prepared bundle** (parquet shards + BPE tokenizer +
   ``prepare_py``).
3. **Literature** (optional) — arXiv Atom API when ``research_topic`` is set; skipped if ``None``.

Dataset source matches upstream ``prepare.py`` (``karpathy/climbmix-400b-shuffle``
parquet shards on HuggingFace). See:
https://github.com/karpathy/autoresearch/blob/master/prepare.py
"""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flyteplugins.hitl as hitl
import prepare
import train
from autoresearch_types import AutoresearchMetrics, AutoresearchOutput, DatasetProfile, HistoryEntry
from report import build_history_section_html, build_summary_html

import flyte
import flyte.errors
import flyte.report
import flyte.sandbox
from flyte.io import Dir, File

# PyTorch + same stack as ``prepare.py`` (BPE via rustbpe, parquet via pyarrow).
PREPARE_PIP_PACKAGES = [
    "torch",
    "numpy",
    "pyarrow",
    "requests",
    "tiktoken",
    "rustbpe",
]

image = flyte.Image.from_debian_base(name="autoresearch-agent-image").with_pip_packages(
    "httpx",
    "pydantic-monty",
    "fastparquet",
    "flyteplugins-hitl==2.0.11",
    *PREPARE_PIP_PACKAGES,
)

experiment_env = flyte.TaskEnvironment(
    name="autoresearch-experiment",
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    image=image,
)

agent_env = flyte.TaskEnvironment(
    "autoresearch-agent",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=image,
    depends_on=[experiment_env, hitl.env],
)


@dataclass
class AutoresearchBundle:
    data_dir: Dir
    tokenizer_dir: Dir


TRAINING_CODE_SYSTEM = """\
You are a **deep learning researcher** agent. You edit an existing ``train.py``
that trains a **PyTorch** causal language model on the **climbmix** text corpus.

The training job runs as a **verbatim** Flyte sandbox (auto_io=False).

**Data bundle** (read-only):
- Input path: ``/var/inputs/data_tgz`` and ``/var/inputs/tokenizer_tgz`` — ``.tar.gz`` files for
  the data and tokenizer directories.

**Required behavior of ``train.py``:**
1. Extract the tarballs to a temp directory; set ``os.environ["AUTORESEARCH_CACHE"]``
   to that directory; ``sys.path.insert(0, that_dir)``
2. Explore **architecture / optimization** (depth, width, heads, batch, LR, etc.).
   Keep the script single-file and self-contained.
3. Write **one** UTF-8 file ``/var/outputs/metrics_json_str`` with a single JSON object:
   - ``val_metric`` (float): validation **val_bpb** (bits per byte) — **lower is better**
   - ``model_name`` (str)
   - ``notes`` (str): brief rationale (params, device, steps, etc.).

**Rules:**
- Modeling code: PyTorch (+ stdlib). The sandbox also installs the same pip deps as
  ``prepare.py`` (``torch``, ``numpy``, ``pyarrow``, ``requests``, ``tiktoken``, ``rustbpe``)
- ``prepare = _load_prepare_module(PREPARE_PY)`` imports the ``prepare.py`` module. into the sandbox.
- No network in the training sandbox.
- For CPU-only sandboxes, keep batch sizes modest; If you want to run GPU workloads, write code that uses CUDA.
  If you want multi-GPU workloads, write pytorch code that uses multiple GPUs. A resource provisioning step will analyse
  the code and provision the appropriate resources.

Optional env tuning (document in ``notes`` if you rely on it):
- ``AUTORESEARCH_TIME_BUDGET`` — training wall time
- ``AUTORESEARCH_EVAL_TOKENS`` — validation token budget inside ``evaluate_bpb``

Literature context (may be empty):
{literature}

Prepared bundle profile (JSON):
{dataset_profile}

Return a short and descriptive main title of the experiment starting with a '# Title header' and below it a
```python fenced block with the full ``train.py`` script
"""

PROVISION_SYSTEM = """\
You size a Flyte sandbox for **one short PyTorch language-model training run**
on a prepared text bundle (parquet + tokenizer in tarballs).

Return **only** a ```json block with exactly:
{{"cpu": <int >= 1>, "memory": "<string like 2Gi, 4Gi, 8Gi, 16Gi>", "gpu": "<string L4:1, L40s:4>"}}

Use **more GPU (L4:1 vs L40s:1)**, **more CPU** (2-8) and **RAM** (4-16Gi) for deeper/wider Transformers
and larger micro-batches.

Prepared bundle profile (JSON):
{dataset_profile}

Training code (Python):
{training_code}

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
                "max_tokens": 16_384,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["content"][0]["text"]


def _extract_title_and_code(text: str) -> tuple[str, str]:
    title_match = re.search(r"^\s*#\s+(.+?)\s*$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled experiment"
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        return title, code_match.group(1).strip()
    return title, text.strip()


def _extract_resources_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "{}"


def _read_baseline_train_script() -> str:
    assert train.__file__ is not None
    return Path(train.__file__).read_text()


def _metrics_from_training_json(payload: dict[str, Any]) -> AutoresearchMetrics:
    return AutoresearchMetrics(
        val_metric=float(payload.get("val_metric", float("inf"))),
        model_name=str(payload.get("model_name", "")),
        notes=str(payload.get("notes", "")),
    )


def _dataset_profile_prompt_json(profile: DatasetProfile) -> str:
    return json.dumps(asdict(profile), indent=2)


@experiment_env.task(cache="auto")
async def build_autoresearch_bundle(
    num_shards: int = 4,
    download_workers: int = 4,
) -> AutoresearchBundle:
    """Download climbmix shards + train BPE; return a tarball for training sandboxes."""
    cache = tempfile.mkdtemp(prefix="autoresearch-cache-")
    os.environ["AUTORESEARCH_CACHE"] = cache

    prepare.download_data(num_shards, download_workers=download_workers)
    prepare.train_tokenizer()

    data_dir = prepare.data_dir()
    tokenizer_dir = prepare.tokenizer_dir()
    data_dir = await Dir.from_local(data_dir)
    tokenizer_dir = await Dir.from_local(tokenizer_dir)
    return AutoresearchBundle(data_dir, tokenizer_dir)


@experiment_env.task(cache="auto")
async def profile_autoresearch_bundle(bundle: AutoresearchBundle) -> DatasetProfile:
    """Summarize tarball contents for the LLM (no full extract)."""
    data_dir = await bundle.data_dir.download()
    tokenizer_dir = await bundle.tokenizer_dir.download()

    parquet_files = sorted(n.name for n in Path(data_dir).glob("*.parquet"))
    tokenizer_pkl = Path(tokenizer_dir) / "tokenizer.pkl"
    token_bytes_pt = Path(tokenizer_dir) / "token_bytes.pt"
    data_bytes = sum(p.stat().st_size for p in Path(data_dir).glob("**/*") if p.is_file())
    tokenizer_bytes = sum(p.stat().st_size for p in Path(tokenizer_dir).glob("**/*") if p.is_file())
    return DatasetProfile(
        n_parquet_files=len(parquet_files),
        parquet_files=parquet_files,
        has_tokenizer_pkl=tokenizer_pkl.exists(),
        has_token_bytes=token_bytes_pt.exists(),
        data_bytes=data_bytes,
        tokenizer_bytes=tokenizer_bytes,
    )


@flyte.trace
async def write_training_code(
    user_intent: str,
    dataset_profile: DatasetProfile,
    literature: str = "",
    previous_code: str = "",
    error: str = "",
    model: str = "claude-sonnet-4-6",
) -> tuple[str, str]:
    system = TRAINING_CODE_SYSTEM.format(
        literature=literature or "(none)",
        dataset_profile=_dataset_profile_prompt_json(dataset_profile),
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": user_intent}]
    if previous_code:
        messages.append({"role": "user", "content": f"Previous code:\n```python\n{previous_code}\n```"})
    if error:
        messages.append({"role": "user", "content": f"Execution error:\n{error}"})
    raw = await _call_llm(model, system, messages)
    return _extract_title_and_code(raw)


@flyte.trace
async def provision_resources(
    dataset_profile: DatasetProfile,
    training_code: str,
    intent: str = "",
    previous_resources: str = "",
    error: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    system = PROVISION_SYSTEM.format(
        dataset_profile=_dataset_profile_prompt_json(dataset_profile),
        training_code=training_code,
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


@experiment_env.task(cache="auto", retries=3)
async def search_arxiv_with_retry(
    query: str,
    max_results: int = 4,
) -> str:
    """Fetch titles/snippets from arXiv Atom API with exponential backoff.

    If ``query`` is empty or whitespace-only, returns an empty string (no network).
    """
    import httpx

    if not (query and query.strip()):
        return ""

    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    last_error: str | None = None

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

    return f"(literature search failed after retries: {last_error})"


@experiment_env.task(report=True)
async def run_training_subjob(
    title: str,
    code: str,
    bundle: AutoresearchBundle,
    resources_json: str,
    packages: list[str],
) -> str:
    """Execute one training experiment inside an isolated sandbox (sub-job)."""
    try:
        resources = flyte.Resources(**json.loads(resources_json))
    except (json.JSONDecodeError, TypeError, ValueError):
        raise

    await flyte.report.replace.aio(
        f"<h3>{html.escape(title)}</h3><pre><code>{code[:8000]}</code>\n\nresources: {resources_json}</pre>"
    )
    await flyte.report.flush.aio()

    sandbox = flyte.sandbox.create(
        name="autoresearch-train",
        code=code,
        inputs={"data_tgz": File, "tokenizer_tgz": File, "prepare.py": File},
        outputs={"metrics_json_str": str},
        packages=packages,
        resources=resources,
        auto_io=False,
    )

    data_dir = await bundle.data_dir.download()
    tokenizer_dir = await bundle.tokenizer_dir.download()

    async def dir_to_tgz(filename: str, dir_path: str) -> File:
        import tarfile

        temp_dir = tempfile.mkdtemp(prefix="autoresearch-temp-")
        temp_file = Path(temp_dir) / f"{filename}.tar.gz"
        with tarfile.open(temp_file, "w:gz") as tar:
            # Add each top-level entry with arcname only (no extra folder).
            for entry in sorted(os.listdir(dir_path)):
                tar.add(os.path.join(dir_path, entry), arcname=entry)
        return await File.from_local(temp_file)

    metrics_json_str = await sandbox.run.aio(
        data_tgz=await dir_to_tgz("data", data_dir),
        tokenizer_tgz=await dir_to_tgz("tokenizer", tokenizer_dir),
        **{"prepare.py": await File.from_local(prepare.__file__)},
    )
    return metrics_json_str


@agent_env.task(retries=3, report=True)
async def autoresearch_agent(
    research_topic: str | None = None,
    user_intent: str = (
        "Explore small Transformer LM architectures on the bundled text data; "
        "minimize val_bpb (bits per byte) within the time budget."
    ),
    num_prepare_shards: int = 4,
    prepare_download_workers: int = 4,
    max_experiment_rounds: int = 4,
    model: str = "claude-sonnet-4-6",
) -> AutoresearchOutput:
    """Orchestrate bundle prep, optional literature (arXiv), provisioning, and training loops."""
    bundle = await build_autoresearch_bundle(num_prepare_shards, prepare_download_workers)
    dataset_profile = await profile_autoresearch_bundle(bundle)

    literature = ""
    if research_topic:
        literature = await search_arxiv_with_retry(research_topic)

    title, code = await write_training_code(
        user_intent=user_intent,
        dataset_profile=dataset_profile,
        literature=literature,
        previous_code=_read_baseline_train_script(),
        model=model,
    )
    resources_json = await provision_resources(dataset_profile, training_code=code, intent=user_intent)

    history: list[HistoryEntry] = []
    best: HistoryEntry | None = None

    for round_idx in range(1, max_experiment_rounds + 1):
        err = ""
        try:
            raw_metrics = await run_training_subjob.override(short_name=f"experiment-{round_idx}")(
                title, code, bundle, resources_json, PREPARE_PIP_PACKAGES
            )
            metrics = _metrics_from_training_json(json.loads(raw_metrics))
            entry = HistoryEntry(
                round=round_idx,
                code=code,
                title=title,
                metrics=metrics,
                resources=resources_json,
            )
            history.append(entry)
            prev_val = best.metrics.val_metric if best and best.metrics is not None else float("inf")
            if best is None or metrics.val_metric < prev_val:
                best = entry
        except flyte.errors.OOMError as exc:
            err = str(exc)
            history.append(HistoryEntry(round=round_idx, code=code, title=title, oom=True, error=err))
            resources_json = await provision_resources(
                dataset_profile,
                training_code=code,
                intent=user_intent,
                previous_resources=resources_json,
                error=err,
                model=model,
            )
            if round_idx == max_experiment_rounds:
                raise RuntimeError(f"OOM persists after {max_experiment_rounds} rounds: {err}") from exc
            continue
        except Exception as exc:
            err = str(exc)
            history.append(HistoryEntry(round=round_idx, code=code, title=title, error=err))

        title, code = await write_training_code(
            user_intent=user_intent,
            dataset_profile=dataset_profile,
            literature=literature,
            previous_code=code,
            error=err,
            model=model,
        )
        resources_json = await provision_resources(dataset_profile, training_code=code, intent=user_intent)

        history_section = build_history_section_html(history)
        tab = flyte.report.get_tab("History")
        tab.replace(history_section)
        await flyte.report.flush.aio()

    summary_html = build_summary_html(user_intent, research_topic, literature, best, history)
    await flyte.report.replace.aio(summary_html)
    await flyte.report.flush.aio()

    return AutoresearchOutput(
        research_topic=research_topic,
        literature_excerpt=literature[:2000] if literature else None,
        dataset_profile=dataset_profile,
        best=best,
        history=history,
    )


@agent_env.task(retries=0, report=True)
async def infinite_research_loop(
    research_topic: str | None = None,
    user_intent: str = (
        "Explore small Transformer LM architectures on the bundled text data; "
        "minimize val_bpb (bits per byte) within the time budget."
    ),
    num_prepare_shards: int = 4,
    max_experiment_rounds: int = 4,
    model: str = "claude-sonnet-4-6",
) -> None:
    event = await hitl.new_event.aio(
        "continue_research_loop_event",
        data_type=bool,
        scope="run",
        prompt="Do you want to continue the research loop?",
    )
    while True:
        await autoresearch_agent(
            research_topic=research_topic,
            user_intent=user_intent,
            num_prepare_shards=num_prepare_shards,
            max_experiment_rounds=max_experiment_rounds,
            model=model,
        )
        continue_research_loop = await event.wait()
        if not continue_research_loop:
            break


if __name__ == "__main__":
    import asyncio

    flyte.init_from_config()

    async def main():
        run = flyte.run(
            infinite_research_loop,
            # research_topic="transformer language modeling bits per byte",
            research_topic=None,
            user_intent="Try a small but reasonable GPT baseline; tune depth/width if time allows. Minimize val_bpb.",
            num_prepare_shards=4,
            max_experiment_rounds=3,
        )
        print(f"View at: {run.url}")
        run.wait()
        print(f"Result: {run.outputs()}")

    asyncio.run(main())
