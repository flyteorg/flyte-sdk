# Autoresearch-style self-healing agent (PyTorch LM)

This folder implements an agent in the spirit of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): **short
experiments**, a **single scalar metric** (**val_bpb** — validation bits per
byte, **lower is better**), and **iteration** after failure — on Flyte with
explicit recovery paths.

## Dataset (same source as upstream `prepare.py`)

Text data comes from the **climbmix** parquet shards referenced in upstream
[`prepare.py`](https://github.com/karpathy/autoresearch/blob/master/prepare.py)
(HuggingFace `karpathy/climbmix-400b-shuffle`). This example downloads a
configurable number of training shards plus the **pinned validation shard**
(`shard_06542`), trains a **BPE tokenizer** (rustbpe + tiktoken), and packs
shards + tokenizer + a copy of `prepare.py` (as `autoresearch_runtime.py`) into
a **tarball** consumed by the training sandbox.

## Files

| File | Role |
|------|------|
| `program.md` | Human-editable “research org” instructions (like upstream `program.md`). |
| `prepare.py` | Download shards, train BPE, dataloaders, **evaluate_bpb** (aligned with upstream). |
| `train.py` | Single editable **PyTorch** training script the agent rewrites each round. |
| `autoresearch_agent.py` | Flyte workflow: bundle prep → literature → provision → training sub-job → heal loops. |

## Local prep (optional)

To materialize the cache under `~/.cache/autoresearch/` without Flyte:

```bash
cd /path/to/flyte-sdk
uv run python examples/genai/autoresearch/prepare.py --num-shards 4
```

## Run the agent

Requires `flyte.init_from_config()` and an Anthropic API key (same secret name
as other `examples/genai` agents: `niels-anthropic-api-key` → `ANTHROPIC_API_KEY`).

The first task run downloads parquet shards from HuggingFace and trains the
tokenizer — **large network use** and several minutes of CPU/RAM.

```bash
cd /path/to/flyte-sdk
uv run python examples/genai/autoresearch/autoresearch_agent.py
```

Tune `num_prepare_shards` (in code or via overrides) for faster/smaller bundles
vs. richer training data.

Set `workshop_demo_flaky_network=False` for production arXiv access.

### Environment tuning

| Variable | Effect |
|----------|--------|
| `AUTORESEARCH_CACHE` | Root for `data/` and `tokenizer/` (set automatically in sandboxes). |
| `AUTORESEARCH_EVAL_TOKENS` | Validation token budget inside `evaluate_bpb` (default matches upstream scale; `train.py` sets a smaller default for Flyte demos). |
| `AUTORESEARCH_TIME_BUDGET` | Wall-clock training seconds in baseline `train.py`. |

### Dependencies note

`prepare.py` / bundles require **`rustbpe`** (Rust extension). If image builds
fail, install a Rust toolchain or use a platform with prebuilt wheels; see
upstream [autoresearch](https://github.com/karpathy/autoresearch) discussion.

## Self-healing hooks

1. **Provisioning** — `provision_resources` proposes CPU/memory from bundle
   profile; **OOM** triggers a new proposal with the error in context.
2. **Training code** — `run_training_subjob` runs the rewritten `train.py` in an
   isolated sandbox; failures trigger **LLM code repair** with traceback.
3. **Literature** — `search_arxiv_with_retry` uses **backoff** on timeouts and
   connection errors (optional **simulated** flake for demos).

## Relationship to `examples/genai/mle_agent`

- `mle_tool_builder_agent.py` — sandbox **resource** adjustment on OOM + **code**
  repair loop; dependency inference.
- `mle_orchestrator_agent.py` — **Monty** orchestration repair loop over
  **fixed** tools.

`autoresearch_agent.py` combines **retrieval with retries**, **provisioning**,
and a **training sandbox sub-job** aimed at iterative “research” runs on **text
tokens** and **val_bpb**.
