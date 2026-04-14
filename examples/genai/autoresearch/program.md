# Autoresearch program (human-editable)

This file is the lightweight “org chart” for the agent, similar in spirit to
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) `program.md`.
Edit it between runs to change behavior without touching Python.

## Mission

Run short, comparable **language modeling** experiments on the **climbmix** text
corpus (same shard source as upstream `prepare.py`): propose a **PyTorch**
`train.py`, execute it in an isolated sandbox against a **prepared bundle**
(parquet + trained BPE + `autoresearch_runtime`), and record **val_bpb**
(validation bits per byte — **lower is better**). Iterate after failures.

## Constraints

- Training code receives `/var/inputs/bundle` (tar.gz): extract, set
  `AUTORESEARCH_CACHE` to the extract root, import `autoresearch_runtime` as the
  runtime API (tokenizer, packed dataloader, `evaluate_bpb`).
- Must write `/var/outputs/metrics_json` with at least: `val_metric` (use
  **val_bpb** here), `model_name`, `notes`.
- No network calls inside the training sandbox.
- Preserve the model forward contract expected by `evaluate_bpb`:
  `loss = model(x, y, reduction="none")` with per-token losses aligned to `y`.

## Self-healing expectations

1. **Resources** — If the worker reports OOM, increase CPU/RAM per the
   provisioning policy; do not change the scientific goal.
2. **Code bugs** — Use the traceback and prior code to produce a minimal fix.
3. **Literature fetch** — Transient HTTP failures should be retried with
   backoff; if search is empty, proceed with a note in `notes`.

## Workshop extensions

- Increase `num_prepare_shards` or full `AUTORESEARCH_EVAL_TOKENS` for stricter
  parity with upstream overnight runs.
- Swap arXiv for Semantic Scholar or internal RAG.
- Use a GPU task environment and larger batches when CUDA is available.
