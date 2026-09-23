"""Central configuration for the TypeSafe (System 1) demo.

This module holds the handful of knobs the examples and the benchmark share:

* **System 1** — the TypeSafe model ("Jev"), reached through the public TypeSafe
  Python SDK using the ``TYPESAFE_API_KEY`` secret that exists in the demo org.
* **System 2** — the autoregressive LLM (Qwen 3.8 27B, Claude Sonnet, Claude
  Opus) that does open-ended reasoning and generation.  These are reached through
  a Union-hosted OpenAI-compatible / Anthropic-compatible model gateway using the
  ``DEMO_QWEN_38_27B_API_KEY`` and ``DEMO_GATEWAY_ANTHROPIC_API_KEY`` secrets.
* **The matrix** — which task types, conditions and providers the benchmark
  runs, how many eval cases per task, and how many repeats per cell.  The eval
  cases themselves live with their task type in the ``tasks`` package.
"""

from typing import Any

# --------------------------------------------------------------------------- #
# Secrets (all exist in the demo org, reachable via the default config.yaml)  #
# --------------------------------------------------------------------------- #
TYPESAFE_SECRET = "TYPESAFE_API_KEY"  # System 1 (Jev)
QWEN_SECRET = "DEMO_QWEN_38_27B_API_KEY"  # System 2, Qwen 3.8 27B
ANTHROPIC_SECRET = "DEMO_GATEWAY_ANTHROPIC_API_KEY"  # System 2, Claude Sonnet / Opus

# --------------------------------------------------------------------------- #
# System 2 (LLM gateway) configuration                                        #
# --------------------------------------------------------------------------- #
# The demo keys are Union "virtual keys" for a hosted model gateway.  The exact
# public base URL / model IDs are deployment specific, so we (a) honour an env
# override and (b) probe a list of candidates on first use, caching the first
# one that answers a chat request.  `LLM_GATEWAY_BASE_URL` is the fastest way to
# pin it if your deployment exposes a dedicated host.
LLM_GATEWAY_BASE_URL_ENV = "LLM_GATEWAY_BASE_URL"

# Candidate OpenAI-compatible base URLs, tried in order on first use.
GATEWAY_BASE_URL_CANDIDATES = [
    "https://llm-gateway.apps.demo.hosted.unionai.cloud",
    "https://llm.unionai.cloud",
    "https://models.unionai.cloud",
    "https://gateway.unionai.cloud",
    "https://llm-gateway.unionai.cloud",
    "https://inference.hosted.unionai.cloud",
]

# Provider metadata: how to authenticate and which model to request.
# `api_style`: "openai" -> POST {base}/v1/chat/completions with Bearer key;
#              "anthropic" -> POST {base}/v1/messages with `x-api-key`.
# `timeout_s` / `max_concurrency` / `retry_budget_s` are optional per-provider
# overrides. They exist because the hosted Claude endpoints and a single
# self-hosted GPU are not the same kind of backend: Claude serves a 1.5k-token
# completion in ~25s no matter how many callers there are, while one g6e.2xlarge
# decodes far slower and degrades under parallel load. With the shared 90s
# timeout and a 64-wide fan-out, every long Qwen generation timed out and the
# whole self-hosted arm came back empty (run u4jw7dmf4mkv2c8h748m).
SYSTEM2_PROVIDERS: dict[str, dict[str, Any]] = {
    "qwen": {
        "label": "Qwen 3.8 27B",
        "secret": QWEN_SECRET,
        "env_var": QWEN_SECRET,
        "model": "qwen38-27b-vllm/qwen38-27b",
        "api_style": "openai",
        # One GPU, shared by the whole fan-out: give it room to finish a long
        # generation, and stop the benchmark from queueing 64 streams onto it.
        "timeout_s": 300.0,
        "retry_budget_s": 700.0,
        "max_concurrency": 8,
    },
    "sonnet": {
        "label": "Claude Sonnet",
        "secret": ANTHROPIC_SECRET,
        "env_var": ANTHROPIC_SECRET,
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "api_style": "openai",
    },
    "opus": {
        "label": "Claude Opus",
        "secret": ANTHROPIC_SECRET,
        "env_var": ANTHROPIC_SECRET,
        "model": "anthropic/claude-opus-4-5-20251101",
        "api_style": "openai",
    },
}

# --------------------------------------------------------------------------- #
# The benchmark matrix                                                        #
# --------------------------------------------------------------------------- #
# Three arms, not two.  The original with/without pair could not answer the
# obvious objection — *System 2 could fill that schema itself* — because it
# changed three things at once: who answers the atomic questions, whether the
# verdict is composed in code or in a prompt, and whether the pipeline may
# abstain.  The middle arm holds the last two fixed and varies only the first.
#
#   arm                   fills the battery   composes the verdict   may abstain
#   --------------------  ------------------  ---------------------  -----------
#   with_system1          System 1 (Jev)      TaskSpec.derive()      yes
#   system2_structured    System 2 (LLM)      TaskSpec.derive()      yes
#   without_system1       System 2 (LLM)      the prompt             no
#
# So `with_system1` vs `system2_structured` isolates *who is the better
# schema-filler*, and `system2_structured` vs `without_system1` isolates *what
# moving composition into code is worth on its own*.
WITH_SYSTEM1 = "with_system1"
SYSTEM2_STRUCTURED = "system2_structured"
WITHOUT_SYSTEM1 = "without_system1"

ARMS = [WITH_SYSTEM1, SYSTEM2_STRUCTURED, WITHOUT_SYSTEM1]

# How each arm is named in tables, charts and Flyte action names.
ARM_LABELS: dict[str, dict[str, str]] = {
    WITH_SYSTEM1: {
        "long": "With <b>Jev</b>",
        "short": "Jev",
        "plain": "with-Jev",
        "action": "evaluate_unit_jev",
        "blurb": "System 1 answers the battery; code composes the verdict.",
    },
    SYSTEM2_STRUCTURED: {
        "long": "System 2 <b>structured</b>",
        "short": "S2-struct",
        "plain": "s2-struct",
        "action": "evaluate_unit_s2_structured",
        "blurb": "System 2 answers the same battery; the same code composes the verdict.",
    },
    WITHOUT_SYSTEM1: {
        "long": "Without Jev",
        "short": "No-Jev",
        "plain": "no-S1",
        "action": "evaluate_unit_no_jev",
        "blurb": "One System 2 call does classification, routing and prose in the prompt.",
    },
}

# Arms that route through `TaskSpec.derive()` — i.e. that get composition in code
# and may abstain.  The report uses this to decide which columns mean anything:
# routing tiers and "S2 calls skipped" are undefined for an arm with no gate.
COMPOSED_ARMS = (WITH_SYSTEM1, SYSTEM2_STRUCTURED)

BENCHMARK_CONDITIONS = [(arm, provider) for arm in ARMS for provider in ("qwen", "sonnet", "opus")]

# Task types benchmarked by default (any subset of `tasks.TASK_KEYS`).
BENCHMARK_TASKS = ["support", "code_review", "contract"]

# Number of eval cases per task (a prefix of that task's `cases`).  Keep it
# small so a `flyte run` finishes in a few minutes while staying meaningful.
NUM_EVAL_CASES = 6

# How many times each (task x condition x case) cell is run.  Repeats are what
# turn a single sample into a distribution: the report uses them for latency
# spread (sigma), per-cell confidence intervals, and *decision stability* — how
# often a cell returns the same label across identical runs.  System 1 is
# expected to be far more stable than free-text System 2 classification.
REPEATS_PER_CASE = 3

# Per-unit concurrency used when fanning the benchmark out across the cluster.
FANOUT_CONCURRENCY = 24

# System 2 lives behind a shared gateway, and the self-hosted Qwen backend behind
# it restarts and scales from zero — so transient 4xx/5xx are normal, not bugs.
# Attempts are total (1 = no retry) and spaced by exponential backoff with jitter,
# capped so a unit cannot stall a whole benchmark cell.
SYSTEM2_MAX_RETRIES = 5
SYSTEM2_RETRY_BASE_S = 1.0  # first backoff window; doubles each attempt
SYSTEM2_RETRY_CAP_S = 20.0  # longest single backoff, and the cap on Retry-After
SYSTEM2_RETRY_BUDGET_S = 180.0  # total time one call may spend retrying before it gives up
SYSTEM2_TIMEOUT_S = 90.0

# Sampling temperature for every System 2 call. Pinned to 0 because the benchmark
# reports *decision stability* — the share of repeats that agree on a label — and
# comparing Jev's determinism against a default-temperature sampler would measure
# a sampling-parameter choice rather than a property of either model. Raise it to
# measure how much of the remaining drift is sampling and how much is the prompt.
SYSTEM2_TEMPERATURE = 0.0

# Output ceiling for a System 2 call that has to emit a whole battery. The
# batteries run to ~90 typed answers, and at 4096 the longest were plausibly
# truncating — which grades as "dropped half the fields" and is indistinguishable
# from the model declining to answer. `ChatResult.truncated` now tells them apart.
SYSTEM2_BATTERY_MAX_TOKENS = 8192
