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
SYSTEM2_PROVIDERS = {
    "qwen": {
        "label": "Qwen 3.8 27B",
        "secret": QWEN_SECRET,
        "env_var": QWEN_SECRET,
        "model": "qwen38-27b-vllm/qwen38-27b",
        "api_style": "openai",
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
# Conditions: {with System 1, without System 1} x {System 2 provider}.  The
# *task types* (customer support / code review / contract review) are the third
# axis and live in the `tasks` package; see `tasks/__init__.py`.
WITH_SYSTEM1 = True
WITHOUT_SYSTEM1 = False
BENCHMARK_CONDITIONS = [
    (WITH_SYSTEM1, "qwen"),
    (WITH_SYSTEM1, "sonnet"),
    (WITH_SYSTEM1, "opus"),
    (WITHOUT_SYSTEM1, "qwen"),
    (WITHOUT_SYSTEM1, "sonnet"),
    (WITHOUT_SYSTEM1, "opus"),
]

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
