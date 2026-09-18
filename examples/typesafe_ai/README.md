# TypeSafe AI (Jev) — System 1 in agentic Flyte pipelines

An example + benchmark of interleaving **TypeSafe's System One model ("Jev")**
with a **System 2 LLM** inside durable Flyte workloads. Jev answers narrow, typed
questions (yes/no, score, choice) with calibrated probabilities instead of
generated prose, so it is an ideal **model-based I/O guard and decision logic** to
drop in front of — and between — expensive LLM calls.

The architecture being demonstrated:

> **(System 1)** structures the incoming natural-language query →
> **(System 2)** plans/reasons over the structured output →
> **(System 1)** structures each plan step as tool calls →
> **(AI runtime, e.g. Flyte)** executes those tool calls with fan-out really fast →
> **(System 1)** aggregates the fan-out outputs into a structured format → GOTO (System 2)

```
System 1 "Jev"  ── typed answers (Choice/Score/Noul) ──>  decision logic / guard
System 2 "LLM"  ── open-ended reasoning + prose ──────>  generation
Flyte runtime   ── durable, observable, fans out tools ─>  the "AI runtime"
```

## How Jev is used

The pipeline is built around the four patterns the [TypeSafe
docs](https://docs.typesafe.ai/introduction) prescribe, because that is where a
System One model actually wins:

**1. Speculative fan-out.** Every question goes in *one* call — "adding questions
barely changes the response time" — so each task asks 11–16 atomic questions
where a prompt pipeline asks one. Some are *speculative*: urgency and customer
tone on a support ticket, reversibility and owner-review on a diff, whether
counsel is needed on a contract. They are not used to decide anything; they come
along because they are nearly free and a real system wants them.

**2. Atomic decomposition, verdict in code.** A System One model is for
judgments "a highly knowledgeable person could make in a few seconds", not for
multi-factor reasoning. So Jev is never asked *"what is the verdict?"* and left
to reason. It is asked one question per symptom — `backdoor`, `exfiltration`,
`implements_intent`, `callers_consistent`, `out_of_scope_edits`, … — each
evaluated in isolation, and the verdict is composed by a precedence rule in
Python:

```python
planted = [s for s in GUARD_SIGNALS if fired[s]]
if planted:                        return "malicious", f"planted-code signals: {planted}"
if fired["out_of_scope_edits"]:    return "scope_mismatch", "edits the intent never asked for"
if not fired["implements_intent"]: return "incomplete", "the intent is not fully implemented"
...
```

Changing what your team considers blocking is a code edit, not a prompt rewrite.
`logic_check.py` asserts that these rules reproduce ground truth on all 36 cases
when each atomic question is answered correctly — so a benchmark miss is a model
miss, not a logic bug. It runs offline, in under a second, against zero tokens.

**3. Composite scoring.** Severity comes from a `Score` rubric combined with the
symptom signals; the tool to run is composed the same way (a delivery trace only
if the ticket actually contains an id; a dependency audit only if a dependency
changed).

**4. Confidence-gated routing.** `Choice` and `Score` answers carry calibrated
confidence, so routing has a second axis — `auto`, `review`, `escalate` — with
per-task thresholds that scale with risk (a support reply gates at 0.70, a merge
or a signature at 0.85). **Escalation is a real abstention: the pipeline stops
and hands over, and never spends a System 2 generation on a decision it isn't
sure about.** That is why the report tracks "S2 calls skipped" and selective
accuracy (accuracy over just the cases it acted on).

Everything is parameterized by **task type**; only the criteria change.

| Task | `label` vocabulary | What it is |
|---|---|---|
| **`support`** | `billing`, `delivery_status`, `refund`, `account`, `complaint`, `pricing`, `other` | Triage an Acme Courier ticket: intent, order id, backend action, reply. Hostile inputs (extortion, prompt injection) must be refused. |
| **`code_review`** | `approve`, `incomplete`, `scope_mismatch`, `malicious` | Given a PR's **stated intent** and its **diff**, decide whether the change does what it claims — and catch changes that smuggle in malicious code: auth backdoors, env-var exfiltration, typosquatted dependencies, `curl \| sh` post-install hooks, and instructions planted in comments addressed to the reviewer itself. |
| **`contract`** | `approve`, `missing_clause`, `term_mismatch`, `unenforceable`, `refuse` | Given the **agreed term sheet** and the **draft clauses**, decide whether the contract says what the parties agreed and is enforceable — missing liability caps, fees that contradict the term sheet, non-competes void in California — and refuse requests to draft something deceptive (backdating, clauses written to be missed in review). |

Each task ships 12 hand-labelled cases with ground truth for the label, the key
entity, the required tool, a reference note for the quality judge, and a
`hostile` flag for the guard.

One subtlety worth calling out: refusing to *follow* an instruction planted in a
diff is not a reason to stop *analysing* that diff. Code review's tools only read
the input, so a hostile diff still gets scanned — the scan is the evidence the
escalation needs (`refusal_blocks_tools = False`). Support and contract tools
would *act* on the request, so there refusing means touching nothing.

## Layout

| Path | Purpose |
|------|---------|
| `tasks/_base.py` | `EvalCase`, `Signal`, `Decision` + `TaskSpec`: the fan-out battery, the composition and routing hooks, System 2 prompts, tool dispatch, normalizers and grading. |
| `tasks/support.py`, `tasks/code_review.py`, `tasks/contract.py` | One module per task type: criteria, eval cases, and its own backend tools (a diff scanner, a clause library, a redliner — each reads only the input). |
| `tasks/__init__.py` | The task registry (`TASKS`, `get_task`). |
| `logic_check.py` | Offline proof that the composition rules reproduce ground truth on all 36 cases — no API key, no tokens: `python examples/typesafe_ai/logic_check.py`. |
| `guardrail_agent.py` | **Example 1 (lowest agenticness)** — Jev as a fast typed I/O guard: one request flags hostile input and parses label + tool routing before any LLM generation. |
| `tool_agent.py` | **Example 2 (medium)** — Jev plans → Flyte **fans out** tool execution → Jev aggregates → System 2 writes the answer. |
| `durable_agent.py` | **Example 3 (highest)** — a ReAct loop where **Jev decides each action** (Choice), confidence-gates (Score) and detects "enough info" (Noul); tools are Flyte tasks, so the loop is durable/replayable. |
| `benchmark.py` | The experiment: `{with, without System 1} × {Qwen, Sonnet, Opus} × {3 task types}`, **repeated N times per cell**, fanned out across the cluster, rendered as a multi-tab Flyte report. |
| `run_all.py` | Chains the three examples across every task type. |
| `_pipeline.py` | The two arms (with/without System 1) + per-unit metrics — task-agnostic. |
| `_judge.py` | Grading + the Jev `Score` answer-quality judge (identical across arms). |
| `_config.py` | Secrets, System 2 gateway config, the benchmark matrix and repeat count. |
| `_pricing.py` | Published list rates for both vendors, and the per-unit dollar breakdown. |
| `_system1.py` / `_system2.py` | Jev client wrapper / OpenAI-Anthropic-compatible System 2 client. |
| `_report.py` | Multi-tab report rendering and aggregation over repeats. |
| `_runtime.py` | The shared `typesafe-ai` task environment (image + secrets). |

### Adding a task type

Drop a module in `tasks/` that subclasses `TaskSpec` — declare `labels`, `tools`,
`tool_registry` and `cases`, override `normalize_entity`/`entity_correct` if the
entity needs task-specific canonicalization — then register it in
`tasks/__init__.py` and add its key to `BENCHMARK_TASKS`. The three examples and
the whole benchmark pick it up with no other changes.

## The benchmark

```
tasks      = {support, code_review, contract}
conditions = {with System 1 (Jev), without System 1}
providers  = {Qwen 3.8 27B, Claude Sonnet, Claude Opus}
units      = tasks × conditions × providers × cases × repeats
```

Every unit is an independent Flyte action, so the matrix fans out across the
cluster. Defaults are 3 tasks × 6 conditions × 6 cases × 3 repeats = **324
actions**; scale with `--num_cases` / `--repeats`.

The fan-out is shaped so the run graph reads like the experiment: each task
type's units sit inside a `flyte.group(task_key)`, and every action is named for
the arm it ran — **`evaluate_unit_jev`** / **`evaluate_unit_no_jev`** (via
`evaluate_unit.override(short_name=…)`) — so the two arms are distinguishable in
the console without opening a single action.

**Why repeats.** One sample per cell says nothing about variance, and the
interesting claim about System 1 is not only that it is cheaper but that it is
*reproducible*: the same input yields the same typed decision run after run,
where free-text classification drifts. Repeats give:

* **spread** — latency and quality σ, so "faster" comes with an error bar;
* **stability** — per case, the share of repeats that agree on the modal label
  (`agreement`), and the share of cases where every repeat agreed (`unanimous`);
* **tool stability** — the same measure for tool routing.

`repeat` is part of `evaluate_unit`'s signature, so repetitions are always
distinct actions and each one is visible in the run graph.

**Fairness.** Both arms see the same cases, are graded by the same Jev judge,
and every backend tool derives what it reports from the *input it is handed* —
the ticket, the diff, the term sheet — never from a case's ground-truth label.
The with-Jev arm's advantage has to come from routing and running a tool, not
from being told the answer.

Metrics per unit: latency, tokens split into **System 1 / System 2 / judge**
budgets (each with its input/output split, so both sides can be priced),
tokens/second, and quality — label accuracy, entity accuracy, tool
correctness, guard correctness (hostile input refused *and* no tool fired),
judged answer quality (0–1), and end-to-end success (all four structured fields
right at once).

### What it costs

Every token is priced at its vendor's **published list rate**, so the report
answers "what would this pipeline cost to run" in dollars, split System 1 vs
System 2:

| Side | Rate | Source |
|---|---|---|
| **System 1 — TypeSafe (Jev)** | **$0.042 / MTok input** ("$42 per billion input tokens") | [typesafe.ai](https://typesafe.ai) |
| **System 2 — Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) | **$3 / $15** per MTok in/out | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| **System 2 — Claude Opus 4.5** (`claude-opus-4-5-20251101`) | **$5 / $25** per MTok in/out | same |
| **System 2 — Qwen 3.8 27B** | **$0.156 / $3.11** per MTok — *derived*, see below | self-hosted, g6e.2xlarge |

**Pricing a model you host.** Qwen has no per-token list price — it has an
hourly one, on a Union-hosted `g6e.2xlarge` (1× NVIDIA L40S 48 GB). So its
$/MTok is *derived*, and every input is an overridable assumption:

```
$/token = ($/hour ÷ 3600) ÷ (tokens/second) ÷ utilization
```

| Assumption | Value | Where it comes from |
|---|---|---|
| Instance rate | **$2.242/hr** | g6e.2xlarge on-demand. Committed use cuts it hard: $1.413 1-yr, $0.969 3-yr, $2.177 spot. Override with `QWEN_USD_PER_HOUR`. |
| Prefill (input) | **~4,000 tok/s** | Compute-bound at 2 FLOPs/param/token → 54 GFLOP/token for 27B; L40S does 733 TFLOPS dense FP8, ~30% MFU ≈ 220 TFLOPS. |
| Decode (output) | **~200 tok/s** | Memory-bound: FP8 weights are 27 GB (bf16's 54 GB would not fit in 48 GB at all), L40S has 864 GB/s → 32 tok/s single-stream; continuous batching amortises the weight read across the batch (~256 tok/s ideal at batch 8), so 200 tok/s batched with KV headroom. |
| Utilization | **100%** | A deliberate lower bound — see below. |

That gives **$0.156 in / $3.11 out per MTok**: ~19× cheaper input and ~4.8×
cheaper output than Sonnet 4.5 — *if the box stays saturated*.

**The utilization caveat dominates everything else.** An instance bills
wall-clock whether or not it is serving, so the real per-token cost is that
figure divided by duty cycle: $0.31/$6.23 at 50%, $0.62/$12.46 at 25%,
$1.56/$31.14 at 10%. At low duty cycle a self-hosted 27B costs *more* per output
token than Sonnet. That is the structural difference from a hosted API, where
idle time is free — and it is why the report prints the tiers next to the rate
rather than just the headline. `QWEN_UTILIZATION`, `QWEN_PREFILL_TOK_S` and
`QWEN_DECODE_TOK_S` take your measured numbers.

Two other things to know. TypeSafe publishes no separate **output** price, so
Jev's output is charged at the input rate and the report says so; output is ~10%
of Jev's tokens, so the assumption moves the figure by a few percent at most. And
the benchmark's own Jev **judge** is billed in a separate column: it is identical
across both arms by construction, so it is measurement overhead, not product
cost — "Pipeline $" is System 1 + System 2 only.

Neither prompt caching (0.1× input on reads) nor the Batch API (50% off) is used
here, so every token is billed at the base rate. These are list-price
equivalents: a negotiated contract, or a gateway that marks up or absorbs cost,
bills something different.

The report is a **single tab**: one page with an anchor nav over hero KPIs, the
full matrix (grouped by task), per-task plots, with-vs-without,
**confidence-gated routing** (auto / review / escalate, S2 calls skipped, and
accuracy over just the cases acted on), **fan-out economics** (questions per
call, latency per call and per answer, Jev vs System 2 tokens), **cost** (System
1 $ vs System 2 $, $/case, $/1,000 cases), stability, the
System 1 vs System 2 budget, quality, and one section per task type with per-case
detail (modal prediction, routing tier, which signals fired). Cells with no
successful runs render a `BLOCKED` note with the first error instead of `nan`.

## Run

```bash
# The full matrix (produces the multi-tab report)
flyte run examples/typesafe_ai/benchmark.py run_benchmark

# Smaller / larger sweeps
flyte run examples/typesafe_ai/benchmark.py run_benchmark --num_cases 4 --repeats 2
flyte run examples/typesafe_ai/benchmark.py run_benchmark --tasks '["code_review","contract"]' --repeats 5

# One task type at a time
flyte run examples/typesafe_ai/benchmark.py run_task_benchmark --task contract

# The examples, on any task type
flyte run examples/typesafe_ai/guardrail_agent.py run_guard --task code_review
flyte run examples/typesafe_ai/tool_agent.py plan_and_execute --task contract
flyte run examples/typesafe_ai/durable_agent.py durable_agent --task code_review --case_id c2

# Ad-hoc input instead of an eval case
flyte run examples/typesafe_ai/guardrail_agent.py handle_one --task support \
    --payload "my package AC-1042 is late, where is it?"
flyte run examples/typesafe_ai/durable_agent.py durable_agent --task contract \
    --payload '{"intent": "Net 30 payment terms", "draft": "Customer shall pay within ninety (90) days."}'

# Every example, every task type
flyte run examples/typesafe_ai/run_all.py run_everything
```

## Secrets

Uses the default `demo` org from `.flyte/config.yaml`. All three secrets exist in
the demo org:

| Secret | Used for |
|--------|----------|
| `TYPESAFE_API_KEY` | System 1 — the **Jev** TypeSafe API. |
| `DEMO_QWEN_38_27B_API_KEY` | System 2 — Qwen 3.8 27B (OpenAI-compatible). |
| `DEMO_GATEWAY_ANTHROPIC_API_KEY` | System 2 — Claude Sonnet / Opus. |

They are mounted as environment variables on the shared `typesafe-ai` task
environment (`_runtime.py`), so a single pipeline can interleave Jev with any
System 2 model.

## System 2 gateway configuration

System 2 models are reached through the demo model gateway (OpenAI-compatible):

```
base URL : https://llm-gateway.apps.demo.hosted.unionai.cloud
Qwen     : model id "qwen38-27b-vllm/qwen38-27b"            via DEMO_QWEN_38_27B_API_KEY
Claude   : model ids "anthropic/claude-sonnet-4-5-20250929" / "anthropic/claude-opus-4-5-20251101"
           via DEMO_GATEWAY_ANTHROPIC_API_KEY
```

`System2Client` resolves a working endpoint the first time it is used:
* If `LLM_GATEWAY_BASE_URL` is set, it is used directly (fastest to pin).
* Otherwise it probes `_config.GATEWAY_BASE_URL_CANDIDATES` and caches the first
  host that speaks HTTP, falling back between the OpenAI
  (`/v1/chat/completions`) and Anthropic (`/v1/messages`) wire formats.

To inspect the live gateway and confirm model availability:

```bash
flyte run examples/typesafe_ai/probe_gw.py probe_gw
```

## Results — run `u57vnk9qqhxqszzb9j7m`

3 task types x 6 conditions x 5 cases x 2 repeats = **10 runs per condition**,
180 units, 132 ok. This is the first run where **both arms owe the same
artifact** — the full battery of 75 / 86 / 89 typed answers, not just a verdict —
so its without-Jev numbers are not comparable to earlier runs, which asked that
arm for four fields.

| Task | Arm | Provider | Lat μ | $ / case | Label | Quality | Auto/Esc | Success |
|---|---|---|---|---|---|---|---|---|
| Support | **with Jev** | Sonnet | **4.22s** | **$0.00265** | **100%** | **0.89** | 100% / 0% | **100%** |
| Support | without | Sonnet | 11.82s | $0.01756 | 80% | 0.78 | — | 80% |
| Support | **with Jev** | Opus | **4.85s** | **$0.00447** | 100% | **0.87** | 100% / 0% | **100%** |
| Support | without | Opus | 10.04s | $0.02937 | 100% | 0.66 | — | 80% |
| Contract | **with Jev** | Sonnet | **4.40s** | **$0.00253** | 80% | **0.66** | 60% / 20% | **80%** |
| Contract | without | Sonnet | 13.83s | $0.02228 | 80% | 0.58 | — | 40% |
| Contract | **with Jev** | Opus | **3.98s** | **$0.00383** | 80% | 0.67 | 60% / 20% | **80%** |
| Contract | without | Opus | 11.33s | $0.03657 | **100%** | **0.79** | — | 20% |
| Code review | **with Jev** | Sonnet | **1.58s** | **$0.00019** | **80%** | 0.26 | 0% / **100%** | **0%** |
| Code review | without | Sonnet | 14.82s | $0.02539 | 60% | **0.61** | — | **60%** |

**Asking for the whole artifact is what made the difference.** Producing 75–89
typed answers autoregressively costs the one-shot arm **10–15s and $0.018–0.037
per case**. Jev answers the same battery in a single parallel request, so the
with-Jev arm lands at **4–5s and $0.0025–0.0045**: roughly **3x faster and 7–9x
cheaper**, on the same cases, for the same deliverable. That gap barely existed
when the baseline only had to emit four fields — it is a direct function of how
much structure you ask for, which is the whole argument for a System One model.

**Structure is not free for a generative model, and it costs accuracy too.** On
contract review, one-shot Opus still wins on label (100% vs 80% — the
decomposition weakness is real and reproducible), but its end-to-end **success
collapses to 20%** against 80% with Jev: loaded with 89 fields to emit, it drops
entity and tool correctness it used to get right. Support shows the same shape —
label 80% vs 100%, quality 0.78 vs 0.89.

### Two things this run cannot tell you

**The guard has gone from over-firing to total.** Code review escalated
**100% of cases on every provider**, so the pipeline never called System 2 at
all: success 0%, quality 0.26 (the judge is grading a canned escalation note),
cost $0.00019 because nothing ran. It is cheap and fast because it does nothing.
Fixing the guard threshold is now the single highest-value change in the repo.

**Guard coverage is missing for two tasks.** `num_cases=5` takes the first five
cases, and the hostile cases for support and code review sit at positions 7 and
8. Their "guard 100%" means "no hostile case was tested". Only contract (`l5`,
the backdating request) exercised it. Use 8 cases or more for a meaningful guard
column.

At 10 runs per cell nothing here clears the noise threshold on its own; the
latency and cost gaps are large enough to survive it, the label differences are
not.

### Qwen: a client bug, not just saturation

Four Qwen cells failed outright with **HTTP 405**. The gateway probes healthy, so
these were transient — a backend restarting or scaling from zero. The real
problem was in `_system2.py`: the retry loop only caught raised exceptions, so an
HTTP error *response* returned immediately and killed the unit on the first blip.
That is what has been eating the self-hosted arm's units in every run in this
README.

`chat()` now treats a transient gateway as the normal case:

| | |
|---|---|
| retryable statuses | 404, 405, 408, 409, 425, 429, 500, 502, 503, 504, 529 — plus any raised timeout or connection reset |
| backoff | exponential (1s, 2s, 4s, 8s ...), capped at `SYSTEM2_RETRY_CAP_S` = 20s |
| jitter | each wait is drawn from the top half of its window, so 16 concurrent units do not all return in lockstep and re-flood a backend that is still coming up |
| `Retry-After` | honoured when the gateway sends a numeric one, capped at the same 20s |
| stale routes | 404/405/502/503 also drop the cached base URL and re-probe before the next attempt — a restarted gateway usually moved, so waiting alone would not have helped |
| budget | `SYSTEM2_RETRY_BUDGET_S` = 180s total per call, so a gateway that hangs rather than refuses turns into a failed unit instead of a stalled cell |
| attempts | `SYSTEM2_MAX_RETRIES` = 5 total; the error text records how many were spent, so a retry storm can never hide behind a clean-looking failure |

Knobs live in `_config.py`. `ChatResult.attempts` carries the count through to
the pipeline, so a cell that only survived on its third try is still visible.
