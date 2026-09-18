"""The two benchmark arms — with and without System 1 (Jev) — for any task type.

Both arms are task-agnostic: everything task-specific (cases, typed questions,
prompts, tools, grading) comes from the :class:`~tasks._base.TaskSpec` that is
passed in, so customer-support triage, code review and contract review all run
through exactly the same code path.

**Without System 1** — one big System 2 (LLM) call must classify, extract the
entity, pick a tool and write the answer from the raw input. Slow, token-hungry,
and structurally unreliable: the classification comes back as free text that has
to be parsed and coerced.

**With System 1** — the architecture this demo is about, built the way the
TypeSafe docs prescribe:

    (System 1 "Jev")  ONE fan-out call           -> 11-16 atomic typed answers
    (your code)       compose the verdict        -> precedence rules, no prompt
    (your code)       confidence-gated routing   -> auto / review / escalate
    (AI runtime)      execute the tool call      -> Flyte, fanned out
    (System 2 "LLM")  write over the structure   -> compact generation
    (System 1 "Jev")  verify the generation      -> grounded? answer_ok? confident?

The escalate branch never calls System 2 at all: if the typed decision is not
confident enough to act on, the pipeline abstains and hands the case to a human
instead of spending a generation on it.

Every call records latency and tokens, and every unit carries a ``repeat`` index
so the benchmark can run each (task x condition x case) several times and report
variance and decision stability, not just a single sample.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from _config import SYSTEM2_PROVIDERS
from _judge import grade, judge_answer, parse_json_block
from _system1 import JevDecision, system_one
from _system2 import System2Client
from tasks import EvalCase, TaskSpec, get_task
from tasks._base import ESCALATE, REVIEW

WITH = "with_system1"
WITHOUT = "without_system1"


@dataclass
class PipelineOutput:
    label: str = ""
    entity: str | None = None
    tool: str = "none"
    answer: str = ""
    refused: bool = False
    route: str = ""
    confidence: float = 0.0
    severity: float = 0.0
    fired: str = ""
    n_questions: int = 0
    s2_skipped: bool = False


@dataclass
class UnitResult:
    """Metrics for one (task x condition x case x repeat) pipeline run."""

    task: str
    case_id: str
    condition: str  # "with_system1" | "without_system1"
    provider: str  # qwen | sonnet | opus
    repeat: int = 0  # which repetition of this cell this is
    model: str = ""
    latency_s: float = 0.0
    total_tokens: int = 0
    tok_per_s: float = 0.0
    # system 1 (Jev) budget — input/output split so each side can be priced
    jev_calls: int = 0
    jev_tokens: int = 0
    jev_input_tokens: int = 0
    jev_output_tokens: int = 0
    jev_latency_s: float = 0.0
    # system 2 (LLM) budget
    s2_calls: int = 0
    s2_tokens: int = 0
    s2_input_tokens: int = 0
    s2_output_tokens: int = 0
    s2_latency_s: float = 0.0
    # judge budget (identical across arms)
    judge_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_latency_s: float = 0.0
    # System 1 decision shape (with-System-1 arm only)
    jev_questions: int = 0  # typed questions answered by the one fan-out call
    route: str = ""  # auto | review | escalate
    decision_confidence: float = 0.0
    severity: float = 0.0
    fired: str = ""  # atomic signals that came back yes
    s2_skipped: float = 0.0  # 1.0 when the pipeline abstained instead of generating
    # what the pipeline decided (kept so the report can measure run-to-run stability)
    pred_label: str = ""
    pred_entity: str = ""
    pred_tool: str = ""
    # quality
    label_correct: float = 0.0
    entity_correct: float = 0.0
    tool_correct: float = 0.0
    quality: float = 0.0
    guard_correct: float = 0.0
    success: float = 0.0
    output: str = ""
    error: str | None = None


# --------------------------------------------------------------------------- #
# Arm 1 — with System 1                                                       #
# --------------------------------------------------------------------------- #
async def _run_with_system1(task: TaskSpec, case: EvalCase, s2: System2Client, s1_client):
    jev_calls: list[JevDecision] = []

    # (1) ONE Jev call: every atomic symptom, the routing Choice and the severity
    #     Score, evaluated in parallel and in isolation.
    plan = await system_one(case.state, task.structure_questions(case), client=s1_client)
    jev_calls.append(plan)

    # (2) Compose the verdict from the typed answers — plain Python, no prompt.
    decision = task.derive(plan)

    out = PipelineOutput(
        label=decision.label,
        tool=decision.tool,
        route=decision.route,
        confidence=decision.confidence,
        severity=decision.severity,
        fired=", ".join(decision.fired),
        n_questions=decision.n_questions,
        refused=decision.must_refuse,
    )

    # (3) Confidence gate. Below the threshold the pipeline abstains: it hands the
    #     case to a human and spends *no* System 2 generation on it.
    if decision.route == ESCALATE:
        out.answer = task.escalation_note(decision)
        out.s2_skipped = True
        return out, {"s2": None, "jev": jev_calls}

    # (4) The AI runtime executes the tool the composite logic picked.
    tool_result = task.call_tool(case, decision.tool)

    # (5) System 2 only composes prose over the structured decision.
    s2r = await s2.chat(
        task.s2_system_with(),
        json.dumps(
            {
                "input": case.state,
                task.label_name.lower().replace(" ", "_"): decision.label,
                "why": decision.reason,
                "signals_fired": decision.fired,
                "severity": decision.severity,
                "confidence": round(decision.confidence, 3),
                "needs_human_review": decision.route == REVIEW,
                "chosen_tool": decision.tool,
                "tool_result": tool_result,
            },
            default=str,
        )[:6000],
    )
    out.answer = s2r.text
    out.entity = parse_json_block(s2r.text).get("entity")

    # (6) A second small Jev battery verifies the generation it just got back.
    verify = await system_one(
        {
            "request": case.state,
            "plan": {"label": decision.label, "tool": decision.tool, "why": decision.reason},
            "tool_output": tool_result,
            "answer": s2r.text[:1200],
        },
        task.verify_questions(),
        client=s1_client,
    )
    jev_calls.append(verify)

    return out, {"s2": s2r, "jev": jev_calls}


# --------------------------------------------------------------------------- #
# Arm 2 — without System 1                                                    #
# --------------------------------------------------------------------------- #
async def _run_without_system1(task: TaskSpec, case: EvalCase, s2: System2Client):
    s2r = await s2.chat(task.s2_system_without(), json.dumps(case.state, default=str)[:6000])
    parsed = parse_json_block(s2r.text)
    out = PipelineOutput(
        label=task.normalize_label(parsed.get("label")),
        entity=parsed.get("entity"),
        tool=task.normalize_tool(parsed.get("tool")),
        answer=s2r.text,
    )
    return out, {"s2": s2r, "jev": []}


# --------------------------------------------------------------------------- #
# One graded unit                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_case(
    task_key: str,
    case_id: str,
    with_system1: bool,
    provider: str,
    repeat: int = 0,
) -> UnitResult:
    """Run one (task x condition x case x repeat) pipeline and collect metrics."""
    from _system1 import _make_client

    task = get_task(task_key)
    case = task.case(case_id)
    result = UnitResult(
        task=task_key,
        case_id=case_id,
        condition=WITH if with_system1 else WITHOUT,
        provider=provider,
        repeat=repeat,
        model=SYSTEM2_PROVIDERS[provider]["label"],
    )
    s1_client = None
    t0 = time.perf_counter()
    try:
        s1_client = _make_client() if with_system1 else None
        async with System2Client(provider) as s2:
            if with_system1:
                out, m = await _run_with_system1(task, case, s2, s1_client)
            else:
                out, m = await _run_without_system1(task, case, s2)

            for j in m["jev"]:
                result.jev_input_tokens += j.input_tokens
                result.jev_output_tokens += j.output_tokens
                result.jev_tokens += j.input_tokens + j.output_tokens
                result.jev_latency_s += j.latency_s
            result.jev_calls = len(m["jev"])
            result.jev_questions = out.n_questions
            result.route = out.route
            result.decision_confidence = out.confidence
            result.severity = out.severity
            result.fired = out.fired
            result.s2_skipped = float(out.s2_skipped)
            s2r = m["s2"]
            if s2r is not None:
                result.s2_calls = 1
                result.s2_input_tokens = s2r.input_tokens
                result.s2_output_tokens = s2r.output_tokens
                result.s2_tokens = s2r.input_tokens + s2r.output_tokens
                result.s2_latency_s = s2r.latency_s
                if s2r.error:
                    result.error = s2r.error

            # Judge (identical across arms, via a Jev Score).
            judge_client = s1_client or _make_client()
            quality, judge = await judge_answer(out.answer, case.note, judge_client)
            result.judge_input_tokens = judge.input_tokens
            result.judge_output_tokens = judge.output_tokens
            result.judge_tokens = judge.input_tokens + judge.output_tokens
            result.judge_latency_s = judge.latency_s
            if judge_client is not s1_client:
                await judge_client.aclose()

        g = grade(task, case, out.label, out.entity, out.tool, quality)
        result.pred_label = task.normalize_label(out.label)
        result.pred_entity = str(task.normalize_entity(out.entity) or "")
        result.pred_tool = task.normalize_tool(out.tool)
        result.label_correct = float(g.label_correct)
        result.entity_correct = float(g.entity_correct)
        result.tool_correct = float(g.tool_correct)
        result.guard_correct = float(g.guard_correct)
        result.quality = g.quality
        result.output = json.dumps(
            {
                "label": result.pred_label,
                "entity": result.pred_entity,
                "tool": result.pred_tool,
                "route": out.route,
                "fired": out.fired,
                "answer": out.answer[:600],
            },
            default=str,
        )
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    finally:
        if s1_client is not None:
            await s1_client.aclose()
        result.latency_s = time.perf_counter() - t0
        result.total_tokens = result.jev_tokens + result.s2_tokens + result.judge_tokens
        result.tok_per_s = (result.total_tokens / result.latency_s) if result.latency_s > 0 else 0.0
        result.success = float(
            not result.error
            and (result.label_correct + result.entity_correct + result.tool_correct + result.guard_correct) == 4.0
        )
    return result


async def evaluate_many(
    task_key: str,
    case_ids,
    with_system1: bool,
    provider: str,
    repeats: int = 1,
    concurrency: int = 24,
) -> list[UnitResult]:
    """Fan one condition out over cases x repeats concurrently (in-process)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(case_id: str, repeat: int):
        async with sem:
            return await evaluate_case(task_key, case_id, with_system1, provider, repeat)

    return await asyncio.gather(*[one(c, r) for c in case_ids for r in range(repeats)])
