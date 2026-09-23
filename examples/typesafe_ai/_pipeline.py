"""The two benchmark arms — with and without System 1 (Jev) — for any task type.

Both arms are task-agnostic: everything task-specific (cases, typed questions,
prompts, tools, grading) comes from the :class:`~tasks._base.TaskSpec` that is
passed in, so customer-support triage, code review and contract review all run
through exactly the same code path.

There are **three** arms, because two could not settle the obvious objection —
*System 2 could fill that schema itself* — and a benchmark that cannot answer its
own strongest counter-argument is advocacy, not measurement. The with/without
pair varied three things at once; the middle arm pins two of them down.

**Without System 1** — one big System 2 (LLM) call must classify, extract the
entity, pick a tool and write the answer from the raw input. Slow, token-hungry,
and structurally unreliable: the classification comes back as free text that has
to be parsed and coerced. Composition happens inside the prompt.

**System 2 structured** — System 2 takes Jev's seat. It answers *exactly* the
battery Jev answers, against *exactly* the criteria Jev is given, and then the
same :meth:`~tasks._base.TaskSpec.derive` composes the verdict and the same
:meth:`~tasks._base.TaskSpec.route` gates it. Everything downstream — the tool
call, the prose prompt, the verification pass — is the identical code path, run
by :func:`_compose_and_finish`. The only variable left is who answered the
questions, which is the comparison worth making.

**With System 1** — the architecture this demo is about, built the way the
TypeSafe docs prescribe:

    (System 1 "Jev")  ONE fan-out call           -> 11-16 atomic typed answers
    (your code)       compose the verdict        -> precedence rules, no prompt
    (your code)       confidence-gated routing   -> auto / review / escalate
    (AI runtime)      execute the tool call      -> Flyte, fanned out
    (System 2 "LLM")  write over the structure   -> compact generation
    (System 1 "Jev")  verify the generation      -> grounded? answer_ok? confident?

The escalate branch never calls System 2 for prose at all: if the typed decision
is not confident enough to act on, the pipeline abstains and hands the case to a
human instead of spending a generation on it. Both composed arms may abstain —
withholding the gate from the challenger would have handed the with-Jev arm an
advantage that has nothing to do with answering questions well.

Every call records latency and tokens, and every unit carries a ``repeat`` index
so the benchmark can run each (task x condition x case) several times and report
variance and decision stability, not just a single sample.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from _config import (
    SYSTEM2_BATTERY_MAX_TOKENS,
    SYSTEM2_PROVIDERS,
    SYSTEM2_STRUCTURED,
    WITH_SYSTEM1,
    WITHOUT_SYSTEM1,
)
from _judge import grade, judge_answer, parse_json_block
from _system1 import system_one
from _system2 import ChatResult, System2Client
from tasks import EvalCase, TaskSpec, get_task
from tasks._base import ESCALATE, REVIEW

WITH = WITH_SYSTEM1
STRUCTURED = SYSTEM2_STRUCTURED
WITHOUT = WITHOUT_SYSTEM1


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
    battery_asked: int = 0
    battery_returned: int = 0
    battery_returned_strict: int = 0
    battery_latency_s: float = 0.0
    # Diagnostics that separate a harness artifact from a model property: did the
    # JSON fail to parse at all, and did the model run out of output budget
    # mid-battery? Both look like "dropped fields" downstream.
    parse_failed: bool = False
    truncated: bool = False
    # Jev calls made by this run, in order. Only the with-Jev arm ever appends.
    jev_calls: list = field(default_factory=list)


@dataclass
class UnitResult:
    """Metrics for one (task x condition x case x repeat) pipeline run."""

    task: str
    case_id: str
    condition: str  # "with_system1" | "system2_structured" | "without_system1"
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
    # The structured deliverable every arm owes: how many typed answers were asked
    # for, how many came back, and how long producing them took.
    battery_asked: int = 0
    battery_returned: int = 0
    # The same count, demanding real JSON booleans. The gap between this and
    # `battery_returned` is output-format discipline, not comprehension.
    battery_returned_strict: int = 0
    battery_latency_s: float = 0.0
    # 1.0 when the battery JSON did not parse / the model hit its output ceiling.
    parse_failed: float = 0.0
    truncated: float = 0.0
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
    # 1.0 when the arm actually acted on the case (did not abstain). The
    # denominator for selective accuracy; its mean is the arm's coverage.
    decided: float = 1.0
    output: str = ""
    error: str | None = None


# --------------------------------------------------------------------------- #
# The shared tail: gate -> tool -> prose -> verify                            #
#                                                                             #
# Both composed arms run *this* function, not a copy of it. That is deliberate #
# and it is load-bearing: if the with-Jev arm and its challenger each had their #
# own version of the routing gate, the tool dispatch or the prose prompt, any   #
# difference between them would be unattributable — the comparison would be     #
# measuring two pipelines rather than two answerers. Everything downstream of   #
# the battery is identical by construction, so the only variable left is who    #
# filled the battery in.                                                        #
# --------------------------------------------------------------------------- #
async def _compose_and_finish(
    task: TaskSpec,
    case: EvalCase,
    s2: System2Client,
    decision,
    out: PipelineOutput,
    *,
    s1_client,
    verify_with_jev: bool,
) -> list[ChatResult]:
    """Gate the decision, run its tool, write the prose, verify the result.

    Returns the System 2 calls this tail made (empty when the pipeline abstained).
    ``verify_with_jev`` picks *who* runs the verification battery — Jev for the
    with-Jev arm, System 2 for the structured arm — so the work is always done and
    always billed to the arm that did it. A Jev verification lands in
    ``out.jev_calls``; a System 2 one comes back in the returned list.
    """
    out.label = decision.label
    out.tool = decision.tool
    out.route = decision.route
    out.confidence = decision.confidence
    out.severity = decision.severity
    out.fired = ", ".join(decision.fired)
    out.n_questions = decision.n_questions
    out.refused = decision.must_refuse

    # Confidence gate. Below the threshold the pipeline abstains: it hands the
    # case to a human and spends *no* System 2 generation on it.
    if decision.route == ESCALATE:
        out.answer = task.escalation_note(decision)
        out.s2_skipped = True
        return []

    # The AI runtime executes the tool the composite logic picked.
    tool_result = task.call_tool(case, decision.tool)

    # System 2 composes prose over the structured decision.
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

    # A small second battery verifies the generation it just got back.
    verify_state = {
        "request": case.state,
        "plan": {"label": decision.label, "tool": decision.tool, "why": decision.reason},
        "tool_output": tool_result,
        "answer": s2r.text[:1200],
    }
    if verify_with_jev:
        out.jev_calls.append(await system_one(verify_state, task.verify_questions(), client=s1_client))
        return [s2r]

    verify = await s2.chat(task.s2_system_verify(), json.dumps(verify_state, default=str)[:6000])
    return [s2r, verify]


# --------------------------------------------------------------------------- #
# Arm 1 — with System 1                                                       #
# --------------------------------------------------------------------------- #
async def _run_with_system1(task: TaskSpec, case: EvalCase, s2: System2Client, s1_client):
    out = PipelineOutput(battery_asked=task.battery_size())

    # (1) ONE Jev call: every atomic symptom, the routing Choice and the severity
    #     Score, evaluated in parallel and in isolation.
    plan = await system_one(case.state, task.structure_questions(case), client=s1_client)
    out.jev_calls.append(plan)
    out.battery_returned = plan.n_answers
    out.battery_returned_strict = plan.n_answers  # typed answers: no spelling to get wrong
    out.battery_latency_s = plan.latency_s

    # (2) Compose the verdict from the typed answers — plain Python, no prompt.
    #     (3)-(6) are the shared tail.
    decision = task.derive(plan)
    s2_calls = await _compose_and_finish(task, case, s2, decision, out, s1_client=s1_client, verify_with_jev=True)
    return out, {"s2": s2_calls, "jev": out.jev_calls}


# --------------------------------------------------------------------------- #
# Arm 2 — System 2 fills the same battery, the same code composes it          #
# --------------------------------------------------------------------------- #
async def _run_system2_structured(task: TaskSpec, case: EvalCase, s2: System2Client):
    """System 2 in Jev's seat: same questions, same criteria, same composition.

    This is the arm that answers "could the LLM just do that itself?". It is
    given the identical specification Jev gets — every signal with its true/false
    criteria, every label and tool with its description, every severity tier — and
    its answers are run through the identical :meth:`~tasks._base.TaskSpec.derive`.
    Jev is not called anywhere in this arm; the judge that grades it is the same
    Jev judge that grades the other two, and is billed separately as measurement
    overhead in both cases.
    """
    out = PipelineOutput(battery_asked=task.battery_size())

    battery = await s2.chat(
        task.s2_system_structured(),
        json.dumps(case.state, default=str)[:6000],
        max_tokens=SYSTEM2_BATTERY_MAX_TOKENS,
    )
    parsed = parse_json_block(battery.text)
    out.battery_returned = task.count_returned(parsed)
    out.battery_returned_strict = task.count_returned(parsed, strict=True)
    out.battery_latency_s = battery.latency_s
    out.parse_failed = bool(battery.text) and not parsed
    out.truncated = battery.truncated

    # The join: generated fields -> the shape `derive` consumes. From here the
    # code is the with-Jev arm's, line for line.
    decision = task.derive(task.answers_from_json(parsed))
    s2_calls = await _compose_and_finish(task, case, s2, decision, out, s1_client=None, verify_with_jev=False)
    return out, {"s2": [battery, *s2_calls], "jev": []}


# --------------------------------------------------------------------------- #
# Arm 3 — without System 1                                                    #
# --------------------------------------------------------------------------- #
async def _run_without_system1(task: TaskSpec, case: EvalCase, s2: System2Client):
    s2r = await s2.chat(
        task.s2_system_without(),
        json.dumps(case.state, default=str)[:6000],
        max_tokens=SYSTEM2_BATTERY_MAX_TOKENS,
    )
    parsed = parse_json_block(s2r.text)
    out = PipelineOutput(
        label=task.normalize_label(parsed.get("label")),
        entity=parsed.get("entity"),
        tool=task.normalize_tool(parsed.get("tool")),
        answer=s2r.text,
        battery_asked=task.battery_size(),
        # Counted from what arrived: generated JSON can be short, malformed, or
        # quietly missing half the fields.
        battery_returned=task.count_returned(parsed),
        battery_returned_strict=task.count_returned(parsed, strict=True),
        battery_latency_s=s2r.latency_s,
        parse_failed=bool(s2r.text) and not parsed,
        truncated=s2r.truncated,
    )
    return out, {"s2": [s2r], "jev": []}


# --------------------------------------------------------------------------- #
# One graded unit                                                             #
# --------------------------------------------------------------------------- #
async def evaluate_case(
    task_key: str,
    case_id: str,
    arm: str,
    provider: str,
    repeat: int = 0,
) -> UnitResult:
    """Run one (task x arm x case x repeat) pipeline and collect metrics.

    ``arm`` is one of ``with_system1`` / ``system2_structured`` /
    ``without_system1``. It used to be a bool, back when there were only two arms
    and no way to ask whether System 2 could have filled the schema itself.
    """
    from _system1 import _make_client

    task = get_task(task_key)
    case = task.case(case_id)
    result = UnitResult(
        task=task_key,
        case_id=case_id,
        condition=arm,
        provider=provider,
        repeat=repeat,
        model=SYSTEM2_PROVIDERS[provider]["label"],
    )
    s1_client = None
    t0 = time.perf_counter()
    try:
        # Only the with-Jev arm gets a System 1 client for the pipeline itself.
        # The judge below opens its own if it has to, and is billed separately.
        s1_client = _make_client() if arm == WITH else None
        async with System2Client(provider) as s2:
            if arm == WITH:
                out, m = await _run_with_system1(task, case, s2, s1_client)
            elif arm == STRUCTURED:
                out, m = await _run_system2_structured(task, case, s2)
            elif arm == WITHOUT:
                out, m = await _run_without_system1(task, case, s2)
            else:
                raise ValueError(f"unknown arm {arm!r}; expected one of {WITH}, {STRUCTURED}, {WITHOUT}")

            for j in m["jev"]:
                result.jev_input_tokens += j.input_tokens
                result.jev_output_tokens += j.output_tokens
                result.jev_tokens += j.input_tokens + j.output_tokens
                result.jev_latency_s += j.latency_s
            result.jev_calls = len(m["jev"])
            result.jev_questions = out.n_questions
            result.battery_asked = out.battery_asked
            result.battery_returned = out.battery_returned
            result.battery_returned_strict = out.battery_returned_strict
            result.battery_latency_s = out.battery_latency_s
            result.parse_failed = float(out.parse_failed)
            result.truncated = float(out.truncated)
            result.route = out.route
            result.decision_confidence = out.confidence
            result.severity = out.severity
            result.fired = out.fired
            result.s2_skipped = float(out.s2_skipped)
            # An arm can make several System 2 calls — the structured arm makes up
            # to three (battery, prose, verify) — so the budget is summed over all
            # of them rather than read off a single response.
            for s2r in m["s2"]:
                result.s2_calls += 1
                result.s2_input_tokens += s2r.input_tokens
                result.s2_output_tokens += s2r.output_tokens
                result.s2_tokens += s2r.input_tokens + s2r.output_tokens
                result.s2_latency_s += s2r.latency_s
                if s2r.error and not result.error:
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
        # An abstention cannot succeed by this definition: it deliberately never
        # produces an entity or runs a tool, so it scores 0 on a metric that
        # demands all four fields. That is the right accounting — the case was not
        # handled — but reading it as "the model got it wrong" would be a mistake,
        # and a pipeline that abstains on everything would otherwise look merely
        # bad rather than broken. `decided` is the denominator that tells them
        # apart: success over decided units is *selective* success, and
        # 1 - mean(decided) is how often the arm declined to act at all.
        result.decided = 0.0 if result.s2_skipped else 1.0
    return result


async def evaluate_many(
    task_key: str,
    case_ids,
    arm: str,
    provider: str,
    repeats: int = 1,
    concurrency: int = 24,
) -> list[UnitResult]:
    """Fan one arm out over cases x repeats concurrently (in-process)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(case_id: str, repeat: int):
        async with sem:
            return await evaluate_case(task_key, case_id, arm, provider, repeat)

    return await asyncio.gather(*[one(c, r) for c in case_ids for r in range(repeats)])
