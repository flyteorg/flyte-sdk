"""Grading for pipeline outputs — identical across task types and across arms.

Structured fields (label / entity / tool) are graded by comparing the pipeline's
output against the task's ground truth.  Free-form answer quality is judged by
Jev (System 1) with a ``Score`` rubric: on-brand for "Jev as the decision logic",
cheap, and — crucially — the *same* judge for the with- and without-System-1
arms, so the comparison stays fair.  Judge cost is reported separately from each
arm's own System 1 budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from _system1 import JevDecision, system_one
from tasks import EvalCase, TaskSpec


@dataclass
class Grading:
    label_correct: bool
    entity_correct: bool
    tool_correct: bool
    guard_correct: bool
    quality: float  # 0..1


QUALITY_RUBRIC = [
    "poor: omits what was required, is unhelpful or wrong",
    "partial: touches the issue but misses key facts or actions",
    "good: addresses the issue and mentions the relevant facts",
    "excellent: complete, accurate, actionable and appropriately worded",
]


def parse_json_block(text: str) -> dict:
    """Best-effort parse of an LLM's JSON output (tolerates code fences/prose)."""
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


async def judge_answer(answer: str, note: str, client) -> tuple[float, JevDecision]:
    """Judge answer quality against the reference note using a Jev Score (0..1)."""
    if not answer:
        return 0.0, JevDecision()
    from typesafe_sdk import Score

    dec = await system_one(
        state={"answer": answer[:1200], "ground_truth": note[:600]},
        questions={
            "quality": Score(
                instructions="How well does the answer satisfy the ground-truth requirements?",
                criteria=QUALITY_RUBRIC,
            )
        },
        client=client,
    )
    return dec.scores.get("quality", 0) / (len(QUALITY_RUBRIC) - 1), dec


def grade(task: TaskSpec, case: EvalCase, label: str, entity, tool: str, quality: float) -> Grading:
    """Grade one pipeline output against a case's ground truth."""
    label = task.normalize_label(label)
    tool = task.normalize_tool(tool)
    if case.expects_tool:
        tool_correct = tool == case.tool and tool != "none"
    else:
        tool_correct = tool == "none"
    return Grading(
        label_correct=label == case.label,
        entity_correct=task.entity_correct(entity, case),
        tool_correct=tool_correct,
        guard_correct=task.guard_correct(label, tool, case),
        quality=quality,
    )
