"""Example 1 — Jev as the model-based I/O guard (lowest "agenticness").

This is the *System 1 as a fast, typed I/O guard* pattern:

    > "it's like a model-based I/O guard that LLMs try to implement today,
    >   but much faster and flexible... filtering out jailbreak phrases from a
    >   raw piece of text, or parsing different intents and instructions for
    >   your different sub agents in one step, then handing clean JSON over
    >   to the subagents."

One raw input goes in; a **single** TypeSafe request answers the task's whole
battery — 11 to 16 atomic questions — and code composes the verdict, picks the
tool, and gates on confidence before any expensive System 2 generation happens.
Clean typed JSON goes downstream; hostile input is refused early.

The guard is task-agnostic: the same battery guards a support ticket, a
pull-request diff, or a draft contract — only the criteria change, and those come
from the :class:`~tasks._base.TaskSpec`.

``run_guard`` also measures the property the whole design rests on: asking *more*
questions in one call barely costs more time. It re-runs one case with 1, then a
quarter, then half, then the full battery of questions and reports latency per
call and per answer.

Run:

    flyte run examples/typesafe_ai/guardrail_agent.py run_guard --task support
    flyte run examples/typesafe_ai/guardrail_agent.py run_guard --task code_review
    flyte run examples/typesafe_ai/guardrail_agent.py handle_one --task contract \
        --payload '{"intent": "...", "draft": "..."}'
"""

import asyncio
import json
import pathlib

from _runtime import env
from _system1 import system_one
from tasks import DEFAULT_TASK, get_task

import flyte
import flyte.report


async def _guard(task_key: str, state: dict, client=None) -> dict:
    """The System 1 guard: one Jev request, the whole battery, a typed verdict."""
    from _system1 import _make_client

    task = get_task(task_key)
    close = client is None
    client = client or _make_client()
    try:
        dec = await system_one(state, task.structure_questions(), client=client)
    finally:
        if close:
            await client.aclose()

    decision = task.derive(dec)
    return {
        "safe": not decision.must_refuse,
        "route": decision.route,
        "label": decision.label,
        "why": decision.reason,
        "fired": decision.fired,
        "confidence": round(decision.confidence, 3),
        "severity": round(decision.severity, 2),
        "tool": decision.tool,
        "questions": decision.n_questions,
        "jev_tokens": dec.input_tokens + dec.output_tokens,
        "jev_latency_s": round(dec.latency_s, 3),
        # the speculative answers: asked in the same call, not used to decide
        "speculative": {sig.name: round(dec.nouls.get(sig.name) or 0.0, 3) for sig in task.signals if sig.speculative},
    }


async def _question_scaling(task_key: str, state: dict, client) -> list[dict]:
    """Time the same state with 1, ¼, ½ and all of the questions.

    This is the speculative-fan-out claim, measured: the per-call latency should
    barely move as the number of questions grows, so the per-answer cost falls.
    """
    task = get_task(task_key)
    battery = task.structure_questions()
    names = list(battery)
    sizes = sorted({1, max(1, len(names) // 4), max(1, len(names) // 2), len(names)})
    out = []
    for n in sizes:
        subset = {name: battery[name] for name in names[:n]}
        dec = await system_one(state, subset, client=client)
        out.append(
            {
                "questions": n,
                "latency_s": round(dec.latency_s, 3),
                "per_answer_ms": round(dec.latency_s * 1000 / n, 1),
                "tokens": dec.input_tokens + dec.output_tokens,
            }
        )
    return out


@env.task
async def run_tool(task_key: str, case_id: str, tool: str) -> dict:
    """Execute one backend tool as a Flyte task."""
    task = get_task(task_key)
    case = task.case(case_id) if case_id else None
    if case is None:
        return {"note": "no case context; tool skipped"}
    return task.call_tool(case, tool)


@env.task
async def handle_one(task: str = DEFAULT_TASK, payload: str = "", case_id: str = "") -> str:
    """Guard a single input through System 1, then (if safe) answer it."""
    from _system2 import System2Client

    spec = get_task(task)
    state = spec.case(case_id).state if case_id else spec.as_state(payload)
    guard = await _guard(task, state)

    # Confidence-gated routing: only the auto/review tiers reach a generation.
    if guard["route"] == "escalate":
        return json.dumps(
            {
                "accepted": False,
                "reason": "guard signal fired" if not guard["safe"] else "confidence below threshold",
                "handed_to": "human reviewer",
                "guard": guard,
            },
            indent=2,
        )

    result = await run_tool(task, case_id, guard["tool"])
    async with System2Client("sonnet") as s2:
        resp = await s2.chat(
            spec.s2_system_with(),
            json.dumps(
                {
                    "input": state,
                    "label": guard["label"],
                    "why": guard["why"],
                    "signals_fired": guard["fired"],
                    "needs_human_review": guard["route"] == "review",
                    "chosen_tool": guard["tool"],
                    "tool_result": result,
                },
                default=str,
            )[:6000],
        )
    return json.dumps(
        {"accepted": True, "route": guard["route"], "guard": guard, "tool_result": result, "answer": resp.text},
        indent=2,
    )


@env.task(report=True)
async def run_guard(task: str = DEFAULT_TASK, num_cases: int = 0) -> str:
    """Fan the guard out over a task's eval cases and render a guard report."""
    from _system1 import _make_client

    spec = get_task(task)
    cases = spec.cases[:num_cases] if num_cases else spec.cases
    client = _make_client()
    try:
        results = await asyncio.gather(*[_guard(task, c.state, client=client) for c in cases])
        scaling = await _question_scaling(task, cases[0].state, client)
    finally:
        await client.aclose()

    def _badge(route: str) -> str:
        color = {
            "auto": "#0f3d2e;color:#34d399",
            "review": "#43341a;color:#fcd34d",
            "escalate": "#4c1d24;color:#fda4af",
        }[route]
        return f"<span style='padding:1px 8px;border-radius:999px;background:{color}'>{route}</span>"

    rows = "".join(
        f"<tr><td>{c.id}</td><td>{spec.preview(c, 56)}</td><td>{_badge(r['route'])}</td>"
        f"<td>{r['label']}</td><td>{c.label}</td>"
        f"<td>{'✓' if r['label'] == c.label else '✗'}</td>"
        f"<td>{r['confidence']}</td><td>{r['tool']}</td><td>{c.tool}</td>"
        f"<td style='font-size:11px'>{', '.join(r['fired']) or '—'}</td>"
        f"<td>{r['jev_tokens']}</td><td>{r['jev_latency_s']}</td></tr>"
        for c, r in zip(cases, results)
    )
    scale_rows = "".join(
        f"<tr><td>{r['questions']}</td><td>{r['latency_s']}</td><td>{r['per_answer_ms']}</td>"
        f"<td>{r['tokens']}</td></tr>"
        for r in scaling
    )
    avg_lat = sum(r["jev_latency_s"] for r in results) / max(1, len(results))
    correct = sum(1 for c, r in zip(cases, results) if r["label"] == c.label)
    caught = sum(1 for c, r in zip(cases, results) if c.hostile and not r["safe"])
    hostile = sum(1 for c in cases if c.hostile)
    spec_names = ", ".join(sig.name for sig in spec.signals if sig.speculative) or "none"

    tab = flyte.report.get_tab("Guard decisions")
    tab.log(
        f"<p>Task: <b>{spec.label}</b> — {spec.blurb}</p>"
        f"<p>Each input is guarded by <b>one</b> TypeSafe request answering "
        f"<b>{results[0]['questions']}</b> typed questions in parallel; the verdict, the tool and the routing "
        f"tier are then composed in Python. Average Jev latency <b>{avg_lat:.3f}s</b> per input; "
        f"{spec.label_name.lower()} correct on {correct}/{len(cases)}; hostile inputs caught "
        f"{caught}/{hostile}. Speculative questions asked but not used to decide: {spec_names}.</p>"
        "<table><thead><tr><th>case</th><th>input</th><th>route</th>"
        f"<th>{spec.label_name}</th><th>truth</th><th>✓</th><th>conf</th><th>tool</th><th>tool truth</th>"
        "<th>signals fired</th><th>tokens</th><th>lat (s)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p><b>Fan-out scaling</b> — the same input, asked progressively more questions in a single call. "
        "Latency per call should barely move, so the cost per answer falls away:</p>"
        "<table><thead><tr><th>questions in the call</th><th>call latency (s)</th>"
        f"<th>ms per answer</th><th>tokens</th></tr></thead><tbody>{scale_rows}</tbody></table>"
    )
    await flyte.report.flush.aio()

    routes = {r["route"]: sum(1 for x in results if x["route"] == r["route"]) for r in results}
    return (
        f"{spec.key}: {len(cases)} inputs, {results[0]['questions']} typed questions per call; "
        f"label {correct}/{len(cases)}; hostile caught {caught}/{hostile}; routes {routes}; "
        f"avg Jev latency {avg_lat:.3f}s; scaling {[(r['questions'], r['latency_s']) for r in scaling]}"
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(run_guard, task="code_review")
    print(run.name, run.url)
