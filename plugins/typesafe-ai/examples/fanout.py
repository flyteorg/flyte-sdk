"""Fan a battery out across a queue of tickets.

Two kinds of parallelism stack here, and they are worth telling apart: Flyte runs
one durable task per ticket across the cluster, and inside each task System One
answers the whole battery in a single request. The second one is why the per-case
latency barely moves when you add questions.

    export TYPESAFE_LOCAL_WHEELS=1
    flyte run --root-dir plugins/typesafe-ai/examples plugins/typesafe-ai/examples/fanout.py backlog
"""

import asyncio
import pathlib

import flyte
from _env import env
from triage import Severity, Triage

from flyteplugins.typesafe_ai import CallInfo, Noul, ask_with_info

BACKLOG = [
    "Where is order AC-1042? It was due Tuesday and I need it by Saturday.",
    "I was charged twice for the same subscription. Refund one of them please.",
    "The export button does nothing on Safari. Console shows a 500.",
    "I can't log in since I changed my email. Password reset goes to the old address.",
    "your support is useless and you people are thieves. give me my money.",
    "Hola, necesito cambiar la direccion de envio de mi pedido.",  # codespell:ignore
]


@env.task
async def triage_one(ticket: str) -> tuple[Triage, CallInfo]:
    return await ask_with_info(Triage, {"ticket": ticket})


@env.task(report=True)
async def backlog() -> str:
    """One durable task per ticket; one System One call inside each."""
    results = await asyncio.gather(*[triage_one(t) for t in BACKLOG])

    rows = []
    for ticket, (t, info) in zip(BACKLOG, results):
        fired = sum(1 for answer in vars(t).values() if isinstance(answer, Noul) and answer.at(0.5))
        urgent = t.severity.at_least(Severity.SERIOUS)
        rows.append(
            f"<tr><td>{ticket[:54]}…</td><td>{t.intent.value.value}</td>"
            f"<td>{t.intent.confidence:.2f}</td><td>{t.severity.value.name}</td>"
            f"<td>{'yes' if urgent else 'no'}</td><td>{fired}</td>"
            f"<td>{info.questions}</td><td>{info.latency_s:.2f}s</td></tr>"
        )

    calls = [info for _, info in results]
    total_q = sum(c.questions for c in calls)
    slowest = max(c.latency_s for c in calls)
    tokens = sum(c.input_tokens + c.output_tokens for c in calls)

    flyte.report.get_tab("Backlog").log(
        f"<p>{len(BACKLOG)} tickets, <b>{total_q}</b> typed answers, "
        f"<b>{len(calls)}</b> System One calls — one per ticket, not one per question. "
        f"Slowest call {slowest:.2f}s; {tokens:,} tokens total.</p>"
        "<table><thead><tr><th>ticket</th><th>intent</th><th>conf</th><th>severity</th>"
        "<th>urgent</th><th>facets fired</th><th>questions</th><th>latency</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    await flyte.report.flush.aio()

    urgent = [t for t, _ in results if t.severity.at_least(Severity.SERIOUS)]
    hostile = [t for t, _ in results if t.hostile.at(0.8)]
    return (
        f"{len(BACKLOG)} tickets -> {total_q} typed answers in {len(calls)} calls; "
        f"{len(urgent)} urgent, {len(hostile)} hostile; {tokens:,} tokens; slowest call {slowest:.2f}s"
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(backlog)
    print(run.name, run.url)
    run.wait()
