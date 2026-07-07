"""Multi-agent via OpenAI handoffs — durable, human-gated, fanned out by Flyte.

Handoffs are the OpenAI Agents SDK's in-conversation multi-agent mechanism: a
triage agent decides which specialist should take over and the SDK transfers
control — all inside a single ``Runner.run``. The SDK owns the "which agent"
decision. Flyte does not replace that; it sits underneath and around it, and
this example shows three concrete pieces of that value on a support queue:

1. Durability across a handoff. Every model turn (triage and the specialist
   it hands off to) is a ``flyte.trace`` record and every tool is a durable Flyte
   action. Ticket #2 crashes after its handoff on the first attempt; on retry
   the whole chain replays — you'll see its ``🧠 live model call`` log lines (from
   both agents) vanish on the second attempt, and it finishes fast.
2. Human-in-the-loop on a sensitive tool. ``issue_refund`` pauses on a Flyte
   condition for a human to approve before it runs — a durable gate the SDK has no
   equivalent for. Ticket #1 routes to billing and waits.
3. Per-tool compute + fan-out. ``run_diagnostic`` runs in a higher-CPU
   environment; and the whole triage-with-handoffs agent is one composable Flyte
   task, so the queue is fanned out in parallel, each ticket a durable action.

Net: the SDK does the micro-orchestration (route the dialog); Flyte is the
execution substrate (durable, observable, human-gated, right-sized) and the
macro-orchestration (scale, compose) around it — without giving up handoffs.

Run:  flyte run openai_handoffs.py support_queue --tickets '["Refund $20 on A-1003.", "API returns 500s, A-2210."]'
      (add `--local` after `run` to run locally and approve the refund condition in your terminal)
"""

import asyncio
import os
from pathlib import Path

import flyte
from agents import Agent, RunConfig
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from flyte._image import PythonWheels

from flyteplugins.agents.openai import tool, run_agent

_secrets = [flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")]
_image = (
    flyte.Image.from_debian_base(name="openai-handoffs")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-agents-core",
        ),
    )
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-agents-openai",
        ),
    )
)

# A higher-compute environment for the heavier tool — same image, more CPU. This
# is how one agent's tools get heterogeneous compute: each runs where it should.
heavy_env = flyte.TaskEnvironment(
    "openai-handoffs-heavy",
    resources=flyte.Resources(cpu=2),
    secrets=_secrets,
    image=_image,
)

env = flyte.TaskEnvironment(
    "openai-handoffs",
    resources=flyte.Resources(cpu=1),
    secrets=_secrets,
    depends_on=[heavy_env],
    image=_image,
)


# A model wrapper that prints only when the model is ACTUALLY called (a trace
# miss). On the retry of the crashed ticket these lines are absent across BOTH
# the triage and specialist turns — that absence is the replay across the handoff.
class _LoggingModel(Model):
    def __init__(self, inner: Model):
        self._inner = inner

    async def get_response(self, *args, **kwargs):
        print("  🧠 live model call (recorded for replay)", flush=True)
        return await self._inner.get_response(*args, **kwargs)

    def stream_response(self, *args, **kwargs):
        return self._inner.stream_response(*args, **kwargs)

    async def close(self) -> None:
        await self._inner.close()


class LoggingModelProvider(ModelProvider):
    def __init__(self):
        self._inner = MultiProvider()

    def get_model(self, model_name):
        return _LoggingModel(self._inner.get_model(model_name))

    async def aclose(self) -> None:
        await self._inner.aclose()


# Tools: each is a durable Flyte task, sized and cached independently
@tool
@env.task(retries=3)
async def lookup_account(account_id: str) -> str:
    """Look up an account's plan, status and open invoices."""
    return f"account {account_id}: Pro plan, status active, 2 open invoices"


@tool
@env.task(retries=3)
async def issue_refund(account_id: str, amount_usd: float) -> str:
    """Issue a refund — pauses on a Flyte condition for human approval before it runs."""

    condition = await flyte.new_condition.aio(
        f"approve_refund_{account_id}",
        prompt=f"Approve a ${amount_usd:.2f} refund to account {account_id}?",
        data_type=bool,
    )
    if not await condition.wait.aio():
        return f"Refund to {account_id} was declined by a human reviewer."
    return f"refunded ${amount_usd:.2f} to account {account_id}"


@tool
@heavy_env.task(retries=3)
async def run_diagnostic(service: str) -> str:
    """Run a diagnostic on a service (a heavier task — runs with more CPU)."""
    await asyncio.sleep(0.2)
    return f"{service}: latency normal, error rate 0.2%, last deploy 3h ago"


# Specialist agents -- each owns its tools
billing_agent = Agent(
    name="billing",
    handoff_description="Handles billing, invoices and refunds.",
    instructions="You handle billing. Look up the account, then resolve the issue. Be concise.",
    tools=[lookup_account, issue_refund],
    model="gpt-4.1",
)

tech_agent = Agent(
    name="tech_support",
    handoff_description="Handles outages, errors and technical diagnostics.",
    instructions="You handle technical issues. Run a diagnostic before answering. Be concise.",
    tools=[lookup_account, run_diagnostic],
    model="gpt-4.1",
)

# Triage agent — hands off to the right specialist
triage_agent = Agent(
    name="triage",
    instructions=(
        "You are first-line support. Decide whether the request is a billing or a "
        "technical issue and hand off to the right specialist. Do not answer yourself."
    ),
    handoffs=[billing_agent, tech_agent],
    model="gpt-4.1",
)


@env.task(report=True, retries=3)
async def handle_ticket(ticket: str, crash_first_attempt: bool = False) -> str:
    """One ticket: triage -> handoff -> specialist resolves it.

    Durability spans the handoff: every model turn (triage AND the specialist) is
    a ``flyte.trace`` record and every tool is a durable action. If this crashes
    after the handoff, the retry replays the whole chain.
    """
    attempt = flyte.ctx().attempt_number if flyte.ctx() else 0
    print(f"▶ handle_ticket attempt {attempt}: {ticket[:48]}…", flush=True)

    answer = await run_agent(
        ticket,
        agent=triage_agent,
        # Inject the logging provider so the replay is visible (its lines vanish
        # on the retry); run_agent wraps it with FlyteModelProvider for durability.
        run_config=RunConfig(model_provider=LoggingModelProvider()),
    )

    # Simulate a worker crash after the handoff chain completed, on the first
    # attempt only (on a backend).
    if crash_first_attempt and os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None and attempt == 0:
        raise RuntimeError("💥 simulated crash after the handoff (first attempt only)")

    return answer


@env.task(report=True, retries=3)
async def support_queue(tickets: list[str]) -> list[str]:
    """Flyte fans the triage-with-handoffs agent out over a ticket queue — each
    ticket a durable, independently-retried, parallel action."""
    with flyte.group("tickets"):
        # Crash ticket #2's first attempt to show replay across its handoff.
        results = await asyncio.gather(
            *(handle_ticket(ticket, crash_first_attempt=(i == 1)) for i, ticket in enumerate(tickets))
        )
    return list(results)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        support_queue,
        tickets=[
            "I was double-charged on my last invoice — please refund $20 to account A-1003.",
            "Our API has been returning 500s for the past hour; account A-2210.",
            "The dashboard service feels slow today, can you check it?",
        ],
    )
    print(f"View at: {run.url}")
    run.wait()
    print(f"Resolutions:\n{run.outputs()}")
