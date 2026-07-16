"""Native Mistral agent handoffs with a human-in-the-loop detail step, durable on Flyte.

Mistral agents support handoffs: an entry agent can hand the conversation off to a
specialist agent (by id) when the request is outside its remit. Here a triage agent routes
a customer message to a billing or technical-support agent — and when that specialist needs
more from the customer, it pauses on a Flyte condition for a human to share a detail, then
resumes with their answer. The whole multi-agent conversation runs inside one durable Flyte
task — each model turn recorded for replay, the routing and the human pause visible in the
task report.

    build_triage_agent  ->  ag_triage  (handoffs -> billing, technical)
    support(ag_triage)   ->  triage hands off; the specialist asks the customer, then resolves
                             ⏸  share_details(...) suspends on a Flyte condition

The specialists are created with the share_details tool declared; the run registers its
executor (a Flyte task that suspends on a condition — durable, not a busy-wait), so the
customer's reply is fed back into the conversation. Conditions are built into Flyte
(``flyte.new_condition``) — no extra plugin needed.

https://docs.mistral.ai/studio-api/agents/handoffs

Run:  flyte run mistral_handoffs.py support_pipeline --message "I was charged twice for my subscription this month."
      (add `--local` right after `run` to share the requested detail in your terminal)
"""

import hashlib
import os

import flyte

from flyteplugins.agents.mistral import run_agent, tool

_MODEL = "mistral-large-latest"

# The function tool the specialists declare at creation. Its executor is registered at run
# time (``run_agent(tools=[share_details])``); the names line up so the run loop dispatches a
# specialist's call to the executor below — even after a handoff.
_SHARE_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "share_details",
        "description": "Ask the customer for a needed detail (e.g. a charge date) and return their reply.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The question to put to the customer."}},
            "required": ["question"],
        },
    },
}

env = flyte.TaskEnvironment(
    "mistral-handoffs",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="mistral_api_key", as_env_var="MISTRAL_API_KEY")],
    image=flyte.Image.from_debian_base(name="mistral-handoffs").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-mistral"]
    ),
)


@tool
@env.task(retries=3)
async def share_details(question: str) -> str:
    """Ask the customer (a human) for a specific detail and return their reply.

    Opens a Flyte condition the run suspends on — durable, not a busy-wait — so a human can
    type the detail the specialist needs, then hands their free-text answer back to the
    conversation. The condition name is derived from the question so a retry re-attaches to
    the same pause instead of starting a new one.
    """
    slug = hashlib.sha1(question.encode(), usedforsecurity=False).hexdigest()[:8]
    condition = await flyte.new_condition.aio(
        f"share_details_{slug}",
        prompt=question,
        data_type=str,
    )
    return await condition.wait.aio()


@env.task(cache="auto", retries=3)
async def build_triage_agent() -> str:
    """Create two specialist agents (each with the share_details tool) and a triage agent.

    Cached, so the server-side agents are created once and reused across runs. The API key
    is read from the environment (a Flyte secret), never an argument. Returns the triage
    agent id — the conversation entry point.
    """
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    billing = await client.beta.agents.create_async(
        model=_MODEL,
        name="billing-agent",
        instructions=(
            "You handle billing and refund questions. Be concise and empathetic. Your reply "
            "is the final answer — you cannot hold a back-and-forth in it. So when you need a "
            "detail from the customer to resolve the issue (the charge date or the invoice id), "
            "get it by calling share_details, then resolve using their answer. Do not ask in "
            "your reply."
        ),
        tools=[_SHARE_DETAILS_TOOL],
    )
    technical = await client.beta.agents.create_async(
        model=_MODEL,
        name="technical-agent",
        instructions=(
            "You handle technical troubleshooting. Give clear, numbered steps. Your reply is "
            "the final answer — you cannot hold a back-and-forth in it. So when you need a "
            "detail from the customer (the exact error message or the steps to reproduce), get "
            "it by calling share_details, then tailor your steps to their answer. Do not ask in "
            "your reply."
        ),
        tools=[_SHARE_DETAILS_TOOL],
    )
    triage = await client.beta.agents.create_async(
        model=_MODEL,
        name="triage-agent",
        instructions=(
            "You are front-desk triage. Read the customer's message and hand off to the "
            "billing agent for charges or refunds, or the technical agent for bugs or "
            "setup. Do not answer the question yourself — hand off."
        ),
        handoffs=[billing.id, technical.id],
    )
    return triage.id


@env.task(report=True, retries=3)
async def support(message: str, triage_agent_id: str) -> str:
    """Drive the triage agent; it hands off to a specialist, which can ask the customer.

    Registering ``share_details`` here installs its executor for the run; the specialists
    declared it at creation, so a handed-off agent's call is dispatched to it by name.
    """
    return await run_agent.aio(message, agent_id=triage_agent_id, tools=[share_details])


@env.task(retries=3)
async def support_pipeline(message: str) -> str:
    """Build the agent team once (cached), then route a customer message through it."""
    triage_agent_id = await build_triage_agent()
    return await support(message, triage_agent_id)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(support_pipeline, message="I was charged twice for my subscription this month.")
    print(f"View at: {run.url}  (share the requested detail to continue)")
    run.wait()
    print(f"Answer:\n{run.outputs()}")
