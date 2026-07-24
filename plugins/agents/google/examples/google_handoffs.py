"""Native Google ADK agent transfer (handoffs) with a human-in-the-loop detail step.

ADK agents transfer control to sub-agents: a parent agent gets an automatic
``transfer_to_agent`` tool and routes the conversation to the right specialist. Here a
triage agent transfers a customer message to a billing or technical-support sub-agent —
and when it needs more from the customer, the specialist asks for a detail by pausing on a
Flyte condition for a human to share it, then resumes with their answer. The whole agent
tree runs inside one durable Flyte task: each model turn recorded for replay (the models are
wrapped with ``durable_model``), the transfer and the human pause visible in the task report.

    triage --transfer--> billing / technical
                         ⏸  share_details(...) suspends on a Flyte condition
                         human types the details  ->  specialist resolves using them

The pause is a durable gate the agent SDK has no equivalent for: the run suspends (not a
busy-wait), survives restarts, and resumes when a human answers.

Run:  flyte run google_handoffs.py support --message "I was charged twice for my subscription this month."
      (add `--local` right after `run` to share the requested details in your terminal)
"""

import hashlib

import flyte

from flyteplugins.agents.google import durable_model, run_agent, tool

_MODEL = "gemini-3.1-flash-lite"

env = flyte.TaskEnvironment(
    "google-handoffs",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY")],
    image=flyte.Image.from_debian_base(name="google-handoffs").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-google"]
    ),
)


@tool
@env.task(retries=3)
async def share_details(question: str) -> str:
    """Ask the customer (a human) for a specific detail and return their reply.

    Opens a Flyte condition the run suspends on — durable, not a busy-wait — so a human
    can type the details the specialist needs, then hands their free-text answer back to
    the agent. The condition name is derived from the question so a retry re-attaches to
    the same pause instead of starting a new one.
    """
    slug = hashlib.sha1(question.encode(), usedforsecurity=False).hexdigest()[:8]
    condition = await flyte.new_condition.aio(
        f"share_details_{slug}",
        prompt=question,
        data_type=str,
    )
    return await condition.wait.aio()


@env.task(report=True, retries=3)
async def support(message: str) -> str:
    """A triage agent that transfers to a specialist, which can ask the customer for details."""
    from google.adk.agents import LlmAgent

    billing = LlmAgent(
        name="billing_agent",
        model=durable_model(_MODEL),
        instruction=(
            "You handle billing and refund questions. Be concise and empathetic. Your reply is "
            "the final answer — you cannot hold a back-and-forth in it. So when you need a detail "
            "from the customer to resolve the issue (the charge date or the invoice id), get it by "
            "calling share_details, then resolve using their answer. Do not ask in your reply."
        ),
        tools=[share_details],
    )
    technical = LlmAgent(
        name="technical_agent",
        model=durable_model(_MODEL),
        instruction=(
            "You handle technical troubleshooting. Give clear, numbered steps. Your reply is the "
            "final answer — you cannot hold a back-and-forth in it. So when you need a detail from "
            "the customer (the exact error message or the steps to reproduce), get it by calling "
            "share_details, then tailor your steps to their answer. Do not ask in your reply."
        ),
        tools=[share_details],
    )
    triage = LlmAgent(
        name="triage_agent",
        model=durable_model(_MODEL),
        instruction=(
            "You are front-desk triage. Transfer to billing_agent for charges or refunds, "
            "or technical_agent for bugs or setup. Do not answer yourself — transfer."
        ),
        sub_agents=[billing, technical],
    )
    return await run_agent(message, agent=triage)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(support, message="I was charged twice for my subscription this month.")
    print(f"View at: {run.url}  (share the requested details to continue)")
    run.wait()
    print(f"Answer:\n{run.outputs()}")
