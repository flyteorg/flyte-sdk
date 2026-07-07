"""Human-in-the-loop: gate a sensitive Claude tool on a human approval.

Claude decides whether to call a tool; Flyte controls what happens when it does.
Here a support agent can issue refunds, but the ``issue_refund`` tool — a durable
Flyte task — pauses on a Flyte condition for a human to approve before it runs. That
is a durable gate the agent SDK has no equivalent for: the run suspends (not a
busy-wait), survives restarts, and resumes when a human signals the condition.

    user: "Please refund my last $42 charge."
    agent -> issue_refund(account=..., amount=42.00)
             ⏸  waits on a Flyte condition for human approval
    human: ✅ approve  ->  "refunded $42.00"
           ❌ decline  ->  "declined by a human reviewer"

Run:  flyte run claude_hitl.py support_agent --request 'Please refund my last $42 charge.' --account_id A-1001
      (add `--local` after `run` to approve/decline the refund condition in your terminal)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.claude import tool, run_agent

env = flyte.TaskEnvironment(
    "claude-hitl",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="claude-hitl")
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-core",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-claude",
                pre=True,
            ),
        )
    ),
)


@tool
@env.task(cache="auto", retries=3)
async def lookup_account(account_id: str) -> str:
    """Look up an account's plan, status and open invoices."""
    return f"account {account_id}: Pro plan, status active, last charge $42.00"


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


@env.task(report=True, retries=3)
async def support_agent(request: str, account_id: str) -> str:
    """A support agent that can issue refunds — but only with human sign-off."""
    return await run_agent(
        f"Customer (account {account_id}) says: {request}",
        tools=[lookup_account, issue_refund],
        instructions=(
            "You are a billing support agent. Look up the account, then if a refund is "
            "warranted call issue_refund. Be concise and state the outcome."
        ),
        model="claude-sonnet-4-5",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(support_agent, request="Please refund my last $42 charge.", account_id="A-1001")
    print(f"View at: {run.url}  (approve/decline the refund condition)")
    run.wait()
    print(f"Result: {run.outputs()}")
