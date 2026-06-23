"""Human-in-the-loop: gate a sensitive Claude tool on a human approval.

Claude decides *whether* to call a tool; Flyte controls *what happens* when it
does. Here a support agent can issue refunds, but the ``issue_refund`` tool — a
durable Flyte task — pauses for a human to approve via the ``flyteplugins-hitl``
web form before it runs. That is a durable gate the agent SDK has no equivalent
for: the run suspends (not a busy-wait), survives restarts, and resumes when a
human responds.

    user: "Please refund my last $42 charge."
    agent -> issue_refund(account=..., amount=42.00)
             ⏸  waits for human approval in the HITL web form
    human: ✅ approve  ->  "refunded $42.00"
           ❌ decline  ->  "declined by a human reviewer"

Run:  python claude_hitl.py   (then approve/decline in the HITL web form)
"""

from pathlib import Path

import flyte
import flyteplugins.hitl as hitl
from flyte._image import PythonWheels

from flyteplugins.agents.claude import function_tool, run_agent

# The Claude Agent SDK bundles the native `claude` CLI in its wheel; the image needs
# the adapter (local wheels under `../dist`) plus the HITL plugin from PyPI.
env = flyte.TaskEnvironment(
    "claude-hitl",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    depends_on=[hitl.env],
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
        .with_pip_packages("flyteplugins-hitl")
    ),
)


@function_tool
@env.task(cache="auto", retries=3)
async def lookup_account(account_id: str) -> str:
    """Look up an account's plan, status and open invoices."""
    return f"account {account_id}: Pro plan, status active, last charge $42.00"


@function_tool
@env.task(retries=3)
async def issue_refund(account_id: str, amount_usd: float) -> str:
    """Issue a refund — pauses for human approval (HITL) before it runs."""
    event = await hitl.new_event.aio(
        f"approve_refund_{account_id}",
        data_type=bool,
        scope="run",
        prompt=f"Approve a ${amount_usd:.2f} refund to account {account_id}?",
    )
    if not await event.wait.aio():
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
    print(f"View at: {run.url}  (approve/decline the refund in the HITL web form)")
    run.wait()
    print(f"Result: {run.outputs()}")
