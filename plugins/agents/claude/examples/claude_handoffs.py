"""Native Claude subagent delegation (handoffs), run durably on Flyte.

The Claude Agent SDK lets a main agent delegate to named subagents — each with its own
prompt and model — and Claude routes the work to the right one. Here a triage prompt
delegates a customer message to a billing or technical-support subagent, and the whole
multi-agent conversation runs inside one durable Flyte task, with the delegation visible
in the task report.

Run:  flyte run claude_handoffs.py support --message "I was charged twice for my subscription this month."
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

from flyteplugins.agents.claude import run_agent

_MODEL = "claude-sonnet-4-5"

env = flyte.TaskEnvironment(
    "claude-handoffs",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_debian_base(name="claude-handoffs").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-claude"]
    ),
)


@env.task(report=True, retries=3)
async def support(message: str) -> str:
    """A main agent that delegates to a billing or technical-support subagent."""
    options = ClaudeAgentOptions(
        agents={
            "billing": AgentDefinition(
                description="Handles billing and refund questions.",
                prompt="You handle billing and refund questions. Be concise and empathetic.",
                model=_MODEL,
            ),
            "technical": AgentDefinition(
                description="Handles technical troubleshooting and setup.",
                prompt="You handle technical troubleshooting. Give clear, numbered steps.",
                model=_MODEL,
            ),
        },
    )
    return await run_agent(
        message,
        instructions=(
            "You are front-desk triage. Delegate to the 'billing' subagent for charges or "
            "refunds, or the 'technical' subagent for bugs or setup. Do not answer yourself."
        ),
        model=_MODEL,
        options=options,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(support, message="I was charged twice for my subscription this month.")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Answer:\n{run.outputs()}")
