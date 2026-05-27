"""HITL approval agent — pauses for a human before running sensitive tools.

This example wires :class:`flyte.ai.agents.Agent` to the
``flyteplugins-hitl`` plugin so that designated tools require explicit human
approval before they execute.

Pattern
-------

1. Mark sensitive tools by wrapping them in an :class:`AgentTool` with
   ``requires_approval=True``.
2. When the LLM asks the agent to run such a tool, the harness invokes the
   ``approval_callback`` and waits for a boolean decision.
3. The default callback uses ``flyteplugins.hitl.new_event`` to spin up a
   human-input web form scoped to the current run. If denied, the agent
   receives a synthetic tool message explaining the rejection so it can
   recover gracefully.

Deploy::

    flyte deploy examples/agents/flyte_agent/hitl_approval_agent.py
    flyte run examples/agents/flyte_agent/hitl_approval_agent.py concierge \
        --request "Refund order #12345 to the customer."
"""

from __future__ import annotations

from typing import Any

import flyteplugins.hitl as hitl

import flyte
from flyte.ai.agents import Agent, AgentTool

env = flyte.TaskEnvironment(
    name="hitl-agent",
    image=(
        flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
            "flyteplugins-hitl>=2.1.3",
            "fastapi",
            "uvicorn",
            "python-multipart",
            "litellm",
        )
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    depends_on=[hitl.env],
)


# ---------------------------------------------------------------------------
# Read-only tools (no approval needed)
# ---------------------------------------------------------------------------


@env.task
async def lookup_order(order_id: str) -> dict[str, Any]:
    """Return order metadata for *order_id* (stub)."""
    return {
        "order_id": order_id,
        "customer": "alex@example.com",
        "amount_usd": 49.99,
        "status": "shipped",
        "delivered_at": "2026-05-20T18:30:00Z",
    }


# ---------------------------------------------------------------------------
# Sensitive tool — gated behind human approval
# ---------------------------------------------------------------------------


@env.task
async def issue_refund(order_id: str, amount_usd: float, reason: str) -> dict[str, Any]:
    """Issue a refund to the customer for *order_id*.

    This is the actual side-effectful call to your payments backend. The HITL
    approval wrapping (below) ensures the LLM cannot trigger it without a
    human in the loop.
    """
    return {
        "order_id": order_id,
        "refunded_usd": amount_usd,
        "reason": reason,
        "status": "refunded",
    }


async def _issue_refund_tool_exec(args: dict[str, Any]) -> Any:
    """Bridge dict-args → ``issue_refund.aio`` keyword args."""
    return await issue_refund.aio(**args)


_refund_tool = AgentTool(
    name="issue_refund",
    description=issue_refund.func.__doc__.strip().splitlines()[0],  # type: ignore[union-attr]
    parameters=issue_refund.json_schema,
    execute=_issue_refund_tool_exec,
    requires_approval=True,
    source="task",
)


agent = Agent(
    name="customer-concierge",
    instructions=(
        "You are a customer-service concierge. You can look up orders and, "
        "with explicit human approval, issue refunds. Only request a refund "
        "after the customer's request matches a real order. Summarize what "
        "happened at the end."
    ),
    model="claude-haiku-4-5",
    tools=[lookup_order, _refund_tool],
    max_turns=10,
)


@env.task(report=True)
async def concierge(request: str) -> str:
    """Run the concierge agent. Sensitive actions pause for human approval."""
    result = await agent.run(request)
    return result.summary or result.error
