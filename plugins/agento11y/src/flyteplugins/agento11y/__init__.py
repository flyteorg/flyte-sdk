"""Grafana Agent Observability for Flyte agents.

One call wires an agent built with the Flyte agents plugin into Grafana's Agent
Observability: generations, tool calls, and cost land in Grafana, nested inside the Flyte
task span, and grouped by Flyte run.

    import flyte
    from flyteplugins.agento11y import init
    from flyteplugins.agents.openai import run_agent

    # Module scope, not inside a task.
    init(service_name="my-agent")

    env = flyte.TaskEnvironment(name="agent_env")

    @env.task
    async def agent(question: str) -> str:
        return await run_agent(question, tools=[...], model="gpt-4.1")

Nothing else changes in the agent code. The adapter offers its framework's run payload to an
instrumentor on the way past, which is how a handler reaches a call the adapter owns rather
than you.

Install the extra for the framework you use, which is what makes its instrumentor available:

    pip install "flyteplugins-agento11y[openai]"

Available extras: langchain, langgraph, openai, claude, google, pydantic-ai. crewai and
mistral have Flyte adapters but no agento11y integration yet, so their runs are still traced
but their generations are not captured.
"""

from ._binding import FlyteIdentityBinding
from ._frameworks import SUPPORTED_FRAMEWORKS
from ._setup import get_client, init, instrumented_frameworks, shutdown
from .links import GrafanaAgentObservability

__all__ = [
    "SUPPORTED_FRAMEWORKS",
    "FlyteIdentityBinding",
    "GrafanaAgentObservability",
    "get_client",
    "init",
    "instrumented_frameworks",
    "shutdown",
]
