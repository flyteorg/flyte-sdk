"""Multi-agent research pipeline — Flyte orchestrating many Claude agents.

The "Flyte as the orchestrator harness" story for the Claude Agent SDK. Instead of
one agent, a small team is composed with ordinary Flyte control flow, and each
agent run is its own durable, observable, cached Flyte action:

    plan --> research (parallel, one per subtopic) --> synthesize

- ``plan`` runs a *planner* agent that breaks the topic into subtopics.
- ``research`` runs a *researcher* agent per subtopic, **fanned out in parallel**
  (``asyncio.gather``); each is a separate durable child action with its own
  ``search_web`` tool calls and report.
- ``synthesize`` runs an *editor* agent that combines the findings.

Every agent is a first-class Flyte node — independently retried, cached,
resource-sized and observable — and the fan-out is real distributed parallelism,
not just asyncio in one process. (Claude's own subagents still work inside a single
``run_agent``; this shows the orchestration layer on top.)

Run:  python claude_multi_agent.py
"""

import asyncio
from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.claude import function_tool, run_agent

# The Claude Agent SDK bundles the native `claude` CLI in its wheel, so the image
# only needs the adapter — installed here from locally-built wheels under `../dist`.
env = flyte.TaskEnvironment(
    "claude-research",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="claude-research").clone(
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

# A real deployment would call a search API here. Tools are Flyte tasks, so this
# could just as well be a GPU retrieval task, a warehouse query, or a browser
# session — each sized and cached independently.
_CORPUS = {
    "battery": "Solid-state cells reached 500 Wh/kg in 2025 pilot lines; cycle life is the open problem.",
    "charging": "Public DC fast-charger count doubled YoY; grid interconnect queues are the bottleneck.",
    "cost": "Pack prices fell below $80/kWh in 2025, crossing the oft-cited parity threshold.",
}


@function_tool
@env.task(cache="auto", retries=3)
async def search_web(query: str) -> str:
    """Search the web for a query and return the most relevant snippet."""
    for key, snippet in _CORPUS.items():
        if key in query.lower():
            return snippet
    return "No strong match; rely on general knowledge and state assumptions."


@env.task(retries=3)
async def plan(topic: str) -> list[str]:
    """Planner agent: decompose a topic into focused research subtopics."""
    text = await run_agent(
        f"Break the topic '{topic}' into exactly 3 focused, distinct research subtopics.",
        instructions="Reply with ONLY a comma-separated list of 3 short subtopics. No numbering, no prose.",
        model="claude-sonnet-4-5",
    )
    # Forgiving parse: accept commas or newlines, strip bullets/numbering.
    raw = [part.strip(" -•0123456789.").strip() for chunk in text.splitlines() for part in chunk.split(",")]
    subtopics = [s for s in raw if s]
    return subtopics[:3] or [topic]


@env.task(report=True, retries=3)
async def research(subtopic: str) -> str:
    """Researcher agent: investigate one subtopic using the search tool."""
    return await run_agent(
        f"Research this subtopic and summarize the key findings as 3 concise bullet points:\n{subtopic}",
        tools=[search_web],
        instructions="You are a rigorous research assistant. Use search_web before answering. Be concise.",
        model="claude-sonnet-4-5",
    )


@env.task(report=True, retries=3)
async def synthesize(topic: str, findings: list[str]) -> str:
    """Editor agent: synthesize the per-subtopic findings into a briefing."""
    notes = "\n\n".join(f"- {f}" for f in findings)
    return await run_agent(
        f"Topic: {topic}\n\nResearch notes:\n{notes}\n\nWrite a tight one-paragraph executive briefing.",
        instructions="You are a sharp editor. Synthesize the notes faithfully; do not invent facts.",
        model="claude-sonnet-4-5",
    )


@env.task(report=True, retries=3)
async def research_pipeline(topic: str) -> str:
    """Orchestrate the agent team: plan → parallel research → synthesize."""
    subtopics = await plan(topic)

    # Fan out one researcher agent per subtopic. Each is a durable child action;
    # grouping keeps them tidy in the Flyte UI.
    with flyte.group("parallel-research"):
        findings = await asyncio.gather(*(research(subtopic) for subtopic in subtopics))

    return await synthesize(topic, list(findings))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(research_pipeline, topic="The state of electric-vehicle batteries in 2025")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Briefing:\n{run.outputs()}")
