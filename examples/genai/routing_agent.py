# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0",
#    "flyteplugins-anthropic>=2.0.0",
# ]
# ///
"""
Orchestration Agent — Query Router

This example demonstrates an orchestration pattern where a lead agent receives
a user query and routes it to the appropriate specialist sub-agent. Each
sub-agent is a Flyte task, so it runs as its own tracked, retryable unit of
work on the cluster.

The router uses Claude to classify the query intent, then dispatches to one of:
  - summarize_text  — condense a long passage
  - translate_text  — translate text to another language
  - answer_question — answer a factual question

This is the "agent-as-orchestrator" pattern: the LLM decides *which* tool to
call rather than executing the work itself.  Each tool is a full Flyte task,
giving you lineage, caching, retries, and resource isolation for free.
"""

import asyncio

from flyteplugins.anthropic import function_tool, run_agent

import flyte

agent_env = flyte.TaskEnvironment(
    "routing-agent",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_uv_script(__file__, name="routing-agent"),
)


# ---------------------------------------------------------------------------
# Specialist sub-agent tasks
# ---------------------------------------------------------------------------


@agent_env.task
async def summarize_text(text: str) -> str:
    """Summarize the provided text into a concise 2-3 sentence summary."""
    result = await run_agent(
        prompt=f"Summarize the following text in 2-3 sentences:\n\n{text}",
        tools=[],
        system="You are a concise summarizer. Return only the summary, nothing else.",
        model="claude-sonnet-4-20250514",
    )
    return result


@agent_env.task
async def translate_text(text: str, target_language: str) -> str:
    """Translate the provided text into the specified target language."""
    result = await run_agent(
        prompt=f"Translate the following text to {target_language}:\n\n{text}",
        tools=[],
        system=("You are a translator. Return only the translation, no explanations or commentary."),
        model="claude-sonnet-4-20250514",
    )
    return result


@agent_env.task
async def answer_question(question: str) -> str:
    """Answer a factual question with a clear, well-sourced response."""
    result = await run_agent(
        prompt=question,
        tools=[],
        system=(
            "You are a knowledgeable assistant. Answer the question accurately "
            "and concisely. If you are unsure, say so."
        ),
        model="claude-sonnet-4-20250514",
    )
    return result


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------


@agent_env.task
async def router_agent(queries: list[str]) -> list[str]:
    """
    Route each query to the right specialist sub-agent.

    The router is itself an LLM call — Claude looks at each query, picks the
    right tool (summarize, translate, or answer), and calls it.  Each tool
    invocation is a separate Flyte task execution.
    """
    tools = [
        function_tool(summarize_text),
        function_tool(translate_text),
        function_tool(answer_question),
    ]

    async def handle(query: str, idx: int) -> str:
        with flyte.group(f"query-{idx}"):
            result = await run_agent(
                prompt=query,
                tools=tools,
                system=(
                    "You are a routing agent. For each user request, decide which "
                    "tool to call:\n"
                    "- summarize_text: when the user wants a summary of some text\n"
                    "- translate_text: when the user wants text translated\n"
                    "- answer_question: when the user asks a factual question\n\n"
                    "Pick the single best tool and call it. Do not try to answer "
                    "the question yourself — always delegate to a tool."
                ),
                model="claude-sonnet-4-20250514",
            )
            return result

    tasks = [handle(q, i) for i, q in enumerate(queries)]
    return list(await asyncio.gather(*tasks))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        router_agent,
        queries=[
            # This should route to summarize_text
            (
                "Please summarize: The transformer architecture was introduced in "
                "'Attention Is All You Need' (2017). It replaced recurrent layers "
                "with self-attention mechanisms, enabling much greater parallelism "
                "during training. This led to models like BERT, GPT, and their "
                "successors, fundamentally changing NLP and eventually computer "
                "vision and other domains."
            ),
            # This should route to translate_text
            "Translate to French: The weather is beautiful today.",
            # This should route to answer_question
            "What is the capital of New Zealand?",
        ],
    )
    print(f"View at: {run.url}")
    run.wait()
    print(f"Results: {run.outputs()}")
