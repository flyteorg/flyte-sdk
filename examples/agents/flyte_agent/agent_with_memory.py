"""Long-lived agent that persists memory to ``flyte.io.Dir``.

This example shows how to give a :class:`flyte.ai.agents.Agent` continuity
across runs by serializing :class:`~flyte.ai.agents.MemoryStore` to a
``flyte.io.Dir`` between invocations.

Use case: a recurring assistant that remembers context across wakeups (e.g. an
"inbox triage" agent that recalls which threads it has already responded to,
or a research agent that builds up its scratchpad over many days).

Each task input takes the previous memory directory (or ``None`` on the first
call) and returns the updated memory + the agent's reply. The directory holds
``messages.json`` (the live transcript), an opt-in ``audit/log.jsonl`` audit
trail, and any path-addressed artifacts the agent / its tools have written.
"""

from __future__ import annotations

import flyte
from flyte.ai.agents import Agent, MemoryStore
from flyte.io import Dir

env = flyte.TaskEnvironment(
    name="persistent-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def add_note(note: str) -> str:
    """Save a free-form note to the agent's scratchpad.

    Returns a short acknowledgement string.
    """
    return f"Noted: {note}"


@env.task
async def list_history(count: int = 5) -> str:
    """Return a small textual summary of recent activity (stub)."""
    return f"Returned {count} recent items (stub data)."


agent = Agent(
    name="memory-assistant",
    instructions=(
        "You are a continuity-aware assistant. You can record notes and look "
        "up recent history. Use the tools when the user asks you to remember "
        "or recall information."
    ),
    model="claude-haiku-4-5",
    tools=[add_note, list_history],
    max_turns=12,
)


@env.task(report=True)
async def chat_once(message: str, prior_memory: Dir | None = None) -> tuple[str, Dir]:
    """One conversation turn that picks up where the last run left off.

    Args:
        message: The user message for this wakeup.
        prior_memory: Optional ``Dir`` previously produced by this task —
            pass ``None`` on the very first invocation.

    Returns:
        A tuple ``(reply, updated_memory_dir)``. Capture the directory and
        pass it back in for the next call to continue the conversation.
    """
    if prior_memory is not None:
        memory = await MemoryStore.load_from_dir(prior_memory)
        flyte.logger.info("Restored %d prior messages from memory.", len(memory.messages))
    else:
        memory = MemoryStore()

    agent.memory = memory
    result = await agent.run(message)

    # Persist the updated memory so it can be passed back next time.
    new_dir = await memory.save_to_dir()
    return (result.summary or result.error, new_dir)


if __name__ == "__main__":
    flyte.init_from_config()
    print("Starting first conversation turn...")
    run = flyte.run(
        chat_once,
        message="Remember that my favorite color is teal and my dog is named Mochi.",
    )
    print(f"First run: {run.url}")
    run.wait()

    reply, mem_dir = run.outputs()
    print(f"First reply: {reply}")

    print("\nSecond conversation turn — should remember the previous facts...")
    run2 = flyte.run(
        chat_once,
        message="What is my dog's name and favorite color?",
        prior_memory=mem_dir,
    )
    print(f"Second run: {run2.url}")
    run2.wait()
    print(f"Second reply: {run2.outputs()[0]}")
