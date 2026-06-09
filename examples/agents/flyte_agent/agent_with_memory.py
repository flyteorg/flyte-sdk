"""Long-lived agent that persists memory to a keyed blob-store namespace.

This example shows how to give a :class:`flyte.ai.agents.Agent` continuity
across runs *functionally*: memory is not attached to the agent, it is passed
into :meth:`Agent.run` per call and returned on the result.

Use case: a recurring assistant that remembers context across wakeups (e.g. an
"inbox triage" agent that recalls which threads it has already responded to,
or a research agent that builds up its scratchpad over many days).

How memory flows
----------------

1. The caller loads (or creates) a deterministic, keyed store with
   ``MemoryStore.get_or_create(key=...)``. It lives under the stable raw-data
   root in the Flyte-managed ``agents/memory-store/v0`` namespace.
2. The store is handed to ``agent.run(message, memory=store)``. The agent
   prepends the prior transcript, runs the tool-use loop, and appends the new
   transcript back to the store. No ``agent.memory = ...`` assignment.
3. The caller saves the updated store with ``memory.save()`` (saving is
   explicit, not magic). The same (updated) store is also returned on
   ``result.memory``.

Note that tools never touch the ``MemoryStore``: they are plain functions and
their effects are captured in the conversation transcript, which is what gives
the agent continuity. The store's path-addressed artifacts / audit / version
features remain available to the *caller* (not tools) via
``memory.read_json`` / ``memory.write_json`` when richer state is needed.
"""

from __future__ import annotations

import flyte
from flyte.ai.agents import Agent, MemoryStore

MEMORY_KEY = "my-assistant"

env = flyte.TaskEnvironment(
    name="persistent-agent",
    image=flyte.Image.from_debian_base().with_pip_packages("litellm"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@env.task
async def current_time() -> str:
    """Return the current UTC time as an ISO-8601 string.

    A plain, stateless tool: it knows nothing about the agent's memory. Whatever
    it returns is recorded in the transcript, so the agent can recall it later.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


agent = Agent(
    name="memory-assistant",
    instructions=(
        "You are a continuity-aware assistant. You remember facts the user "
        "shares with you across conversations because your prior transcript is "
        "always available. When the user asks what time it is, call current_time."
    ),
    model="claude-haiku-4-5",
    tools=[current_time],
    max_turns=12,
)


@env.task(report=True)
async def chat(message: str, memory_key: str = MEMORY_KEY) -> str:
    """One conversation turn that picks up where the last run left off.

    Args:
        message: The user message for this wakeup.
        memory_key: Deterministic memory namespace. Reuse the same key to keep
            continuity across runs.

    Returns:
        The agent's reply. The updated memory is saved explicitly via
        ``memory.save`` before this task returns.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    flyte.logger.info("Restored %d prior messages from memory.", len(memory.messages))

    # The agent prepends the prior transcript and appends this turn back onto the
    # store, returning it on ``result.memory``. Saving is explicit: persist the
    # updated transcript to the deterministic key path before returning.
    result = await agent.run.aio(message, memory=memory)
    await memory.save.aio()
    return result.summary or result.error


if __name__ == "__main__":
    flyte.init_from_config()
    print("Starting first conversation turn...")
    run = flyte.run(
        chat,
        message="Remember that my favorite color is teal and my dog is named Mochi.",
        memory_key=MEMORY_KEY,
    )
    print(f"First run: {run.url}")
    run.wait()

    reply = run.outputs()[0]
    print(f"First reply: {reply}")

    print("\nSecond conversation turn — should remember the previous facts...")
    run2 = flyte.run(
        chat,
        message="What is my dog's name and favorite color?",
        memory_key=MEMORY_KEY,
    )
    print(f"Second run: {run2.url}")
    run2.wait()
    print(f"Second reply: {run2.outputs()[0]}")
