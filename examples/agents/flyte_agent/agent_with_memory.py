"""Long-lived agent that persists memory to a keyed blob-store namespace.

This example shows how to give a :class:`flyte.ai.agents.Agent` continuity
across runs by loading a deterministic :class:`~flyte.ai.agents.MemoryStore`
with ``MemoryStore.get_or_create(key=...)``.

Use case: a recurring assistant that remembers context across wakeups (e.g. an
"inbox triage" agent that recalls which threads it has already responded to,
or a research agent that builds up its scratchpad over many days).

The store lives under the stable raw-data root in the Flyte-managed
``agents/memory-store/v0`` namespace. It holds ``messages.json`` (the live
transcript), an opt-in ``audit/log.jsonl`` audit trail, and any path-addressed
artifacts the agent / its tools have written.
"""

from __future__ import annotations

import flyte
from flyte.ai.agents import Agent, ConcurrencyError, MemoryStore

MEMORY_KEY = "my-assistant"
NOTES_PATH = "notes/notes.json"

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
    memory = await MemoryStore.get_or_create.aio(key=MEMORY_KEY)
    notes = await memory.read_json.aio(NOTES_PATH, default=[])
    sha = await memory.current_sha.aio(NOTES_PATH)
    notes.append(note)
    try:
        await memory.write_json.aio(NOTES_PATH, notes, expected_sha=sha, reason="agent note")
    except ConcurrencyError:
        # Another tool/task updated the notes between our read and write.
        # Surface a retryable result to the agent rather than silently dropping
        # memory.
        return "Memory changed while saving the note; please retry add_note."
    await memory.save.aio()
    return f"Noted: {note}"


@env.task
async def list_history(count: int = 5) -> str:
    """Return recent persisted notes and conversation messages."""
    memory = await MemoryStore.get_or_create.aio(key=MEMORY_KEY)
    notes = await memory.read_json.aio(NOTES_PATH, default=[])
    recent_notes = notes[-count:]
    recent_messages = memory.messages[-count:]

    parts: list[str] = []
    if recent_notes:
        parts.append("Persisted notes:\n" + "\n".join(f"- {note}" for note in recent_notes))
    if recent_messages:
        msg_lines = []
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            if content:
                msg_lines.append(f"- {role}: {content}")
        if msg_lines:
            parts.append("Recent transcript:\n" + "\n".join(msg_lines))
    return "\n\n".join(parts) if parts else "No persisted memory found yet."


agent = Agent(
    name="memory-assistant",
    instructions=(
        "You are a continuity-aware assistant. You can record notes and look "
        "up recent history. When the user asks you to remember something, call "
        "add_note. When the user asks you to recall something, call list_history "
        "and answer from the returned persisted notes."
    ),
    model="claude-haiku-4-5",
    tools=[add_note, list_history],
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
        The agent's reply. Memory is saved back to the keyed store before the
        task returns.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    flyte.logger.info("Restored %d prior messages from memory.", len(memory.messages))

    agent.memory = memory
    result = await agent.run.aio(message)

    # Persist the updated transcript (and any tool-written artifacts) back to the
    # deterministic key path so the next run picks up where this one left off.
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
