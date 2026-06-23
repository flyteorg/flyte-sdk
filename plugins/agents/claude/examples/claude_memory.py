"""Cross-run agent memory on Flyte — a Claude agent remembers across separate runs.

``run_agent(..., memory_key=...)`` persists the conversation transcript to a
durable, keyed ``MemoryStore`` and resumes it on the next run with the same key —
so the agent remembers across workers and restarts. (The Claude CLI's own session
lives in an ephemeral per-pod dir; Flyte makes it durable and addressable by
thread, and resumes it via the SDK's session-mirror.)

Two separate runs share one ``memory_key``: it learns a fact in run 1 and recalls
it in run 2.

Memory is keyed under the active org/project/domain, so run with a configured
context (``flyte.init_from_config()`` / a backend). ``memory_key`` is a single
segment (a user/thread id).

Run:  python claude_memory.py
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.claude import run_agent

# The Claude Agent SDK bundles the native `claude` CLI in its wheel, so the image
# only needs the adapter — installed here from locally-built wheels under `../dist`.
env = flyte.TaskEnvironment(
    "claude-memory",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="claude-memory").clone(
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


@env.task(report=True, retries=3)
async def chat(message: str, memory_key: str) -> str:
    """One turn of a memory-backed conversation, keyed by ``memory_key``.

    Because ``memory_key`` is stable across runs, the agent resumes the prior
    conversation every time it is called with the same key.
    """
    return await run_agent(
        message,
        instructions="You are a friendly assistant. Use the conversation history to stay consistent.",
        model="claude-sonnet-4-5",
        memory_key=memory_key,
    )


if __name__ == "__main__":
    flyte.init_from_config()

    # Run 1: the agent learns a fact.
    r1 = flyte.run(chat, message="Hi! My name is Alice and I love hiking.", memory_key="user-alice")
    r1.wait()
    print(f"run 1: {r1.outputs()}")

    # Run 2 — a SEPARATE run with the same memory_key: the agent recalls it.
    r2 = flyte.run(chat, message="What's my name and what do I like?", memory_key="user-alice")
    print(f"View at: {r2.url}")
    r2.wait()
    print(f"run 2 (recall): {r2.outputs()}")
