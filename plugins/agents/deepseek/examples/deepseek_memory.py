"""Cross-run agent memory on Flyte — a DeepSeek agent remembers across separate runs.

``run_agent(..., memory_key=...)`` persists the harness's session transcript to a
durable, keyed ``MemoryStore`` and restores it on the next run with the same key —
so the agent remembers across workers and restarts. (The harness writes its
sessions as JSONL under an ephemeral per-pod ``session_root``; Flyte makes that
durable and addressable by thread, then prompts the same session id so the
harness simply continues the conversation.)

Two separate runs share one ``memory_key``: it learns a fact in run 1 and recalls
it in run 2.

Memory is keyed under the active org/project/domain, so run with a configured
context (``flyte.init_from_config()`` / a backend). ``memory_key`` is a single
segment (a user/thread id).

Run:  flyte run deepseek_memory.py chat --message "Hi! My name is Alice and I love hiking." --memory_key user-alice
      flyte run deepseek_memory.py chat --message "What's my name and what do I like?" --memory_key user-alice
      (add `--local` after `run` to run locally; the shared `--memory_key` ties the two runs)
"""

import flyte

from flyteplugins.agents.deepseek import run_agent

# The `deepseek-harness-sdk` wheel pulls in the bundled harness runtime, so the
# image only needs the adapter — installed here from locally-built wheels.
env = flyte.TaskEnvironment(
    "deepseek-memory",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="deepseek_api_key", as_env_var="DEEPSEEK_API_KEY")],
    image=flyte.Image.from_debian_base(name="deepseek-memory").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-deepseek"]
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
        model="deepseek-v4-flash",
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
