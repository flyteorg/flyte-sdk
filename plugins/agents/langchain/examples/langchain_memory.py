"""Cross-run agent memory on Flyte — the agent remembers across separate runs.

``run_agent(..., memory_key=...)`` backs the LangChain session with a
durable, keyed ``MemoryStore`` (object storage), so two runs with the same
``memory_key`` continue the same conversation — across workers and restarts.

This drives the agent in two separate runs sharing one ``memory_key``: it
learns a fact in run 1 and recalls it in run 2.

Memory is keyed under the active org/project/domain, so run with a configured
context (``flyte.init_from_config()`` / a backend). ``memory_key`` is a single
segment (a user/thread id).

Run:  flyte run langchain_memory.py chat --message "Hi! My name is Alice and I love hiking." --memory_key user-alice
      flyte run langchain_memory.py chat --message "What's my name and what do I like?" --memory_key user-alice
      (add `--local` after `run` to run locally; the shared `--memory_key` ties the two runs)
"""

import flyte

from flyteplugins.agents.langchain import run_agent

env = flyte.TaskEnvironment(
    "langchain-memory",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="langchain-memory")
    .with_local_v2_plugins(["flyteplugins-agents-core", "flyteplugins-agents-langchain"])
    .with_pip_packages("langchain-openai"),
)


@env.task(report=True, retries=3)
async def chat(message: str, memory_key: str) -> str:
    """One turn of a memory-backed conversation, keyed by ``memory_key``.

    Because ``memory_key`` is stable across runs, the agent sees the prior turns
    every time it is called with the same key.
    """
    from langchain_openai import ChatOpenAI

    return await run_agent.aio(
        message,
        model=ChatOpenAI(model="gpt-4o"),
        instructions="You are a friendly assistant. Use the conversation history to stay consistent.",
        memory_key=memory_key,
    )


if __name__ == "__main__":
    flyte.init_from_config()

    # Run 1: the agent learns a fact.
    r1 = flyte.run(chat, message="Hi! My name is Alice and I love hiking.", memory_key="user-alice")
    r1.wait()
    print(f"run 1: {r1.outputs()}")

    # Run 2 — a separate run with the same memory_key: the agent recalls it.
    r2 = flyte.run(chat, message="What's my name and what do I like?", memory_key="user-alice")
    print(f"View at: {r2.url}")
    r2.wait()
    print(f"run 2 (recall): {r2.outputs()}")
