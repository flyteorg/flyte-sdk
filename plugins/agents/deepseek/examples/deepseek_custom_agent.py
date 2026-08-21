"""Bring your own harness configuration — SDK-native config, Flyte underneath.

The adapter is a thin layer, not a wrapper that hides the SDK. When you need the
harness configured your way — a custom Cordis plugin composition, a proxied
endpoint, tighter timeouts — build a ``DeepSeekHarnessConfig`` exactly as you
would outside Flyte and pass it as ``config=``.

The adapter layers only what it must on top:

- ``cwd`` — the workspace, because that is where the tool shims are published;
- ``session_root`` — because that is what durable resume and cross-run memory
  mirror.

Everything else on the config is yours, and ``model`` / ``provider`` /
``max_tokens`` / ``**kwargs`` on ``run_agent`` override fields on top of it.

Run:  flyte run deepseek_custom_agent.py configured_agent --question "What's the population of Tokyo?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte
from deepseek_harness import DeepSeekHarnessConfig

from flyteplugins.agents.deepseek import run_agent, tool

env = flyte.TaskEnvironment(
    "deepseek-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="deepseek_api_key", as_env_var="DEEPSEEK_API_KEY")],
    image=flyte.Image.from_debian_base(name="deepseek-custom-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-deepseek"]
    ),
)


@tool
@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


@env.task(report=True, retries=3)
async def configured_agent(question: str) -> str:
    """Drive the agent from a hand-built SDK config."""
    # Native SDK configuration — the same object you would use standalone. Point
    # ``cordis`` at your own plugin composition to change the harness's own tools,
    # persistence or provider routes (keep the JSON-RPC server entry in it).
    config = DeepSeekHarnessConfig(
        provider="deepseek-official",
        model="deepseek-v4-flash",
        max_tokens=49_152,
        request_timeout_seconds=300.0,
        # cordis="/opt/harness/my.cordis.yml",
        # base_url="https://my-proxy.internal/v1",
    )

    return await run_agent(
        question,
        tools=[get_population],
        instructions="You are a concise city-facts assistant. Use the provided tools to answer.",
        config=config,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(configured_agent, question="What's the population of Tokyo?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
