"""Run :class:`Agent` behind the built-in chat UI.

Because :class:`flyte.ai.agents.Agent` implements the
:class:`flyte.ai.agents.protocol.AgentProtocol`, it plugs straight into
:class:`flyte.ai.chat.AgentChatAppEnvironment` — you get a hosted chat shell,
tool sidebar, and NDJSON streaming for free.

Run locally::

    export ANTHROPIC_API_KEY=sk-...
    uv run python examples/agents/flyte_agent/agent_chat_ui.py
"""

from __future__ import annotations

import pathlib
from typing import Any

import flyte
from flyte.ai.agents import Agent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

# ---------------------------------------------------------------------------
# Durable Flyte tasks → tools
# ---------------------------------------------------------------------------

task_env = flyte.TaskEnvironment(
    name="chat-agent-tools",
    image=(
        flyte.Image.from_debian_base()
        .with_apt_packages("git")
        .with_pip_packages("litellm", "httpx")
        .with_commands(["uv pip install git+https://www.github.com/flyteorg/flyte-sdk.git@87205fda"])
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
)


@task_env.task
async def search_docs(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search internal documentation (stub) and return matching snippets."""
    corpus = [
        {
            "title": "Tasks",
            "body": (
                "Define a task by decorating an async function with @env.task on a "
                "flyte.TaskEnvironment. Tasks run on the cluster with the env's image "
                "and resources, and you launch them with flyte.run(my_task, ...)."
            ),
        },
        {
            "title": "Environments",
            "body": (
                "flyte.TaskEnvironment groups related tasks under a shared image, resources, "
                "secrets, and reuse policy. Use depends_on=[...] to chain environments and "
                "flyte.deploy(env) to ship them to the cluster."
            ),
        },
        {
            "title": "Triggers",
            "body": (
                "Schedule a task by attaching a flyte.Trigger with a flyte.Cron automation, "
                'e.g. flyte.Trigger(name="hourly", automation=flyte.Cron("0 * * * *")) passed '
                "to env.task(triggers=[...])."
            ),
        },
        {
            "title": "Auth",
            "body": (
                "Authenticate the Flyte CLI against your cluster with "
                "`flyte create config --endpoint <your-endpoint>`. The command writes a config "
                "file and starts the OAuth device flow so subsequent `flyte` commands are "
                "authorized."
            ),
        },
        {
            "title": "Secrets",
            "body": (
                "Mount cluster-managed secrets into a task by passing "
                "flyte.Secret(key='my-secret', as_env_var='MY_SECRET') to "
                "TaskEnvironment(secrets=[...]); the value is available as os.environ['MY_SECRET'] "
                "at runtime."
            ),
        },
        {
            "title": "Caching",
            "body": (
                "Flyte memoizes task outputs by hashing the task's code and inputs. Override "
                "the cache scope with env.task(cache='auto'|'disable'|'override') or "
                "flyte.with_runcontext(overwrite_cache=True).run(...) to force a fresh run."
            ),
        },
    ]
    needle = query.lower()
    matches = [doc for doc in corpus if needle in doc["body"].lower() or needle in doc["title"].lower()]
    return matches[:max_results]


@task_env.task
async def summarize(text: str, max_words: int = 40) -> str:
    """Stub summarizer — replace with your model of choice."""
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = Agent(
    name="docs-helper",
    instructions=(
        "You are a friendly internal docs assistant. Use search_docs to find "
        "relevant snippets, then call summarize when the user wants a TL;DR. "
        "Always cite the doc title in your final answer."
    ),
    model="claude-haiku-4-5",
    tools=[search_docs, summarize],
    max_turns=8,
)


# ---------------------------------------------------------------------------
# Optional: route LLM/tool execution through a parent task so durable tools
# work correctly with the chat UI. See ``codemode_durable_agent_ui.py`` for
# the same pattern applied to :class:`CodeModeAgent`.
# ---------------------------------------------------------------------------


@task_env.task(report=True)
async def chat_entrypoint(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Parent task that owns the agent loop and the nested tool tasks."""
    result = await agent.run.aio(message, history=history)
    return {
        "summary": result.summary,
        "error": result.error,
        "attempts": result.attempts,
        "charts": [],
        "code": "",
    }


# ---------------------------------------------------------------------------
# Chat app environment
# ---------------------------------------------------------------------------

env = AgentChatAppEnvironment(
    name="flyte-native-agent-chat-ui",
    agent=agent,
    task_entrypoint=chat_entrypoint,
    title="Internal docs assistant",
    subtitle="Backed by a flyte.ai.agents.Agent + durable Flyte task tools.",
    theme=CustomTheme(accent_color="#6F2AEF", accent_hover_color="#8B52F2"),
    passthrough_auth=True,
    prompt_nudges=[
        {"label": "Basics", "prompt": "Can you show me a hello world example?"},
        {"label": "Triggers", "prompt": "How do I schedule a task?"},
        {"label": "Apps", "prompt": "How do I serve a model using a FastAPI app?"},
    ],
    depends_on=[task_env],
    image=(flyte.Image.from_debian_base().with_pip_packages("litellm", "fastapi", "uvicorn")),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env)
    print(f"Agent chat UI deployed: {deployments[0].summary_repr()}")
    print(f"Url: {deployments[0].envs['flyte-native-agent-chat-ui'].deployed_app.url}")
