"""flyteplugins-agents — run agents from existing agent SDKs on Flyte.

This plugin makes Flyte the **durable orchestration runtime** underneath agent
frameworks you already use. You keep writing agents in the framework of your
choice; Flyte provides the execution substrate: durability and replay, automatic
retries / self-healing, per-tool containerized execution (CPU/GPU, caching), and
observability.

Each framework is a separate, lazily-imported adapter behind its own optional
extra, so installing one never pulls the others:

```bash
pip install flyteplugins-agents[openai]
```

```python
from flyteplugins.agents.openai import function_tool, run_agent
```

Adapters:

- ``flyteplugins.agents.openai`` — the OpenAI Agents SDK (``openai-agents``).

The division of labor that makes this work: the **agent run** is a Flyte
``@env.task`` (the durable parent), each **model turn** is a ``flyte.trace``
(a memoized, replayable leaf), and each **tool** is a Flyte task invoked as a
durable child action.
"""

__all__: list[str] = []
