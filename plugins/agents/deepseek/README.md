# flyteplugins-agents-deepseek

Run [DeepSeek Harness](https://deepseek-harness.github.io/deepseek-harness/en/guide/python-sdk)
agents on Flyte. You keep writing DeepSeek Harness code; Flyte is the runtime
underneath.

```bash
pip install flyteplugins-agents-deepseek
```

```python
import flyte
from flyteplugins.agents.deepseek import tool, run_agent

env = flyte.TaskEnvironment(
    "deepseek-agent",
    secrets=[flyte.Secret(key="deepseek_api_key", as_env_var="DEEPSEEK_API_KEY")],
)


@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    return await run_agent(question, tools=[get_weather], model="deepseek-v4-flash")
```

## How it maps to Flyte

- The loop runs inside the harness runtime subprocess (JSON-RPC over stdio).
  `run_agent` drives it from inside your `@env.task` — the blocking
  `DeepSeekHarness.run` is bridged off the event loop with `asyncio.to_thread`, so
  the task stays responsive while the agent works.
- Tools are Flyte tasks. When the agent calls one it runs as a durable Flyte child
  action (its own container/resources, retries, caching). The input schema is
  derived via the Flyte type engine.
- `workspace=` points the harness's own tools (bash, string editor) at a real
  directory — for example a `flyte.io.Dir` you downloaded — so the agent can read
  and edit actual files and you can hand the result downstream as an artifact.

### How tools work here

This adapter differs from the client-side ones, and it's worth knowing why.

DeepSeek Harness has **no tool-registration channel**: its wire protocol is
`initialize` / `session/prompt`, and its tool surface is whatever its Cordis
plugin composition provides *inside the runtime subprocess*. There is nowhere to
hand a Python function.

What every composition does provide is local bash, scoped to a working directory
the adapter chooses. So the bridge meets it there:

1. each tool is published into `<workspace>/.flyte_tools/<name>` as a small
   executable shim (stdlib-only Python, run under this process's own interpreter,
   so the harness runtime needs nothing installed);
2. `run_agent` listens on a Unix domain socket in a private temp directory;
3. the model runs `.flyte_tools/get_weather '{"city": "Paris"}'`, the shim forwards
   the JSON arguments over the socket, and the adapter awaits `task.aio(...)` — a
   durable Flyte child action — before writing the result back for the shim to print.

Because the harness has no tool-declaration message either, the tool manual
(names, parameter types, an example invocation per tool) is prepended to the
prompt. A failing tool comes back as a non-zero exit with the reason on stderr, so
the agent sees the error and can react rather than the run dying.

The shims are the only thing the adapter writes into your workspace, and they are
removed when the run ends.

If you own the Cordis composition, the runtime does have a first-class in-process
tool API (`harness.registerTool(ctx, tool)`) — but it is a TypeScript plugin API
inside the runtime, not something the Python SDK can reach, so it is not an option
for an adapter that has to work with the stock composition.

## Durability

Two layers, both real:

- Tool calls are durable Flyte child actions (own container/resources, retries,
  caching) — always, regardless of `durable`.
- The conversation survives a crash. With `durable=True`, `run_agent` mirrors the
  harness's JSONL session store (`session_root` / `DSH_SESSION_ROOT`) onto a
  `flyte.Checkpoint`. The session id is derived from the task's action, so it is
  stable across retries; on a retry the prior attempt's transcript is restored into
  the session root and the same session is prompted, so the harness continues the
  conversation instead of starting over.

We delegate to the harness's own session persistence because the model loop runs
in its runtime subprocess (which Flyte doesn't intercept), so a model turn can't be
a `flyte.trace` leaf the way it is for other client-side SDKs. Session resume is the
coarser-grained equivalent — whole-session, not per-turn — and it no-ops cleanly
when there's no checkpoint context (e.g. local runs).

## Observability

`run_agent` renders a timeline into the task report (`report=True`): assistant
turns and turn endings from the harness's streamed session events, the harness's
own tool activity, and each Flyte-task tool's arguments and result (or error),
recorded by the bridge as it dispatches them.

Session notifications arrive on the worker thread running the blocking harness
call; each is marshalled back onto the event loop before touching the report, so
the timeline stays correctly ordered against the tool rows.

## Memory

Pass `memory_key` (a user/thread id) for cross-run memory — the agent resumes the
same conversation across separate runs:

```python
await run_agent(message, model="deepseek-v4-flash", memory_key="user-alice")
```

The session archive is persisted to a durable, keyed `MemoryStore` and restored on
the next run with the same key, which then prompts the same harness session id.
That also covers crash-resume, so it supersedes the per-run `durable` checkpoint.

## Bring your own configuration

Pass a fully-built `DeepSeekHarnessConfig` as `config=` to keep SDK-native setup —
a custom `cordis` composition, `base_url` / `api_key`, timeouts. The adapter layers
only `cwd` (the workspace, where shims are published) and `session_root` (what
resume and memory mirror) on top; `model` / `provider` / `max_tokens` and any extra
keyword arguments override fields on it.

## Runtime

The `deepseek-harness-sdk` wheel pins the matching `deepseek-harness-runtime-bin`
platform wheel, which bundles the single-file `dsh-jsonrpc-agent` executable — so
`pip install flyteplugins-agents-deepseek` is all the runtime image needs (no
separate Node.js install). Wheels are published for Linux x86-64/aarch64 and
macOS 14+ arm64.

Set `DEEPSEEK_API_KEY` in the environment (a Flyte secret, as above), and
`DEEPSEEK_BASE_URL` if you route through a proxy.

The SDK is currently published as a pre-release, so installing it may require
`--prerelease=allow` (uv) / `--pre` (pip) depending on your resolver settings.

## Examples

See [`examples/`](examples/):

- [`deepseek_durable_agent.py`](examples/deepseek_durable_agent.py) — a single
  durable agent: tools as Flyte tasks, assistant turns and tool outcomes in the report,
  in both the async and sync call forms.
- [`deepseek_crash_resume.py`](examples/deepseek_crash_resume.py) — crash & resume:
  the task crashes on its first attempt; on retry the conversation resumes from the
  `flyte.Checkpoint`-backed session and completed tool calls are cache hits. Run on a
  backend to see resume.
- [`deepseek_workspace_agent.py`](examples/deepseek_workspace_agent.py) — a real
  workspace: the agent reads, edits and tests an actual `flyte.io.Dir` with the
  harness's bash and editor, verifies its own work through a Flyte-task tool, and the
  patched directory becomes the task's output.
- [`deepseek_multi_agent.py`](examples/deepseek_multi_agent.py) — multi-agent
  orchestration: a planner agent decomposes a topic, researcher agents fan out in
  parallel, an editor agent synthesizes — each agent its own durable action.
- [`deepseek_memory.py`](examples/deepseek_memory.py) — cross-run memory: two
  separate runs share a `memory_key`; the agent learns a fact in run 1 and recalls it
  in run 2.
- [`deepseek_custom_agent.py`](examples/deepseek_custom_agent.py) — bring your own
  `DeepSeekHarnessConfig` (custom Cordis composition, proxied endpoint, timeouts).

## Conformance

This adapter passes the shared `flyteplugins.agents.core.testing.assert_adapter_conforms`
check — the same one every adapter runs — so it follows the common format
(`tool` + `run_agent`, tool tasks wired to the resolver) despite a very different
underlying SDK shape.
