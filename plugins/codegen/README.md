# Code Generation and Evaluation Plugin

Generate code from natural language prompts and validate it by running tests in an isolated sandbox. Works with any model that supports structured outputs (GPT-4, Claude, Gemini, etc. via LiteLLM) or directly with the Agent SDK (Claude-only).

> **Note:** Only Python is supported today.

## Installation

```bash
pip install flyteplugins-codegen

# For Agent SDK mode (Claude-only)
pip install flyteplugins-codegen[agent-sdk]
```

## Quick start

```python
import flyte
from flyte.io import File
from flyte.sandbox import sandbox_environment
from flyteplugins.codegen import AutoCoderAgent

agent = AutoCoderAgent(name="summarize-sales", resources=flyte.Resources(cpu=1, memory="1Gi"))

env = flyte.TaskEnvironment(
    name="my-env",
    secrets=[flyte.Secret(key="openai_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base().with_pip_packages(
        "flyteplugins-codegen",
    ),
    depends_on=[sandbox_environment],  # Required
)

@env.task
async def process_data(csv_file: File) -> tuple[float, int, int]:
    result = await agent.generate.aio(
        prompt="Read the CSV and compute total_revenue, total_units and row_count.",
        samples={"sales": csv_file},
        outputs={"total_revenue": float, "total_units": int, "row_count": int},
    )
    return await result.run.aio()
```

## Two approaches

### 1. LiteLLM (default)

Uses structured-output LLM calls to generate code, detect packages, build sandbox images, run tests, diagnose failures and iterate. Works with any model that supports structured outputs (GPT-4, Claude, Gemini, etc. via LiteLLM).

```python
agent = AutoCoderAgent(
    name="my-task",
    model="gpt-4.1",           # Any LiteLLM-compatible model
    max_retries=10,            # LLM-driven retry loop
)

result = await agent.generate.aio(
    prompt="...",
    samples={"input": my_file},
    outputs={"result": str},
)
```

**How it works:**

```
prompt + samples
    |
    v
[generate_plan] --> CodePlan
    |
    v
[generate_code] --> CodeSolution (dependencies + code)
    |
    v
[detect_packages] --> pip/system packages
    |
    v
[build_image] --> Sandbox image with deps
    |
    +-- skip_tests=True? --> return result (no tests)
    |
    v
[generate_tests] --> pytest suite
    |
    v
[execute_tests] --> pass? return result
    |                  |
    |                  fail
    v                  |
[diagnose_error] --> logic/environment/test_error
    |
    +-- logic error ---------> regenerate code with patch instructions
    +-- environment error ---> add packages, rebuild image
    +-- test error ----------> fix test expectations
    |
    v
  (repeat up to max_retries)
```

### 2. Agent SDK

Uses the Claude Agent SDK to autonomously generate, test and fix code. The agent has access to `Bash`, `Read`, `Write` and `Edit` tools and iterates on its own. Test execution is intercepted and run in an isolated `Sandbox`.

```python
agent = AutoCoderAgent(
    name="my-task",
    model="claude-sonnet-4-5-20250929",
    use_agent_sdk=True,         # Requires ANTHROPIC_API_KEY as a Flyte secret
)

result = await agent.generate.aio(
    prompt="...",
    samples={"input": my_file},
    outputs={"result": str},
)
```

**Key differences from LiteLLM:**

- Agent runs autonomously (no structured retry loop)
- Requires `ANTHROPIC_API_KEY` as a Flyte secret
- Claude-only (not model agnostic)
- Traces agent tool calls, reasoning and test results in the Flyte UI
- Test commands are intercepted via hooks and run in isolated sandbox environments

## API reference

### `AutoCoderAgent`

Create an agent instance with configuration, then call `generate()` per task.

```python
agent = AutoCoderAgent(name="my-agent", model="gpt-4.1")

# Sync
result = agent.generate(prompt="...")

# Async
result = await agent.generate.aio(prompt="...")
```

**Constructor parameters (agent-level config):**

| Parameter             | Type              | Default        | Description                                                   |
| --------------------- | ----------------- | -------------- | ------------------------------------------------------------- |
| `name`                | `str`             | `"auto-coder"` | Unique name for tracking and image naming                     |
| `model`               | `str`             | `"gpt-4.1"`    | LiteLLM model identifier                                      |
| `system_prompt`       | `str`             | `None`         | Custom system prompt override                                 |
| `api_key`             | `str`             | `None`         | Env var name for LLM API key                                  |
| `api_base`            | `str`             | `None`         | Custom API base URL                                           |
| `litellm_params`      | `dict`            | `None`         | Extra LiteLLM params (temperature, max_tokens, etc.)          |
| `base_packages`       | `list[str]`       | `None`         | Always-install pip packages                                   |
| `resources`           | `flyte.Resources` | `None`         | Resources for sandbox execution (default: cpu=1, 1Gi)         |
| `image_config`        | `ImageConfig`     | `None`         | Registry, registry_secret, python_version                     |
| `max_retries`         | `int`             | `10`           | Max retry iterations (LiteLLM mode)                           |
| `max_sample_rows`     | `int`             | `100`          | Rows to sample from data for context                          |
| `skip_tests`          | `bool`            | `False`        | Skip test generation and execution (LiteLLM mode only)        |
| `network_access`      | `bool`            | `False`        | Allow generated code to access the network inside the sandbox |
| `retries`             | `int`             | `0`            | Number of retries for sandboxes                           |
| `timeout`             | `int`             | `None`         | Timeout in seconds for sandboxes                          |
| `env_vars`            | `dict[str, str]`  | `None`         | Environment variables to pass to sandboxes                |
| `secrets`             | `list`            | `None`         | `flyte.Secret` objects to make available to sandboxes      |
| `cache`               | `str`             | `"auto"`       | CacheRequest for sandboxes: `"auto"`, `"override"`, or `"disable"` |
| `use_agent_sdk`       | `bool`            | `False`        | Use Claude Agent SDK instead of LiteLLM                       |
| `agent_sdk_max_turns` | `int`             | `50`           | Max turns for Claude Agent SDK                                |

**`generate()` parameters (per-call):**

| Parameter     | Type                              | Default  | Description                                                                                                    |
| ------------- | --------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `prompt`      | `str`                             | required | Natural-language task description                                                                              |
| `schema`      | `str`                             | `None`   | Free-form context about data formats, structures, or schemas. Included verbatim in the LLM prompt.             |
| `constraints` | `list[str]`                       | `None`   | Natural-language constraints (e.g., `"quantity must be positive"`)                                             |
| `samples`     | `dict[str, File \| pd.DataFrame]` | `None`   | Sample data. Sampled for LLM context, converted to File inputs for the sandbox. Used as defaults at runtime.   |
| `inputs`      | `dict[str, type]`                 | `None`   | Non-sample CLI argument types (e.g., `{"threshold": float}`). Sample entries are auto-added as File inputs.    |
| `outputs`     | `dict[str, type]`                 | `None`   | Output types. Supported: `str, int, float, bool, datetime, timedelta, File`.                                   |

### `CodeGenEvalResult`

Returned by `agent.generate()`. Key fields:

```python
result.success        # bool — did tests pass?
result.solution       # CodeSolution — generated code
result.tests          # str — generated test code
result.output         # str — test output
result.exit_code      # int — test exit code
result.error          # str | None — error message if failed
result.attempts       # int — number of iterations used
result.image          # str — built sandbox image with all deps
result.detected_packages        # list[str] — pip packages detected
result.detected_system_packages # list[str] — apt packages detected
result.generated_schemas        # dict[str, str] | None — Pandera schemas as code
result.data_context             # str | None — extracted data context
result.original_samples         # dict[str, File] | None — sample data as Files
```

#### `result.as_task()`

Create a reusable sandbox from the generated code:

```python
task = result.as_task(name="run-on-data")

# Call with your declared inputs — returns a tuple of outputs
total_revenue, total_units, transaction_count = task(sales_csv=my_file)

# If samples were provided, they are injected as defaults — override as needed
total_revenue, total_units, transaction_count = task(threshold=0.5)  # samples used for data inputs

# With sandbox options
task = result.as_task(
    name="run-on-data",
    retries=3,
    timeout=600,
    env_vars={"API_URL": "https://..."},
)
```

The task runs the generated script in the built sandbox image. Inputs are passed as `--name value` CLI arguments. Outputs are read from `/var/outputs/{name}` files.

#### `result.run()`

One-shot execution using sample data as defaults:

```python
# Sync
total_revenue, total_units, transaction_count = result.run()

# Async
total_revenue, total_units, transaction_count = await result.run.aio()

# Override specific inputs
total_revenue, total_units, transaction_count = result.run(threshold=0.5)
```

Requires `samples` to have been passed to `generate()`.

## Data handling

When you pass `samples`, the plugin automatically:

1. **Converts DataFrames to CSVs** and uploads as `File` objects
2. **Infers Pandera schemas** — conservative type + nullability checks inferred from the sample data (no value constraints)
3. **Applies natural-language constraints** — if `constraints` are provided, each one is parsed by the LLM into a Pandera check (e.g., `"quantity must be positive"` → `pa.Check.gt(0)`) and added to the schema
4. **Extracts comprehensive context** — column stats, distributions, patterns, sample rows
5. **Includes everything in the prompt** — the serialized schemas and data context are injected into the LLM prompt so the generated code is aware of exact column types, nullability and validation rules

Pandera is used purely for **prompt enrichment**, not runtime validation. The generated code itself doesn't import Pandera — it just benefits from the LLM knowing the precise data structure. The schemas are also stored on `result.generated_schemas` for inspection.

```python
result = await agent.generate.aio(
    prompt="Clean and validate the data, remove duplicates",
    samples={"orders": orders_df, "products": products_file},
    constraints=["quantity must be positive", "price between 0 and 10000"],
    outputs={"cleaned_orders": File},
)

# Access generated schemas
print(result.generated_schemas)  # {"orders": "DataFrameSchema(...)", "products": "..."}
```

## Configuration

### Image configuration

```python
agent = AutoCoderAgent(
    name="my-task",
    image_config=ImageConfig(
        registry="my-registry.io",
        registry_secret="registry-creds",
        python_version=(3, 12),
    ),
)
```

### LiteLLM configuration

```python
agent = AutoCoderAgent(
    name="my-task",
    model="anthropic/claude-sonnet-4-20250514",
    api_key="ANTHROPIC_API_KEY",     # env var name
    litellm_params={
        "temperature": 0.3,
        "max_tokens": 4000,
    },
)
```

### Skipping tests

Set `skip_tests=True` to skip test generation and execution. The agent will still generate code, detect packages, and build the sandbox image, but won't generate or run tests. This is useful when you trust the LLM output or want faster turnaround.

```python
agent = AutoCoderAgent(
    name="my-task",
    model="gpt-4.1",
    skip_tests=True,  # No test generation or execution
)

result = await agent.generate.aio(
    prompt="Parse JSON logs and extract error counts",
    samples={"logs": log_file},
    outputs={"error_count": int},
)

# result.as_task() and result.run() still work
error_count = await result.run.aio()
```

> **Note:** `skip_tests` only applies to LiteLLM mode. In Agent SDK mode, the agent autonomously decides when to test.

### Environment setup

`sandbox_environment` must be listed as a dependency of your TaskEnvironment:

```python
from flyte.sandbox import sandbox_environment

env = flyte.TaskEnvironment(
    name="my-env",
    image=flyte.Image.auto(),
    depends_on=[sandbox_environment],  # Required
)
```

This allows dynamically-created sandboxes to be registered with Flyte.

> **Tip:** Use one `AutoCoderAgent` per task. Each `generate()` call builds its own sandbox image and manages its own package/image state. Running multiple agents in the same task can cause resource contention and makes failures harder to diagnose.

## Module Structure

```
codegen/
├── __init__.py              # Public API: AutoCoderAgent, CodeGenEvalResult, types
├── auto_coder_agent.py      # AutoCoderAgent — config + generate() orchestrator
├── core/
│   └── types.py             # Pydantic models: CodeGenEvalResult, CodeSolution, CodePlan, etc.
├── data/
│   ├── extraction.py        # Extract context from DataFrames/Files (stats, patterns, samples)
│   └── schema.py            # Pandera schema inference, constraint parsing via LLM
├── execution/
│   ├── agent_sdk.py         # Claude Agent SDK path with hooks and sandbox test interception
│   ├── docker.py            # Image building (create_image_spec, incremental builds)
│   └── testing.py           # Test execution in sandboxes
├── generation/
│   ├── llm.py               # LLM calls: plan, code, tests, diagnosis, fixes, verification
│   └── prompts.py           # Prompt templates and constants
```

### Data flow

```
User calls agent.generate(prompt, samples, outputs, ...)
│
├─ Data Processing (both paths)
│  ├─ Convert DataFrames → CSV Files
│  ├─ Infer Pandera schemas
│  ├─ Apply user constraints (LLM-parsed)
│  └─ Extract data context (stats, patterns, samples)
│
├─ LiteLLM Path (default)                 ├─ Agent SDK Path (use_agent_sdk=True)
│  ├─ generate_plan()                     │  ├─ Build prompt with all context
│  ├─ generate_code()                     │  ├─ Launch Claude agent with hooks:
│  ├─ detect_packages()                   │  │  ├─ PreToolUse: trace + classify commands
│  ├─ build_image()                       │  │  │  ├─ pytest → run in sandbox
│  ├─ execute_tests()                     │  │  │  ├─ safe (ls, cat, ...) → allow
│  ├─ diagnose_error() (if failed)        │  │  │  └─ denied (apt, pip, curl, ...) → block
│  ├─ fix code/tests/env                  │  │  ├─ PostToolUseFailure: trace errors
│  └─ repeat until pass or max_retries    │  │  └─ Stop: trace summary
│                                         │  ├─ Agent writes solution.py, tests.py, packages.txt
│                                         │  ├─ pytest intercepted → sandbox execution
│                                         │  └─ Agent iterates until tests pass
│
└─ Return CodeGenEvalResult
   ├─ .solution (code)
   ├─ .image (sandbox image with deps)
   ├─ .as_task() → reusable sandbox
   └─ .run() → execute on sample data
```

## Error handling

The LiteLLM path classifies test failures into three types:

| Type          | Meaning                    | Action                                           |
| ------------- | -------------------------- | ------------------------------------------------ |
| `logic`       | Bug in generated code      | Regenerate code with specific patch instructions |
| `environment` | Missing package/dependency | Add package, rebuild image                       |
| `test_error`  | Bug in generated test      | Fix test expectations                            |

If the same error persists after fixes, the plugin reclassifies it (logic <-> test_error) to try the other approach.

## Observability

### LiteLLM path

- Logs every iteration with attempt count, error type, and package changes
- Tracks total input/output tokens across all LLM calls
- Results include full conversation history for debugging

### Agent SDK path

- Traces each tool call (name + input detail) via `PreToolUse` hook
- Traces tool failures via `PostToolUseFailure` hook
- Traces a summary when the agent finishes (total tool calls, tool distribution, final image/packages)
- Classifies Bash commands as safe, denied, or pytest (intercepted for sandbox execution)
- All traces appear in the Flyte UI under the task

## Examples

See the `examples/` directory:

- **`example_csv_processing.py`** — Process CSVs with different schemas using LiteLLM. Shows batch processing with multiple CSV formats.
- **`example_csv_processing_sync.py`** — Synchronous version of CSV processing. Shows `agent.generate()` and `result.run()` without async.
- **`example_csv_processing_agent.py`** — CSV processing using Agent SDK with `use_agent_sdk=True`.
- **`example_dataframe_analysis.py`** — DataFrame analysis with constraints, `base_packages`, and `as_task()` for reusable execution.
- **`example_dataframe_analysis_agent.py`** — Same DataFrame analysis using Agent SDK.
- **`example_prompt_only.py`** — Log file analysis with `schema`, `constraints`, `samples`, and explicit `inputs`/`outputs`.
- **`example_prompt_only_agent.py`** — Same log analysis using Agent SDK.
- **`example_multi_input.py`** — Multi-input data join with primitives (`float`, `bool`).
- **`example_multi_input_agent.py`** — Same multi-input join using Agent SDK.
- **`example_durable_execution.py`** — Durable execution with injected failures, retries, and caching (LLM approach).
- **`example_durable_execution_agent.py`** — Same durable execution using Agent SDK.
