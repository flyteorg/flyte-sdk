# Agent Handoff System

A high-performance agent routing system using tag-based filtering and semantic similarity matching with OpenAI.

## Project Structure

```
handoff/
├── pyproject.toml           # Project dependencies and configuration
├── agent_handoff.py         # Main implementation
├── README.md                # This file
├── README_AGENT_HANDOFF.md  # Detailed usage guide
└── PERFORMANCE_NOTES.md     # Performance optimization details
```

## Quick Start

### 1. Install Dependencies

This project uses `uv` for dependency management:

```bash
# Install with uv
cd examples/genai/handoff
uv sync

# Or install globally
uv pip install -e .
```

### 2. Configure OpenAI API Key

Set your OpenAI API key as a Flyte secret:

```bash
flyte secrets set openai-api-key OPENAI_API_KEY="sk-your-key-here"
```

### 3. Run the Example

```bash
# Using Python
python agent_handoff.py

# Or using uv
uv run agent_handoff.py
```

## How It Works

The image is built using `Image.from_uv_project()` which:
1. Reads dependencies from `pyproject.toml`
2. Creates a Docker image with all required packages
3. Uses `uv` for fast dependency resolution

```python
image = flyte.Image.from_uv_project(
    root_dir=pathlib.Path(__file__).parent,
    name="agent-handoff",
)
```

## Development

### Adding Dependencies

Edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "flyte>=2.0.0b35",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "your-new-package>=1.0.0",  # Add here
]
```

Then sync:

```bash
uv sync
```

### Running Tests

```bash
# Add test dependencies
uv add --dev pytest

# Run tests
uv run pytest
```

### Code Formatting

```bash
# Add ruff for linting
uv add --dev ruff

# Format code
uv run ruff format .

# Check for issues
uv run ruff check .
```

## Environment Configuration

The `TaskEnvironment` is configured for high performance:

```python
env = flyte.TaskEnvironment(
    "agent-handoff",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_uv_project(...),
    reusable=flyte.ReusePolicy(
        replicas=(1, 5),      # Scale between 1-5 replicas
        concurrency=10,       # 10 concurrent executions per replica
    ),
)
```

## Usage Examples

### Basic Handoff

```python
import flyte
from agent_handoff import run_handoff

flyte.init_from_config()

# Run agent selection
run = flyte.run(
    run_handoff,
    query="I need help analyzing sales data and creating visualizations",
    threshold=0.6
)

print(f"Run URL: {run.url}")
run.wait()
```

### Advanced - Get Detailed Results

```python
from agent_handoff import agent_handoff_workflow

run = flyte.run(
    agent_handoff_workflow,
    query="Set up a CI/CD pipeline for deployment",
    threshold=0.7,
    show_top_n=10
)

run.wait()
result = run.result()

print(f"Selected Agent: {result.selected_agent.name}")
print(f"Extracted Tags: {result.extracted_tags}")
print(f"Top Scores:")
for score in result.all_scores[:5]:
    print(f"  {score.agent.name}: {score.score:.3f}")
```

## Architecture

- **Pure Functions**: `filter_agents_by_tags`, `select_agent` (zero overhead)
- **Traces**: `extract_tags_from_query`, `get_embedding`, `score_agents` (lightweight observability)
- **Tasks**: `agent_handoff_workflow`, `run_handoff` (full checkpointing and retries)

See `PERFORMANCE_NOTES.md` for detailed performance characteristics.

## Customization

### Adding New Agents

Edit the `AGENT_REGISTRY` in `agent_handoff.py`:

```python
Agent(
    id="your-agent",
    name="Your Custom Agent",
    description="Detailed description for semantic matching...",
    tags=["tag1", "tag2", "tag3"],
    threshold=0.7
)
```

### Adjusting Threshold

```python
# More permissive
run = flyte.run(run_handoff, query="...", threshold=0.5)

# More strict
run = flyte.run(run_handoff, query="...", threshold=0.85)
```

## Deployment

### Local Deployment

```bash
flyte deploy agent_handoff.py
```

### Production Deployment

```bash
# Build and push image
flyte build agent_handoff.py --push

# Deploy to Flyte cluster
flyte deploy agent_handoff.py --domain production
```

## Troubleshooting

### "No module named 'openai'"

The dependencies are installed in the Docker image. If running locally:

```bash
uv sync
source .venv/bin/activate  # Activate virtual environment
```

### "Authentication error"

Ensure your OpenAI API key is set correctly:

```bash
flyte secrets set openai-api-key OPENAI_API_KEY="sk-..."

# Verify
flyte secrets get openai-api-key
```

### Image Build Fails

Check that `pyproject.toml` is valid:

```bash
uv check
```

## Performance

- **Single query**: ~900ms
- **10 parallel queries**: ~1.5s total
- **Per-query average**: ~150ms (with concurrency)

For scaling to 100+ agents, see recommendations in `PERFORMANCE_NOTES.md`.

## Documentation

- **README_AGENT_HANDOFF.md** - Comprehensive usage guide with examples
- **PERFORMANCE_NOTES.md** - Detailed performance analysis and optimizations

## License

See main Flyte SDK license.
