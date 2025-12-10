# Agent Handoff System

A deterministic workflow system for intelligently routing user queries to specialized agents using semantic matching and tag-based filtering.

## Overview

This example demonstrates a production-ready agent handoff system that:

1. **Extracts workflow tags** from user queries using GPT-4o-mini
2. **Filters the agent registry** by matching tags
3. **Scores candidates** using semantic similarity (OpenAI embeddings)
4. **Routes to the best agent** if score exceeds threshold
5. **Provides clear feedback** when no suitable agent is found

## Architecture

```
User Query
    ↓
[Tag Extraction] ──→ GPT-4o-mini extracts relevant workflow tags
    ↓
[Agent Filtering] ──→ Filter registry by matching tags
    ↓
[Semantic Scoring] ──→ Compute similarity with OpenAI embeddings
    ↓
[Agent Selection] ──→ Select if score > threshold
    ↓
[Handoff] ──→ Route to agent OR raise error
```

## Agent Registry

The system includes 10 specialized agents:

| Agent ID | Name | Tags | Purpose |
|----------|------|------|---------|
| `data-analyst` | Data Analytics Agent | data, analytics, sql, visualization, reporting | Data analysis and visualization |
| `code-reviewer` | Code Review Agent | code, review, testing, quality, debugging | Code quality and reviews |
| `doc-writer` | Documentation Agent | documentation, writing, technical, guides, api | Technical documentation |
| `ml-engineer` | Machine Learning Agent | ml, ai, models, training, deployment | ML model development |
| `devops-agent` | DevOps Automation Agent | devops, cicd, infrastructure, deployment, automation | CI/CD and infrastructure |
| `security-audit` | Security Audit Agent | security, audit, vulnerability, compliance, testing | Security assessments |
| `api-designer` | API Design Agent | api, rest, graphql, design, integration | API architecture |
| `customer-support` | Customer Support Agent | support, customer, troubleshooting, help, tickets | Customer assistance |
| `data-pipeline` | Data Pipeline Agent | data, pipeline, etl, ingestion, orchestration | ETL and data workflows |
| `frontend-dev` | Frontend Development Agent | frontend, ui, react, design, components | UI development |

## Setup

### Prerequisites

1. OpenAI API key
2. Flyte 2.0 environment

### Configure Secret

```bash
# Set your OpenAI API key as a secret
flyte secrets set openai-api-key OPENAI_API_KEY="sk-your-key-here"
```

## Usage

### Basic Usage

```python
import flyte

# Initialize Flyte
flyte.init_from_config()

# Run the handoff workflow
run = flyte.run(
    run_handoff,
    query="I need help analyzing sales data and creating visualizations",
    threshold=0.6
)

print(f"Run URL: {run.url}")
run.wait()
```

### Advanced Usage - Get Detailed Results

```python
# Get detailed workflow information
run = flyte.run(
    agent_handoff_workflow,
    query="Set up a CI/CD pipeline for deployment",
    threshold=0.6,
    show_top_n=10
)

run.wait()
result = run.result()

print(f"Extracted Tags: {result.extracted_tags}")
print(f"Filtered Agents: {result.filtered_count}")
print(f"Selected Agent: {result.selected_agent.name}")
print(f"\nTop Scores:")
for score in result.all_scores[:5]:
    print(f"  {score.agent.name}: {score.score:.3f}")
```

### Run from Command Line

```bash
# Run the example
python agent_handoff.py

# Or use uv to run with inline dependencies
uv run agent_handoff.py
```

## Example Queries

### Successful Handoffs

```python
# Data Analytics - Routes to Data Analytics Agent
"I need help analyzing sales data and creating a visualization dashboard"

# Machine Learning - Routes to ML Engineer Agent
"Can you help me train a neural network for image classification?"

# DevOps - Routes to DevOps Agent
"I want to set up a CI/CD pipeline for my application deployment"

# Security - Routes to Security Audit Agent
"Perform a security audit and check for vulnerabilities in my application"

# API Design - Routes to API Designer Agent
"Design a RESTful API for a user management system"
```

### Failed Handoffs (Ambiguous Queries)

```python
# Too generic - Will fail with high threshold
"I need some help with my project"

# No matching tags - Will show top candidates
"How do I make coffee?"
```

## Configuration

### Threshold Tuning

The `threshold` parameter controls how strict the matching is:

- **0.5-0.6**: Permissive - Accepts broader matches
- **0.7**: Default - Good balance of precision
- **0.8+**: Strict - Only very close matches

```python
# Permissive matching
run = flyte.run(run_handoff, query="...", threshold=0.5)

# Strict matching
run = flyte.run(run_handoff, query="...", threshold=0.85)
```

### Custom Agent Registry

Extend or modify `AGENT_REGISTRY` to add your own agents:

```python
Agent(
    id="your-agent",
    name="Your Custom Agent",
    description="Detailed description of what this agent does...",
    tags=["tag1", "tag2", "tag3"],
    threshold=0.7  # Optional: per-agent threshold
)
```

## Workflow Tasks

### 1. `extract_tags_from_query(query: str) -> List[str]`
Uses GPT-4o-mini to extract relevant workflow tags from the query.

**Example:**
```python
query = "I need help with SQL queries and data visualization"
tags = await extract_tags_from_query(query)
# Returns: ["data", "sql", "visualization"]
```

### 2. `filter_agents_by_tags(agents: List[Agent], tags: List[str]) -> List[Agent]`
Filters agents that have at least one matching tag.

**Example:**
```python
filtered = await filter_agents_by_tags(AGENT_REGISTRY, ["data", "sql"])
# Returns: [data-analyst, data-pipeline]
```

### 3. `score_agents(query: str, agents: List[Agent]) -> List[AgentScore]`
Computes semantic similarity between query and agent descriptions using OpenAI embeddings.

**Example:**
```python
scores = await score_agents("analyze sales data", filtered_agents)
# Returns: [AgentScore(agent=data-analyst, score=0.85), ...]
```

### 4. `select_agent(scored_agents: List[AgentScore], threshold: float) -> Optional[Agent]`
Selects the top agent if it meets the threshold.

**Example:**
```python
selected = await select_agent(scored_agents, threshold=0.7)
# Returns: Agent or None
```

### 5. `handoff_to_agent(agent: Agent, query: str) -> str`
Simulates handing off to the selected agent.

### 6. `agent_handoff_workflow(query: str, threshold: float, show_top_n: int) -> HandoffResult`
Main workflow that orchestrates all steps.

### 7. `run_handoff(query: str, threshold: float) -> str`
User-friendly wrapper that returns formatted results or raises errors.

## Error Handling

When no agent meets the threshold, the system raises a `RuntimeError` with details:

```
❌ HANDOFF FAILED

No agent met the threshold (0.8). Top 10 candidates:
  1. Data Analytics Agent (ID: data-analyst) - Score: 0.742
  2. ML Engineer Agent (ID: ml-engineer) - Score: 0.689
  3. DevOps Agent (ID: devops-agent) - Score: 0.651
  ...

Please clarify your query or lower the threshold.
```

## Performance

- **Tag Extraction**: ~500ms (GPT-4o-mini)
- **Embedding Generation**: ~200ms per agent (parallelized)
- **Total for 10 agents**: ~1-2 seconds

## Scaling to 100 Agents

For production with 100 agents:

1. **Batch embeddings**: Pre-compute agent embeddings and cache
2. **Use vector DB**: Store embeddings in Pinecone/Weaviate for fast retrieval
3. **Add caching**: Cache tag extraction for similar queries
4. **Optimize filtering**: Use more sophisticated tag matching

## Extending the System

### Add Multi-Agent Routing

For complex queries that need multiple agents:

```python
@env.task
async def multi_agent_handoff(query: str) -> List[Agent]:
    """Route to multiple agents in parallel."""
    result = await agent_handoff_workflow(query, threshold=0.6)
    top_3 = [score.agent for score in result.all_scores[:3]]
    return top_3
```

### Add Conversation Memory

Track conversation history for context-aware routing:

```python
class ConversationState(BaseModel):
    history: List[str]
    current_agent: Optional[Agent]

@env.task
async def contextual_handoff(state: ConversationState, query: str) -> ConversationState:
    """Route considering conversation history."""
    # Combine query with recent history
    context = " ".join(state.history[-3:] + [query])
    result = await agent_handoff_workflow(context)
    state.current_agent = result.selected_agent
    state.history.append(query)
    return state
```

### Add Agent Feedback Loop

Incorporate user feedback to improve routing:

```python
class AgentFeedback(BaseModel):
    agent_id: str
    query: str
    successful: bool
    score: float

@env.task
async def adaptive_handoff(query: str, feedback_history: List[AgentFeedback]) -> Agent:
    """Adjust scores based on historical feedback."""
    # Apply learned weights to improve routing
    # ...
```

## Testing

```bash
# Run with test queries
python agent_handoff.py

# Test specific query
python -c "
import flyte
from agent_handoff import run_handoff

flyte.init_from_config()
run = flyte.run(run_handoff, 'your test query here')
run.wait()
print(run.result())
"
```

## Troubleshooting

### "No module named 'openai'"
```bash
pip install openai>=1.0.0
```

### "Authentication error"
Ensure your OpenAI API key is set:
```bash
flyte secrets set openai-api-key OPENAI_API_KEY="sk-..."
```

### All queries fail threshold
Lower the threshold or improve agent descriptions for better semantic matching.

## Production Considerations

1. **Rate Limiting**: Add retry logic for OpenAI API calls
2. **Monitoring**: Track handoff success rates per agent
3. **A/B Testing**: Test different thresholds in production
4. **Fallback Agent**: Add a general-purpose agent as fallback
5. **Human in the Loop**: Add approval step for low-confidence handoffs

## References

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Flyte 2.0 Documentation](https://docs.flyte.org/)
- [Semantic Search Best Practices](https://www.pinecone.io/learn/semantic-search/)
