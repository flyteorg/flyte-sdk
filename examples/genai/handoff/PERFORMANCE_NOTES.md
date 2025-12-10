# Agent Handoff - Performance Optimizations

## Overview

The agent handoff system has been optimized for high performance with the following design decisions:

## Architecture Choices

### 1. **Task vs Trace vs Pure Functions**

| Function | Type | Rationale |
|----------|------|-----------|
| `extract_tags_from_query` | `@flyte.trace` | Calls OpenAI API - needs observability, lightweight tracing |
| `filter_agents_by_tags` | Pure function | Deterministic logic - no external calls, no overhead needed |
| `get_embedding` | `@flyte.trace` | Calls OpenAI API - needs observability, lightweight tracing |
| `score_agents` | `@flyte.trace` | Orchestrates embeddings - needs observability for debugging |
| `select_agent` | Pure function | Simple comparison logic - deterministic, no overhead |
| `handoff_to_agent` | `@flyte.trace` | Simulates agent interaction - lightweight trace for observability |
| `agent_handoff_workflow` | `@env.task` | Main workflow - full task for checkpointing and retries |
| `run_handoff` | `@env.task` | User-facing entry point - full task for error handling |

### 2. **Grouping Strategy**

**Before (Over-grouped):**
```python
with flyte.group("tag-extraction"):
    ...
with flyte.group("agent-filtering"):
    ...
with flyte.group("agent-scoring"):
    ...
with flyte.group("agent-selection"):
    ...
```

**After (Optimized):**
```python
with flyte.group("agent-selection"):
    # All operations consolidated
    # Traces provide granular observability
    extracted_tags = await extract_tags_from_query(query)  # trace
    filtered_agents = filter_agents_by_tags(...)           # pure function
    scored_agents = await score_agents(query, ...)         # trace
    selected = select_agent(scored_agents, threshold)      # pure function
```

**Benefits:**
- ✅ Reduced overhead from fewer group contexts
- ✅ Single logical unit in UI
- ✅ Traces still provide granular visibility
- ✅ Simpler execution graph

### 3. **Concurrency Configuration**

```python
env = flyte.TaskEnvironment(
    "agent-handoff",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_uv_script(__file__, name="agent-handoff", pre=True),
    concurrency=10,  # Allow 10 parallel executions
)
```

**Benefits:**
- ✅ Reusable environment across multiple requests
- ✅ Up to 10 parallel handoff workflows
- ✅ Efficient resource utilization

## Performance Characteristics

### Latency Breakdown

For a typical query with 10 agents:

| Operation | Type | Latency | Parallelized |
|-----------|------|---------|--------------|
| Tag extraction | Trace (OpenAI) | ~500ms | No |
| Agent filtering | Pure function | <1ms | N/A |
| Get query embedding | Trace (OpenAI) | ~200ms | No |
| Get agent embeddings (10x) | Trace (OpenAI) | ~200ms each | ✅ Yes |
| Compute similarities | Pure function | <10ms | N/A |
| Select agent | Pure function | <1ms | N/A |
| **Total** | | **~900ms** | |

**Key Optimization:** Agent embeddings are computed in parallel using `asyncio.gather`, reducing 10x200ms = 2000ms to just 200ms.

### Scaling to 100 Agents

With 100 agents, the bottleneck is embedding generation:
- Sequential: 100 × 200ms = 20 seconds ❌
- Parallel (current): ~200ms ✅

**Further optimizations for production:**

1. **Pre-compute and cache agent embeddings**
   ```python
   # Cache agent embeddings at startup
   AGENT_EMBEDDINGS = {
       agent.id: await get_embedding(agent.description)
       for agent in AGENT_REGISTRY
   }
   ```
   This reduces per-query latency to just query embedding (~200ms) + similarity computation (<10ms).

2. **Use vector database for large registries**
   ```python
   # Store embeddings in Pinecone/Weaviate
   # Fast k-NN search instead of computing all similarities
   results = vector_db.search(query_embedding, top_k=10)
   ```

3. **Batch processing for multiple queries**
   ```python
   # Process multiple user queries in parallel
   tasks = [agent_handoff_workflow(q) for q in queries]
   results = await asyncio.gather(*tasks)
   ```

## Overhead Comparison

### Function Call Overhead

| Type | Overhead | Use Case |
|------|----------|----------|
| Pure function | ~0μs | Deterministic logic, no I/O |
| `@flyte.trace` | ~10-100μs | Observability without checkpointing |
| `@env.task` | ~1-10ms | Full task with retries, checkpointing |

### Example Calculation

For 100 agent evaluations:
- **All tasks:** 100 × 5ms = 500ms overhead
- **Traces + pure:** 100 × 0.05ms = 5ms overhead
- **Savings:** 495ms (99% reduction)

## Monitoring and Observability

Despite using pure functions, we maintain full observability:

1. **Traces** capture:
   - OpenAI API calls (latency, tokens, cost)
   - Embedding generation per agent
   - Scoring computation

2. **Task** captures:
   - Complete workflow execution
   - Input/output parameters
   - Success/failure status
   - Retry attempts

3. **Group** provides:
   - Logical organization in UI
   - Single expandable section showing all steps

## Best Practices

### ✅ DO:
- Use pure functions for deterministic logic
- Use `@flyte.trace` for I/O operations that need observability
- Use `@env.task` for top-level workflows that need retries
- Consolidate related operations in a single group
- Parallelize independent operations with `asyncio.gather`

### ❌ DON'T:
- Make pure functions into tasks (unnecessary overhead)
- Create too many nested groups (cluttered UI)
- Use tasks for every function (breaks performance)
- Forget to parallelize independent I/O operations

## Performance Testing

To benchmark the current implementation:

```python
import time

# Test single handoff
start = time.time()
result = await agent_handoff_workflow(
    "Analyze sales data and create dashboard",
    threshold=0.6
)
duration = time.time() - start
print(f"Latency: {duration:.2f}s")

# Test parallel handoffs
queries = [EXAMPLE_QUERIES[0]] * 10
start = time.time()
results = await asyncio.gather(*[
    agent_handoff_workflow(q, threshold=0.6)
    for q in queries
])
duration = time.time() - start
print(f"10 parallel queries: {duration:.2f}s")
print(f"Per-query avg: {duration/10:.2f}s")
```

Expected results:
- Single query: ~900ms
- 10 parallel queries: ~1.5s total (due to concurrency)
- Per-query average: ~150ms (thanks to parallelization)

## Production Recommendations

For production deployments with high throughput:

1. **Cache agent embeddings** - Pre-compute at startup
2. **Increase concurrency** - Set to expected peak load
3. **Add rate limiting** - Respect OpenAI API limits
4. **Monitor trace metrics** - Set up alerts for slow queries
5. **Use vector DB** - For registries > 50 agents
6. **Add circuit breakers** - Protect against OpenAI downtime

## Conclusion

The optimized architecture provides:
- ✅ **Sub-second latency** for typical queries
- ✅ **High throughput** with parallel execution
- ✅ **Full observability** with minimal overhead
- ✅ **Scalability** to 100+ agents with caching
- ✅ **Production-ready** error handling and retries
