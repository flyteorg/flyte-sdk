# 1. Basics: name your outputs, then declare them

Two ideas that work on any Flyte 2 backend, and that everything after this builds on.

## Artifacts: publish under a name

An artifact is a task output (a `File`, `Dir` or `DataFrame`) published under a name, with optional partitions.
Each publish is a new version and records the run that produced it.

```python
@env.task(produces_artifacts=True)
async def train(...) -> File:
    return artifacts.new(model_file, artifacts.Metadata(name="churn_model", partitions={"date": day}))

model = Artifact.get("churn_model", date=date(2026, 9, 8))   # anywhere: another task, a notebook, an app
```

| File | Shows |
|---|---|
| `artifact_example.py` | Publish, produce and consume, including tasks with several outputs |
| `partitioned_artifacts.py` | Partitions as identity, `listall` over a range, `partition_values` |
| `produced_artifacts.py` | Publish another task's outputs without editing that task (`artifacts.produces(...)`) |

```bash
python artifact_example.py
flyte run partitioned_artifacts.py main
```

## Handles: declare artifacts next to the code that makes them

With plain artifacts, a producer and its readers are connected only by a string. A **handle** declares the
artifact once, at module level, and the task decorators name it:

```python
orders  = artifacts.Artifact("orders",  type=DataFrame, partitions={"date": artifacts.Daily})
revenue = artifacts.Artifact("revenue", type=DataFrame, partitions={"date": artifacts.Daily})

@env.task(consumes_artifacts={"trailing_data": orders.window(date=TimeRange(days=3))}, produces_artifacts=(revenue,))
async def daily_revenue(trailing_data: list[DataFrame], date: datetime) -> DataFrame: ...
```

```
$ flyte deploy handles_example.py env
  ✓ 3 tasks, 3 artifact handles, 2 dependency edges resolved
```

Deploy checks every declaration against the task signature: a dimension nothing supplies, an input no handle
feeds, or a typo in a name fails the deploy. It also records what each task produces and reads, so the console
draws `raw_orders → orders → revenue` without anyone drawing it.

| File | Shows |
|---|---|
| `handles_example.py` | Three handles, a source, a 3-day window, a parameter bound by its name (`date`) |

**Next:** [`../2_etl_backfill/`](../2_etl_backfill/): ask the platform for partitions of a pipeline declared this
way, and let it plan, reuse and backfill.
