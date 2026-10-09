# Artifacts, lineage and materialization

These examples teach one idea in four steps: **name what your tasks produce, declare what they read, and let the
platform work out what to run.** You stop writing orchestration DAGs, backfill scripts and deploy glue. You ask
for the data, model or app you need, and the platform plans, reuses and builds.

Start at 1 and stop wherever you have what you need. Each step works on its own.

| | Example | You'll learn | Needs |
|---|---|---|---|
| 1 | [`1_basics/`](1_basics/) | Publish and read artifacts; partitions; declare them as handles next to the code | any Flyte 2 backend |
| 2 | [`2_etl_backfill/`](2_etl_backfill/) | A partitioned ETL pipeline: plan, build, backfill a range, reuse, rebuild, missing data, a nightly refresh | Union lineage* |
| 3 | [`3_train_and_serve/`](3_train_and_serve/) | Retrain when new data is published, serve the model from an app, quality gates, rollbacks, a train-and-serve factory | Union lineage* |
| 4 | [`4_multi_team/`](4_multi_team/) | Four teams, one graph nobody wrote: cross-repo references, sinks, triggers, snapshots, factories and diffs | Union lineage* |

\* Deploying and running the tasks works on any Flyte 2 backend. Planning and materializing (`flyte materialize`,
`flyte.materialize`, `refresh=`, `flyte factory`) are experimental and need the Union lineage service and
`flyteplugins-union` (`pip install flyteplugins-union`).

## The concepts, in one page

**Artifact.** A task output (`File`, `Dir`, `DataFrame`) published under a name. Every publish is a new version that
records the run that produced it. `Artifact.get("churn_model")` is "the latest churn model", anywhere.

**Partition.** Part of an artifact's identity: `trips[date=2026-09-07, city=nyc]`. A time dimension
(`artifacts.Daily`) lets a range like `2026-09-01..2026-09-07` expand into partitions.

**Handle.** An artifact declared once at module level, with its type and partitions, then named in task decorators:

```python
trips = artifacts.Artifact("trips", type=DataFrame, partitions={"date": artifacts.Daily, "city": str})

@env.task(consumes_artifacts={"raw": raw_trips}, produces_artifacts=(trips,))
async def clean(raw: File, date: datetime, city: str) -> DataFrame: ...
```

Deploy checks the declarations against the signature and records them. A team that can't import a handle
(another repo) states it with `Artifact.ref(name, partitions=...)`.

**Source.** An artifact nothing in the graph builds (`source=True`). It lands from outside, and the planner reads
it, or reports it missing.

**Mapping.** How a task reads partitions, declared on the input:

| Declaration | To build partition D, reads |
|---|---|
| `raw_trips` (identity) | the same partition |
| `trips.all("city")` | every city of D, as a list |
| `daily_stats.window(date=TimeRange(days=7))` | the 7 days up to D |

A parameter named like a dimension (`date`, `city`) receives the value being built.

**Lineage graph.** What all deployed declarations imply, across every team and repo. Nobody draws it; the console's
**Lineage** view shows it, and it updates within a minute of a deploy.

**Materialize.** Ask for partitions of any artifact, or for an app; the planner walks the graph backwards and
builds only what is missing:

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-01..2026-09-30 --plan   # what would run
flyte materialize artifact weekly_stats --partition date=2026-09-01..2026-09-30          # a backfill
flyte materialize app sentiment-api --partition date=2026-09-15                          # build, then serve
```

The same from Python (`flyte.materialize(...)`) and from the console (the play button on any graph node).

**Reuse.** A partition is reused when the same task version already built it from the same inputs: that's
the task cache. Repeating a request is cheap. `--input task.param=v` changes the key and `--rebuild task`
skips it.

**Refresh.** Keep an artifact fresh by declaring when, on the handle: `refresh=Refresh(flyte.Cron(...))` for a
schedule, or `refresh=Refresh(source_handle)` to rebuild when new data is published. Deploying registers the
trigger.

**Serve.** An app with `consumes_artifacts={"model": model}` is a graph node too. Materializing it builds the
model and redeploys the app pinned to it. It never moves back to an older partition unless you pin one
(`--version`).

**Factory.** The graph, frozen: every task pinned to a deployed version, with its triggers, in one reviewed file.
`flyte factory snapshot <node> -o f.yaml` writes one from the live graph, and `flyte factory diff` shows what
changed upstream since.

## Why

Every ML, AI and data team ends up rebuilding the same plumbing:

- **Paths instead of names.** "The features for 9 September" is an S3 path convention someone has to remember.
- **One DAG everybody edits.** The pipeline behind a dashboard spans teams but lives in one repo; every upstream
  change is a cross-team PR.
- **Backfills are scripts.** Someone writes the loop and guesses what can be skipped.
- **"Where did this come from?"** Finding the model and data behind a bad report means digging through logs.
- **Fast vs. stable.** Teams ship whenever they like, but production shouldn't change under you.

| You are | Start with | Before | After |
|---|---|---|---|
| Data engineer | 2 | A shared DAG, backfill scripts, cache logic by hand | A backfill is a range; reuse is automatic; missing data is named before anything runs |
| ML engineer | 3 | Retraining cron jobs, models in buckets, deploy scripts | New data retrains; the app serves a pinned, versioned model; rollback by version |
| Platform team | 4 | Prod pipelines that change whenever any team deploys | Pinned, versioned factories with a diff before every change |
| Analyst | 4 | "Is today's report built on today's model?" | Every partition links to what made it |

## Running the examples

Each example's README walks through it step by step, with the output to expect, and its `run.sh` runs the whole
walkthrough:

```bash
cd 2_etl_backfill   && ./run.sh --config <config> [--queue <queue>]
cd 3_train_and_serve && ./run.sh --config <config> [--queue <queue>]
cd 4_multi_team      && ./run_e2e.sh --config <config> [--queue <queue>]
```

The examples use separate artifact names, so they can share one project.
