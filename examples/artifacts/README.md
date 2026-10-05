# From artifacts to factories

These examples follow one path. You start by giving a task's output a name, and you end with a versioned
production pipeline that several teams feed. Nobody writes an orchestration DAG along the way. Each stage is
opt-in and builds on the previous one, so you can stop at any stage and keep what you have.

| Stage | You write | You get | Example |
|---|---|---|---|
| 1. Artifacts | `artifacts.new(value, Metadata(name, partitions))` | Named, versioned, partitioned outputs that record what made them | `artifact_example.py`, `partitioned_artifacts.py`, `produced_artifacts.py` |
| 2. Handles | `Artifact(...)` at module top, named in `@env.task(consumes_artifacts=, produces_artifacts=)` | Checks at deploy time, plus a graph drawn from the code | `handles_example.py` |
| 3. Emergent lineage | Nothing new: each team declares its own handles | One graph across teams and repos, with no shared DAG | `emergent_lineage/` |
| 4. Pull | `flyte.materialize(handle, date=...)`, `Artifact(refresh=...)`, `handle.materialize_on(...)` | Builds, backfills and schedules where the platform plans what to run | `emergent_lineage/pull.py`, `analytics/report.py`, `triggers/weekly_review.py` |
| 5. Factory | `flyte factory snapshot`, or `fc.build(...)` / YAML | The same graph, frozen and pinned, ready to review and run in production | `emergent_lineage/factories/churn.py`, `churn.yaml` |
| 6. Evolve | `flyte factory diff`, then redeploy | Upstream changes adopted on your schedule, with rollbacks to known versions | `emergent_lineage/README.md` |

## Why this exists

Every ML, AI and data team ends up rebuilding the same plumbing:

- **Paths instead of names.** Outputs are S3 prefixes in config files. "The features for 9 September" is a path
  convention. Whether a path holds the latest run, or which code wrote it, is in someone's head.
- **One DAG that everybody edits.** The pipeline that feeds the dashboard spans three teams. It lives in one repo,
  owned by whoever wrote it first, and every upstream change is a cross-team PR into it.
- **Backfills are scripts.** Rebuilding a month means a loop someone writes by hand. They guess which steps
  can be skipped, and rerun everything when they're unsure.
- **No answer to "where did this come from?"** When the report looks wrong, finding the model, the features and the
  raw data behind it means digging through logs.
- **Moving fast and staying stable fight each other.** Teams want to ship whenever they like. The pipeline behind
  the morning report needs not to change under you.

The tools below remove each of these in turn, without asking anyone to adopt a framework up front.

## Stage 1: name your outputs (artifacts)

An artifact is a task output (a `File`, `Dir` or `DataFrame`) published under a name, with optional partitions
such as `date` and `region`. Each publish is a new version and records the action that made it.

```python
@env.task(produces_artifacts=True)
async def train(...) -> File:
    return artifacts.new(model_file, artifacts.Metadata(name="churn_model", partitions={"date": day}))

model = Artifact.get("churn_model", date=date(2026, 9, 8))   # anywhere: another task, a notebook, an app
```

- `artifact_example.py`: publish, produce and consume, including multi-output tasks.
- `partitioned_artifacts.py`: partitions as identity, `listall` over a range, and `partition_values`.
- `produced_artifacts.py`: publish another task's outputs without editing the task (`artifacts.produces(...)`).

**What it's worth:** there's no path convention left to maintain. "The latest churn model" and "the features for
the 8th, region us" become lookups. Every version records the run that produced it.

## Stage 2: declare artifacts next to the code that makes them (handles)

In stage 1, `train` and the code that reads its model are connected only by a string. In stage 2 a module
declares its artifacts once, as **handles**, and names them in the task decorator:

```python
orders  = artifacts.Artifact("orders",  type=DataFrame, partitions={"date": artifacts.Daily})
revenue = artifacts.Artifact("revenue", type=DataFrame, partitions={"date": artifacts.Daily})

@env.task(consumes_artifacts={"week": orders.window(date=TimeRange(days=7))}, produces_artifacts=(revenue,))
async def daily_revenue(week: list[DataFrame], date: datetime) -> DataFrame: ...
```

```
$ flyte deploy handles_example.py env
  ✓ 3 tasks, 3 artifact handles, 2 dependency edges resolved
```

Deploy now checks the declarations against the task signatures. It catches a dimension nothing supplies, an
input that no handle feeds, or a typo in a name, and refuses to ship anything that can't be planned. It also
records what each task produces and consumes, so the console draws `raw_orders → clean_orders → orders →
daily_revenue → revenue` without anyone drawing it.

The declaration covers the details that orchestration code usually spells out by hand:

- `orders.window(date=7d)` reads a trailing window.
- `events.all("region")` fans in every value of a dimension.
- Identity reads the same partition.
- A parameter named like a dimension (`date`) receives that dimension's value.

**What it's worth:** the data contract moves into code review, next to the function it describes. Mistakes
surface at deploy, not at 3am. The handle is a plain Python object, so other teams import it instead of copying
strings. A team in another repo, which can't import it, states it with `Artifact.ref` (see stage 3).

## Stage 3: one graph, many teams (emergent lineage)

`emergent_lineage/` is the same idea across four teams:

- Data Platform cleans events.
- ML builds features, trains a model and serves it.
- Analytics builds a report, emails it and keeps it fresh.
- ML Quality validates every new model.

Each team has its own module, environment and `flyte deploy`. No module imports another team's task. They
share only handles: imported within a repo, or restated across repos with `Artifact.ref`. A reference states the
name and partitions to read, and deploy checks them against the owner's declaration in the registry:

```python
# analytics repo: the ML team's artifacts, which this repo cannot import
features    = artifacts.Artifact.ref("features", type=DataFrame, partitions={"date": artifacts.Daily})
churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})
```

```
raw_events ──clean──▶ events ──featurize (all region)──▶ features ──train (30d)──▶ churn_model ──▶ app:churn-scoring
                                                          └──report (7d)──▶ daily_report ◀── (identity)
daily_report ──▶ send_report (sink)       churn_model ──▶ trigger:revalidate-on-new-model
```

That graph isn't written down anywhere. It emerges from the declarations at deploy time and appears in the
console's **Lineage** view. Every node links to its artifact, task, app or trigger. Each team keeps its own
entities next to its own code:

- The sink (`analytics/notify.py`) consumes the report and publishes nothing.
- The outbound trigger (`triggers/revalidate.py`, `flyte.OnArtifact(churn_model)`) can be attached before any model
  exists.
- The app (`apps/scoring.py`, `consumes_artifacts={"model": churn_model}`) is served from an artifact.

**What it's worth:** teams ship on their own schedules and the end-to-end picture stays accurate, because it is
computed from what is actually deployed. Every edge is typed and planned, or explicitly label-only.
"Which model made this report?" becomes one click.

## Stage 4: ask for what you need (pull)

Once the graph exists, you don't run pipelines. You ask for an artifact partition, and the platform walks the
declarations backwards. It reuses every partition already built by the same code from the same inputs, and
builds only what is missing.

```python
flyte.materialize(daily_report, date=datetime(2026, 9, 8), plan_only=True)        # what would run, launches nothing
flyte.materialize(daily_report, date=TimeRange("2026-09-02", "2026-09-08"), concurrency=8)  # a backfill
flyte.materialize(daily_report, date=day, rebuild=["ml.featurize"])                # force one step and what is downstream
flyte.materialize(send_report, date=day)                                           # a sink is a target too
```

```
92 instances across 4 builds, 60 lookups. The cache decides what runs; nothing was launched.
```

The same pull is available in four places:

- In code: `emergent_lineage/pull.py`.
- On the CLI: `flyte materialize daily_report --partition date=2026-09-08 --plan`.
- In the console: the play button on any graph node, which opens a live materialization graph.
- On a schedule, declared once by the artifact's owner, on the handle. Deploying the module that produces it
  registers the trigger:

```python
daily_report = artifacts.Artifact("daily_report", type=File, partitions={"date": artifacts.Daily},
                                  refresh=artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=TimeRange(days=1)))
```

A team that only reads an artifact, even through an `Artifact.ref`, keeps it fresh with
`churn_model.materialize_on(flyte.Cron("0 5 * * 1"), lag=TimeRange(days=1))` (`triggers/weekly_review.py`).
Either way, the trigger plans the target partition and builds whatever is missing upstream.

**What it's worth:**

- Backfills are a range instead of a script.
- Nobody decides what to skip; the cache key does.
- The 30-day training window and the 7-day report window are planned for you, partition by partition.
- When a source partition is missing, the plan names it before anything runs, and the console shows how to
  publish it.

## Stage 5: freeze it for production (factories)

The emergent graph always reflects whatever each team deployed last. That is right while building, and wrong
behind a report the business reads every morning, because one team's redeploy shouldn't silently change it.

A **factory** is the same graph, frozen. It is functionally equivalent to the lineage graph: sources, builds,
sinks, served apps, inbound schedules and outbound triggers all have a direct counterpart. The difference is
that every task is pinned to a version and the whole thing is versioned as one object.

```bash
flyte factory snapshot send_report -o factories/churn.yaml      # or .py; or "Snapshot" in the console
flyte factory validate factories/churn.yaml
flyte factory deploy   factories/churn.yaml
flyte factory materialize churn send_report --partition date=2026-09-08 --wait
```

You can write one directly in Python. `emergent_lineage/factories/churn.py` reuses the teams' handles and calls
their deployed tasks by name, so it contains no business logic:

```python
clean      = fc.build(events).using(Task.get("ingest.clean"), raw=fc.source(raw_events))
featurized = fc.build(features).using(Task.get("ml.featurize"), per_region=clean.all("region"))
model      = fc.build(churn_model).using(Task.get("ml-train.train"), history=featurized.window(date=TimeRange(days=30)))
report     = fc.build(daily_report).using(Task.get("analytics.report"), week=featurized.window(date=TimeRange(days=7)), model=model)
sent       = fc.sink("send_report").using(Task.get("analytics-notify.send_report"), report=report)

churn = fc.Factory("churn", sent, fc.serve("churn-scoring").using("churn-scoring", model=model), triggers=[
    fc.on(flyte.Cron("0 2 * * *"), sent, lag=TimeRange(days=1)),
    fc.trigger(Task.get("ml-quality.validate"), on=model, inputs={"model": flyte.TriggeredArtifact, "threshold": 0.82}),
])
```

`emergent_lineage/factories/churn.yaml` is the same factory as a spec with no Python. The console's factory
editor reads and writes this format next to a live graph.

| In the lineage graph (colocated, live) | In a factory (one file, pinned) | In the spec |
|---|---|---|
| `Artifact(..., source=True)` | `fc.source(handle)` | `source: true` |
| `@env.task(consumes_artifacts=..., produces_artifacts=...)` | `fc.build(handle).using(task, **inputs)` | `builds[]` |
| a task that consumes and publishes nothing | `fc.sink(name).using(task, **inputs)` | `sink:` |
| `AppEnvironment(consumes_artifacts=...)` | `fc.serve(name).using(app, **params)` | `serve:` |
| `Artifact(..., refresh=artifacts.Refresh(cron, lag=))`, or `handle.materialize_on(cron, lag=)` | `fc.on(cron, target, lag=)` | `triggers[].targets` |
| `flyte.Trigger(automation=flyte.OnArtifact(handle))` | `fc.trigger(task, on=handle, inputs=)` | `triggers[].run` |
| a label-only `lineage.consumes` | `fc.reference(name)` | `references:` |

**What it's worth:**

- Production runs a graph you reviewed, not one you inherited.
- The tasks are the teams' own deployed versions, so nothing is forked or rewritten.
- The schedule, the email and the downstream checks are versioned with the graph they belong to.
- Platform engineers get one object to deploy, promote and audit. Data scientists still never write a DAG.

## Stage 6: evolve on your schedule

Teams keep shipping. The factory doesn't move until you choose:

```bash
flyte factory diff churn                         # which tasks have new versions upstream, and what changed in the graph
flyte factory snapshot send_report -o factories/churn.yaml && git diff   # adopt them as a reviewable change
flyte factory deploy factories/churn.yaml
flyte factory materialize churn churn-scoring --version churn_model=<v>   # roll the served model back to a known version
```

**What it's worth:** "stable" and "fast" stop competing. Upstream teams iterate in the live lineage graph. The
production factory adopts their work as a reviewed diff, and you can roll back to the last known-good version.

## Who gets what

| You are | Before | After |
|---|---|---|
| **ML engineer** | Models in buckets, training windows assembled by hand, no record of which model served when | `churn_model` versions with cards and provenance; `window(30d)` planned for you; the app served from an artifact, rolled back by version |
| **AI / LLM engineer** | Eval sets, prompts and checkpoints scattered across runs | The same handles for datasets, checkpoints and endpoints, with an outbound trigger to run evals on every new checkpoint |
| **Data engineer** | A shared DAG, hand-written backfills, cache logic in scripts | Each team owns its modules; a backfill is a range; reuse decided by code and input versions |
| **Analyst / report owner** | "Is today's report built on today's model?" | Every partition links to what made it; the nightly schedule and the email live in one reviewed factory |
| **Platform team** | Prod pipelines that change whenever any team deploys | Pinned, versioned factories with a diff before every change |

## Running the examples

Stages 1 and 2 run on any Flyte 2 backend. Stages 3 to 6 need the Union lineage service and
`flyteplugins-union` (`pip install flyteplugins-union`) for the `flyte materialize` and `flyte factory` commands.

```bash
python artifact_example.py                                # stage 1
flyte deploy handles_example.py env                       # stage 2
cd emergent_lineage && ./run_e2e.sh --config <config>     # stages 3-4: deploy every team, seed, materialize
python pull.py --config <config> plan                     # stage 4 from Python
flyte factory deploy factories/churn.py --dry-run         # stage 5
```

`emergent_lineage/README.md` covers the multi-team example in depth, including devbox notes.
