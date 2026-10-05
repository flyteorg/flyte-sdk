# Emergent lineage: four teams, one graph nobody wrote

Stages 3 to 6 of the journey in [`../README.md`](../README.md): the graph emerges from each team's declarations
(stage 3), you pull what you need from it (stage 4, `pull.py`), and freeze it into a production factory
(stages 5-6, `factories/`).

Each module here belongs to a different team, has its own `TaskEnvironment` or `AppEnvironment`, and is
deployed with its own `flyte deploy`. No module imports another team's task. They share only artifact
handles (plain Python objects) and labels, and the lineage graph falls out of those declarations at deploy.

Sharing a handle by import works while the teams share a repo. Across repos, a team states what it reads with
`artifacts.Artifact.ref(name, partitions=...)` instead, the way `analytics/report.py` reads the ML team's
`features` and `churn_model`. A reference is an `Artifact`, so it reads exactly like one, but it carries only
what a reader may state. It cannot be produced (`produces_artifacts`, `.at()`, `.expect()` refuse it), and the
platform always ranks it below the owner's declaration. `flyte deploy` checks each reference's partitions
against the registry, before building anything:

```
$ flyte deploy --root-dir . analytics/report.py env
  ✓ 1 task, 3 artifact handles, 2 dependency edges resolved
    2 artifact references checked against the registry
```

When the owner has changed the partitions since, the deploy fails and prints the line to paste:

```
artifact reference 'events' at stale.py:4 states partitions [date: Daily], but its owner's declaration in the
registry has [date: Daily, region]. Update the reference:
    events = artifacts.Artifact.ref("events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
```

A name the registry doesn't know yet (the owner hasn't deployed or published) is a note, not a failure.

| Module | Team | Shows |
|---|---|---|
| `ingest/events.py` | Data Platform | `raw_events` (`source=True`) and `events` handles; `clean` binds `raw` by identity and `date`/`region` with `get_partition_value`; `events.expect(region=[...])` (ladder level 5) |
| `ingest/seed.py` | Data Platform | lands `raw_events` from outside the graph with `produces_artifacts=True` + `artifacts.new(file, raw_events.at(...))` (levels 0-1) |
| `ml/features.py` | ML | `featurize` reads `events.all("region")` into a `list[DataFrame]`; its `date` parameter is bound implicitly (named like a dimension of `features`, no default) |
| `ml/train.py` | ML | `train` reads `features.window(date=TimeRange(days=30))`, publishes with `artifacts.new` + a model `Card` |
| `analytics/report.py` | Analytics | fan-in from another repo: `features` and `churn_model` as `Artifact.ref`s (no import of the ML modules), `features.window(7d)` plus `churn_model` by identity, `date` via `artifacts.partition("date")`; publishes from the declaration alone |
| `analytics/notify.py` | Analytics | a sink: `send_report` consumes `daily_report` (identity) and publishes nothing |
| `triggers/revalidate.py` | ML Quality | `flyte.OnArtifact(churn_model)`: a trigger bound to a handle, attachable before any version exists |
| `analytics/report.py` (again) | Analytics | `daily_report` declares `refresh=artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=TimeRange(days=1))`: deploying the module registers a nightly trigger that materializes yesterday's report (an inbound trigger on `daily_report`) |
| `triggers/weekly_review.py` | ML Quality | `churn_model.materialize_on(flyte.Cron("0 5 * * 1"), lag=...)` on an `Artifact.ref`: a reader keeps an artifact it doesn't own fresh; returns an environment to deploy |
| `apps/scoring.py` | ML | `churn-scoring` app with `consumes_artifacts={"model": churn_model}` |
| `apps/dashboard.py` | Analytics | `AppEndpoint("churn-scoring")` parameter (app -> app edge) plus a label-only `lineage.consumes=daily_report` |
| `legacy/legacy_clean.py`, `legacy/adapter.py` | Data Platform / ML | section 8: a wrapper task declares what a task you cannot edit produces |
| `notebook/explore.py` | Research | `Artifact.get(features, date=...)`, `Artifact.get("features", ...)`, `listall`, `Artifact.create` |
| `pull.py` | anyone | stage 4: `flyte.materialize` to plan, build one day, backfill a week, force a rebuild, run the sink |
| `factories/churn.py`, `factories/churn.yaml` | Platform | stages 5-6: the same graph as a production factory, in Python (reusing the teams' handles, calling their deployed tasks) and as a spec |

## What emerges

Deploying the modules one by one prints, after each deploy, what that deploy's declarations resolved to:

```
$ flyte deploy --root-dir . analytics/report.py env
  ✓ 1 task, 3 artifact handles, 2 dependency edges resolved
```

Across all of them the graph is:

```
raw_events ──clean──▶ events ──featurize (all region)──▶ features ──train (window 30d)──▶ churn_model
                                                          │                                 │
                                                          └──report (window 7d)──▶ daily_report ◀──┘ (identity)
churn_model ──▶ trigger:revalidate-on-new-model          churn_model ──▶ app:churn-scoring ──▶ app:churn-dashboard
daily_report ──▶ task:analytics-notify.send_report (sink)
daily_report ┄┄▶ app:churn-dashboard   (┄ label-only)
⏱ refresh-daily-report.keep_report_fresh ──▶ daily_report     ⏱ refresh-churn-model.for_weekly_review ──▶ churn_model
```

Five typed handles and five resolvable edges; the rest are entity edges the backend derives (apps,
triggers), a typed sink (`send_report`), or label-only edges the planner never crosses. The two refresh
policies (⏱) draw as inbound triggers pointing into their targets, so `flyte factory snapshot` turns them into
`fc.on(...)` and `send_report` into `fc.sink(...)`.

## Keeping artifacts fresh

The owner of an artifact declares when it should be materialized, on the handle:

```python
daily_report = artifacts.Artifact(
    "daily_report", type=File, partitions={"date": artifacts.Daily},
    refresh=artifacts.Refresh(flyte.Cron("0 2 * * *"), lag=artifacts.TimeRange(days=1), name="keep_report_fresh"),
)
```

Deploying a module that *produces* `daily_report` registers the trigger. Modules that only import the handle
don't. `refresh=` also takes a bare `flyte.Cron` / `flyte.FixedRate`, a source handle (materialize on each new
version of it, e.g. `artifacts.Refresh(raw_events, region="us")`), or a list of policies.

Anyone else, including a team that only has an `Artifact.ref`, keeps an artifact fresh with `materialize_on`
(`triggers/weekly_review.py`):

```python
churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})
for_weekly_review = churn_model.materialize_on(flyte.Cron("0 5 * * 1"), lag=artifacts.TimeRange(days=1))
```

Both compile to the same thing. A generated `refresh-<artifact>` environment holds one small task per policy,
whose body calls `flyte.materialize` for the partition the trigger names. You never write, name or import that
task: the platform rebuilds it from the policy at run time. Its image is the factory image (it needs
`flyteplugins-union`); `materialize_on(..., image=...)` overrides it.

Every version of every artifact carries a card (`cards.py`), shown on the artifact's **Artifact Card** tab:
the partition and what it is, headline numbers, then the schema, numeric ranges and first rows for data
(`raw_events`, `events`, `features`), feature means of churned vs retained users for `churn_model`, and the
report itself for `daily_report`, with which task built it from what. A task attaches one by publishing with
`artifacts.new(value, handle.at(..., card=await artifacts.Card.create_from.aio(content=html, ...)))`.

## Running it

Against a devbox (or any backend with the Union lineage service and `flyteplugins-union` installed):

```bash
cd examples/artifacts/emergent_lineage
./run_e2e.sh --config ~/path/to/config.yaml --date 2026-09-08
```

Options: `--date` is the partition to materialize; `--start` is the first day of raw data to seed (default:
30 days before `--date`, which the 30-day training window needs); `--project`, `--domain` and `--queue` are
passed through.

> **The script deploys live cron triggers.** Deploying `analytics/report.py` registers `daily_report`'s refresh
> policy, which runs at 02:00 every day and materializes the previous day's report until you deactivate it (for
> example in the console's Triggers view, or with
> `flyte update trigger keep-report-fresh refresh-daily-report.keep_report_fresh --deactivate`).
> `triggers/weekly_review.py` adds one for `churn_model` on Mondays at 05:00.

The script:

1. deploys each task module separately (`flyte deploy --root-dir . <file> <env>`), so every deploy only knows
   its own file plus the handles it imports;
2. seeds 31 days of `raw_events` for `us` and `eu` with `flyte run ingest/seed.py seed`;
3. runs `flyte materialize daily_report --partition date=2026-09-08 --plan`, which prints the instance DAG,
   every parameter of every task in the walk, and where each value comes from (registry, partition,
   default);
4. runs the real `flyte materialize daily_report --partition date=2026-09-08 --wait` (blocks until the run
   finishes), which builds `events`,
   `features` and `churn_model` back to `raw_events` and then the report (the task cache skips anything
   already fresh);
5. deploys the two apps. They come last because `churn-scoring` resolves `churn_model@latest` when it is
   deployed, and that fails before any version of `churn_model` exists;
6. runs `notebook/explore.py` to read the results back through the handles.

`--queue <name>` is passed through to `flyte materialize`, so every action of the walk runs on that queue.

### Local devbox notes

These apply only to the local union-devbox, not to a hosted tenant:

- **Use `--queue testcluster`.** The devbox's default queue caps each action at 700m CPU, which is too small for
  this walk.
- **App deploys from the CLI need an `x-user-subject` header.** The local console stamps that header on its
  requests, but the CLI doesn't send it, so `flyte deploy apps/scoring.py scoring` is rejected on the devbox.
  The SDK deliberately does not add it. `run_e2e.sh` handles it: when the config points at `localhost`, it puts
  `../devbox_shim` on `PYTHONPATH`, whose `sitecustomize.py` adds the header (`--no-devbox-shim` turns that off).
  By hand: `PYTHONPATH=../devbox_shim flyte deploy --root-dir . apps/scoring.py scoring`. Hosted tenants derive
  the subject from the authenticated identity and don't need it.

`--with-legacy` also deploys the section 8 adapter. It is a second producer of `events`, so it is off by
default to keep the walk to one producer per artifact. `--skip-materialize` stops after seeding (useful
without `flyteplugins-union`).

From Python, the same pull is `pull.py` (`plan`, `day`, `backfill`, `rebuild`, `send`):

```bash
python pull.py --config ~/path/to/config.yaml plan    # the instance DAG and cache probe, nothing launched
python pull.py --config ~/path/to/config.yaml backfill
```

## Pinning the graph as a factory

The emergent graph follows whatever each team deploys next. To freeze it, snapshot it into a factory you own
with every task version pinned (`flyteplugins-union`):

```bash
flyte factory snapshot daily_report -o factories/analytics.yaml   # declarative spec, deploys with no Python
flyte factory snapshot daily_report -o factories/analytics.py     # or a Python module
flyte factory validate factories/analytics.yaml
flyte factory deploy factories/analytics.yaml
flyte factory diff analytics                                      # what moved upstream since the snapshot
```

The output's extension picks the format; the YAML/JSON spec is the same one the console reads and writes.

`factories/churn.py` and `factories/churn.yaml` are a checked-in production factory for this graph (the
same graph two ways; both compile to identical factories). Each resolves every task to its latest deployed
version when it deploys and stays pinned to it:

```bash
flyte factory deploy factories/churn.py --dry-run     # validate against what is deployed, register nothing
flyte factory deploy factories/churn.yaml
flyte factory materialize churn send_report --partition date=2026-09-08 --wait
```

Set `CHURN_TRAIN_QUEUE` to run the training step on its own queue (`testcluster` on a local devbox).

## Checking it without a backend

`tests/flyte/artifacts/test_emergent_lineage_examples.py` imports every module here and runs the
deploy-time extraction and validation over all of them, asserting the node and edge set above. That is
the whole point: the graph is computed from separately authored modules, not written down anywhere.
