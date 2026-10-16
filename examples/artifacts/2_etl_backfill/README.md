# 2. Partitioned ETL and backfills

A daily ETL pipeline where you never write a DAG or a backfill script. You declare what each task reads and
publishes; then you ask for the partitions you want, and the platform plans, reuses and builds.

**You'll learn:** source artifacts, partition dimensions, the three ways a task reads partitions (identity,
`all`, `window`), planning with `--plan`, backfilling a range, how reuse works, how to force a rebuild, what
happens when data is missing, and how to freeze the pipeline into a factory.

> Steps 3 onwards are experimental: they need the Union lineage service and `flyteplugins-union`
> (`pip install flyteplugins-union`). Steps 1 and 2 work on any Flyte 2 backend.

## The pipeline

```
raw_trips[date, city] ──clean──▶ trips[date, city] ──summarize (all cities)──▶ daily_stats[date]
                                                                                    │
                                                    weekly_stats[date] ◀──rollup (7-day window)──┘
```

| File | What it is |
|---|---|
| `pipeline.py` | The four artifacts and the three tasks that build them. Read this first. |
| `land.py` | Stands in for the export job that drops raw files: publishes `raw_trips[date, city]` |
| `backfill.py` | The walkthrough below, from Python (`flyte.materialize(...)`) |
| `run.sh` | The walkthrough below, as one script |

## Concepts in this example

**Artifact and partition.** An artifact is a named, versioned task output. `partitions={"date": artifacts.Daily,
"city": str}` gives every version a date and a city; `trips[2026-09-07, nyc]` is one partition. `Daily` makes
`date` a time dimension, so a range `2026-09-01..2026-09-07` expands into seven partitions.

**Source.** `raw_trips` is declared `source=True`: nothing in the graph builds it. The planner reads it from the
registry, and reports it missing instead of trying to build it.

**How a task reads partitions.** Declared once, on the input, in `consumes_artifacts=`:

| Declaration | Reads, to build partition `date=D` |
|---|---|
| `{"raw": raw_trips}` (identity) | the same partition: `raw_trips[D, city]` |
| `{"per_city": trips.all("city")}` | every city of that day, as a list: `trips[D, nyc]`, `trips[D, sf]` |
| `{"week": daily_stats.window(date=TimeRange(days=7))}` | the 7 days up to D: `daily_stats[D-6 .. D]` |

A parameter named like a dimension (`date`, `city`) receives the partition being built; no binding needed.

**Reuse is the cache.** A partition is reused when the same task version already built it from the same inputs.
Asking again is cheap; nobody decides what to skip.

**Expected values.** `trips.expect(city=["nyc", "sf"])` tells the planner which cities a day should have, so
`all("city")` knows what to wait for, and a missing city reads as "never arrived" rather than "not built".

## Walkthrough

Pass your config to every command (`flyte --config <config> ...`), and `--queue <name>` to the materialize
commands if your default queue is small. `./run.sh --config <config> [--queue <name>]` runs all of it.

### 1. Deploy, and land two weeks of raw data

```bash
flyte deploy --root-dir . pipeline.py env
flyte deploy --root-dir . land.py env
flyte run --root-dir . land.py land --start 2026-09-01 --end 2026-09-14
```

Deploy checks the declarations against the task signatures and prints what it resolved:

```
✓ 4 tasks, 4 artifact handles, 3 dependency edges resolved
  + refresh-daily-stats: nightly_stats (cron 0 3 * * *)
```

The fourth task is generated from `refresh=` on `daily_stats` (step 7). Open **Lineage** in the console to see the
graph drawn from these declarations. It can take up to a minute to appear after a deploy.

### 2. Plan one day

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-07 --plan
```

```
weekly_stats[2026-09-07]              trips-etl.rollup
  week         ← daily_stats[2026-09-01 .. 2026-09-07]  window(7d), 7 partitions
  date         = 2026-09-07                            partition

daily_stats[2026-09-07]               trips-etl.summarize
  per_city     ← trips[2026-09-07, nyc], [.., sf]      all("city"), 2 partitions
  + 6 more: daily_stats[2026-09-06], daily_stats[2026-09-05], ...

trips[2026-09-07, nyc]                trips-etl.clean
  raw          ← raw_trips[2026-09-07, nyc]            identity
  city         = nyc                                   enumerated by all("city")
  min_fare     = 2.5                                   default
  + 13 more: trips[2026-09-07, sf], trips[2026-09-06, nyc], ...

22 instances across 3 builds, 14 lookups. The cache decides what runs; nothing was launched.
```

One week of stats needs 7 days of `daily_stats`, which need 14 `trips` partitions, which read 14 raw files.
Every parameter of every call is accounted for, with where its value comes from.

### 3. Build it

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-07 --wait
```

The run page shows the plan as a live graph. When it finishes, every partition it built is in the registry,
and each one records the run that made it:

```
built: weekly_stats 1, daily_stats 7, trips 14     read: raw_trips 14
```

### 4. Backfill a week

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-08..2026-09-14 --concurrency 8 --wait
```

A backfill is the same request over a range. The days it shares with step 3 are reused, not rebuilt:

```
built: weekly_stats 7, daily_stats 7, trips 14     reused: daily_stats 6, trips 12
```

Run the same command again and everything is reused (`weekly_stats 7, daily_stats 13, trips 26` reused, nothing
built). Backfills are safe to repeat.

### 5. Change a parameter, or force a rebuild

A constant is part of the cache key. Overriding one rebuilds what depends on it, for the partitions you ask for:

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-07 --input trips-etl.clean.min_fare=5 --wait
# built: weekly_stats 1, daily_stats 7, trips 14
```

After fixing a bug the cache can't see (in a library, say), force a step to run again. Everything downstream of
it follows; everything upstream is reused:

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-14 --rebuild trips-etl.summarize --wait
# built: weekly_stats 1, daily_stats 7     reused: trips 14
```

### 6. Missing data

Nothing has landed after 2026-09-14. Ask for 2026-09-20 and the plan says so before anything runs:

```bash
flyte materialize artifact weekly_stats --partition date=2026-09-20 --plan
```

```
! Cannot plan weekly_stats. 12 source partitions are not in the registry.

  raw_trips: 2026-09-15..2026-09-20 × {nyc, sf}

  Publish them.
  If they will land while the materialization runs, pass --no-source-check.
```

Land them (`flyte run --root-dir . land.py land --start 2026-09-15 --end 2026-09-20`) and ask again.

### 7. Keep it fresh

`daily_stats` declares a refresh policy:

```python
refresh=artifacts.Refresh(flyte.Cron("0 3 * * *"), lag=artifacts.TimeRange(days=1), name="nightly_stats")
```

Deploying `pipeline.py` registered it as a trigger: every night at 03:00 it materializes yesterday's
`daily_stats`, and whatever that needs upstream. There's no scheduler or DAG to keep in sync with the code.
Deactivate it when you're done experimenting:

```bash
flyte update trigger nightly-stats refresh-daily-stats.nightly_stats --deactivate
```

### 8. Freeze it for production (optional)

The graph above follows whatever is deployed last. To pin it, snapshot it into a factory: one reviewed file with
every task pinned to its deployed version and the nightly trigger included.

```bash
flyte factory snapshot weekly_stats -o trips_factory.yaml     # or .py
flyte factory validate trips_factory.yaml
flyte factory deploy trips_factory.yaml
```

```yaml
  builds:
    - task: trips-etl.clean@e8cf7ddfa5046c8c3fbef5f85b837a43
      outputs: [trips]
      inputs:
        raw: {artifact: raw_trips}
        date: {partition: date}
        city: {partition: city}
    ...
  triggers:
    - name: nightly-stats
      targets: [daily_stats]
      schedule: {cron: 0 3 * * *}
      lag: 1d
```

`../4_multi_team/` covers factories in depth.

## From Python

Every step has a Python form, in `backfill.py`:

```python
flyte.materialize(weekly_stats, date=datetime(2026, 9, 7), plan_only=True)
flyte.materialize(weekly_stats, date=artifacts.TimeRange(datetime(2026, 9, 8), datetime(2026, 9, 14)), concurrency=8)
flyte.materialize(weekly_stats, date=datetime(2026, 9, 7), inputs={"trips-etl.clean.min_fare": 5.0})
flyte.materialize(weekly_stats, date=datetime(2026, 9, 14), rebuild=["trips-etl.summarize"])
```

```bash
python backfill.py --config <config> plan    # or day, backfill, again, min-fare, missing
```

## Try it yourself

- Add a third city: land it with `flyte run --root-dir . land.py land_trips --day 2026-09-07 --city la`, add it to
  `CITIES`, redeploy, and plan 2026-09-07. Which partitions are rebuilt, and why?
- Change `rollup` to a 3-day window. How many `daily_stats` partitions does one `weekly_stats` read now?
- Ask for `trips` directly: `flyte materialize artifact trips --partition date=2026-09-03 --partition city=sf --plan`.
  Any artifact in the graph is a target.

**Next:** [`../3_train_and_serve/`](../3_train_and_serve/) trains a model whenever new data is published, and
serves it.
