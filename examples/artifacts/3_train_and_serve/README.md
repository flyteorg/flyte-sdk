# 3. Train and serve on new data

A model that retrains whenever a new day of labeled data is published, and an app that serves it. There is no
training schedule to maintain and no deploy script. The model declares what data it trains on, the app declares
which model it serves, and publishing data sets the rest in motion.

**You'll learn:** retraining on a trailing window, `refresh=` triggered by new data, quality gates, serving an app
from an artifact with `consumes_artifacts=`, `flyte materialize app`, why an app never rolls back by accident and
how to roll it back on purpose, and a factory that retrains and redeploys in one versioned unit.

> Experimental beyond deploying: needs the Union lineage service and `flyteplugins-union`.

## The graph

```
reviews[date] ──train (7-day window)──▶ sentiment_model[date] ──serve──▶ app:sentiment-api
     │                                         ▲
     └── on each new version ── refresh ───────┘
```

| File | What it is |
|---|---|
| `data.py` | `reviews`, a source artifact, and `land_reviews`, which stands in for the labeling job that publishes it |
| `train.py` | `sentiment_model`, trained on the 7 days of reviews up to its date; retrains on new data; a quality gate |
| `serve.py` | `sentiment-api`, an app whose `model` parameter is bound to `sentiment_model` |
| `served.py` | Prints which model version the deployed app serves |
| `factory.py` | The production version: new data → retrain → redeploy, pinned and versioned |
| `run.sh` | The walkthrough below, as one script |

## Concepts in this example

**A trailing window.** `reviews.window(date=TimeRange(days=7))` makes `train` for 2026-09-14 read
`reviews[2026-09-08 .. 2026-09-14]`. The newest day is held out to measure accuracy.

**Refresh on a source.** `refresh=artifacts.Refresh(reviews)` on `sentiment_model` means: each new version of
`reviews[D]` materializes `sentiment_model[D]`. Deploying `train.py` registers that trigger
(`+ refresh-sentiment-model: retrain_on_new_reviews (on new reviews)`). Compare `Refresh(flyte.Cron(...))` in
`../2_etl_backfill/`, which runs on a schedule instead.

**A quality gate is just a failure.** `train` raises when accuracy is below `min_accuracy`. A failed build
publishes nothing, so nothing downstream reads it and the app keeps serving the last good model.

**An app bound to an artifact.** `consumes_artifacts={"model": sentiment_model}` on the `AppEnvironment` makes the
app a node in the graph. `flyte materialize app` builds what the app reads, then redeploys it pinned to that
exact version.

**No rollbacks by accident.** An app serves one partition. Materializing an older date (a backfill, a late
trigger) builds that model but leaves the app on the newer one, unless you pin a version on purpose.

## Walkthrough

Pass `--config <config>` to every command, and `--queue <name>` to materialize commands if your default
queue is small. `./run.sh --config <config> [--queue <name>]` runs all of it.

### 1. Publish two weeks of data, then deploy the model

```bash
flyte deploy --root-dir . data.py env
flyte run --root-dir . data.py land --start 2026-09-01 --end 2026-09-14
flyte deploy --root-dir . train.py env
```

```
✓ 2 tasks, 2 artifact handles, 1 dependency edge resolved
  + refresh-sentiment-model: retrain_on_new_reviews (on new reviews)
```

Land the history before deploying `train.py`. Its trigger fires on every new day of data, and the first
days have no full week behind them yet.

### 2. Train the first model, then deploy the app

```bash
flyte materialize artifact sentiment_model --partition date=2026-09-14 --wait
flyte deploy --root-dir . serve.py api
python served.py --config <config>       # model sentiment_model 5e7763...
```

The app resolves the newest `sentiment_model` when it starts, so deploy it after a model exists.
(Deploying it first fails with `Failed to materialize artifact sentiment_model@latest`.) The lineage graph
takes up to a minute to pick up a deploy, so the model has to be visible there before you materialize it.

### 3. New data arrives

```bash
flyte run --root-dir . data.py land_reviews --day 2026-09-15
```

Publishing `reviews[2026-09-15]` fires the refresh trigger, which trains `sentiment_model[2026-09-15]` from
`reviews[2026-09-09 .. 2026-09-15]`. Watch it under **Runs**, or on the `sentiment_model` node in **Lineage**.

### 4. Serve the new model

```bash
flyte materialize app sentiment-api --partition date=2026-09-15 --wait
python served.py --config <config>       # model sentiment_model <the 2026-09-15 version>
```

The model for that date is already built, so it's reused; the run ends by redeploying `sentiment-api` pinned to it.
Ask for a date with no model yet and the same command trains it first.

Ask for an older date now, and the model for it is built but the app isn't moved back:

```
sentiment-api 2026-09-14 skipped | not deployed: the app serves sentiment_model@... (date=2026-09-15),
newer than 2026-09-14; to roll back, pin what it serves with --version
```

### 5. The quality gate

Raise the bar above what the model achieves:

```bash
flyte materialize app sentiment-api --partition date=2026-09-16 --input sentiment-train.train.min_accuracy=0.95 --wait
```

```
! run ... failed
0 built, 0 reused, 1 failed, 1 blocked. sentiment_model[2026-09-16]: RuntimeUserError: accuracy 0.84 < 0.95: not publishing this model
```

`served.py` still prints the previous model. Rerun without `--input` and the 2026-09-16 model is trained and served.

### 6. Roll back

To serve an earlier model on purpose, pin it with `--version`. A pinned model is read as-is: nothing upstream is
read or built.

```bash
flyte get artifact sentiment_model                     # the versions, newest first
flyte materialize app sentiment-api --partition date=2026-09-15 --version sentiment_model=<version> --wait
```

```
sentiment_model 2026-09-15 pinned  | pinned to version <version>
sentiment-api   2026-09-15 built   | deployed: ... serves model=sentiment_model@<version>
```

### 7. Production: one factory for train and serve

Steps 3 and 4 are two separate actions. `factory.py` makes them one, triggered by the data:

```python
source = fc.source(reviews)
model = fc.build(sentiment_model).using(Task.get("sentiment-train.train", auto_version="latest"),
                                        history=source.window(date=fc.TimeRange(days=7)))
api = fc.serve("sentiment-api").using("sentiment-api", model=model)
sentiment = fc.Factory("sentiment", api, triggers=[fc.on(source, name="retrain-and-serve")])
```

```bash
flyte factory deploy factory.py --dry-run
flyte factory deploy factory.py
flyte factory materialize sentiment sentiment-api --partition date=2026-09-16 --wait
```

```
sentiment_model 2026-09-16 built | built and published
sentiment-api   2026-09-16 built | deployed: ... serves model=sentiment_model@63095a...
```

From now on, each new `reviews` partition trains the model and redeploys the app. Task versions are pinned when
the factory deploys, so a teammate's change to `train.py` reaches production only when you redeploy the
factory (`flyte factory diff sentiment` lists what changed). When the factory owns retraining, remove `refresh=`
from `sentiment_model`, so new data doesn't train twice.

## Try it yourself

- Change the window to 14 days. What does the plan for 2026-09-14 say now, and why?
- Land a day with `--rows 20`. Does the model still pass the gate?
- Materialize the app for a range: `--partition date=2026-09-15..2026-09-17`. Which models are built, and which
  one is served?

**Next:** [`../4_multi_team/`](../4_multi_team/): the same ideas across four teams that don't share code, and
factories in depth.
