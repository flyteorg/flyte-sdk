# Queue examples

These examples demonstrate how to route tasks to named queues and how queue
configuration shapes the behavior of bursts of work. Each example targets one
of the queues declared in the cluster config; the examples assume the
following queues exist with these limits:

| Queue           | MaxActionConcurrency | MaxRunConcurrency | MaxDepth | Used by                    |
| --------------- | -------------------- | ----------------- | -------- | -------------------------- |
| `serial-1`      | 1                    | —                 | —        | `serial_one.py`            |
| `small-3`       | 3                    | —                 | —        | `small_concurrency.py`     |
| `depth-limited` | —                    | —                 | 5        | `depth_backpressure.py`    |
| `bulk-a`        | 2                    | —                 | —        | `multi_queue.py`           |
| `bulk-b`        | 4                    | —                 | —        | `multi_queue.py`           |
| `runs-1`        | —                    | 1                 | —        | `serial_runs.py`           |

Targeting a queue from your code is one parameter on the task decorator:

```python
env = flyte.TaskEnvironment(name="my_env")

@env.task(queue="serial-1")
async def my_task(): ...
```

You can also set `queue=` on `TaskEnvironment` to default all tasks in that
env to a queue, then override per task. In these examples the queue is set
on the **child step tasks** while the entry-point `main` task stays on the
default routing — that way the root run goes through your default cluster
pool and only the workload that should be capped is pinned to a queue.

## Running the examples

All four examples are runnable with `flyte run`:

```bash
flyte run examples/queues/serial_one.py main
flyte run examples/queues/small_concurrency.py main
flyte run examples/queues/depth_backpressure.py main
flyte run examples/queues/multi_queue.py main
```

Each example prints `START` and `END` lines with timestamps on every step so
you can read the log and confirm the expected overlap pattern.

## What to expect

### `serial_one.py`
Submits 5 sleep tasks (4s each) to a queue with capacity 1. **No two `[serial-1]
step … START` lines should overlap.** Total wall time ≈ `count × sleep_seconds`
= 20s. If you see two steps running concurrently, the queue cap is not being
honored.

### `small_concurrency.py`
Submits 10 sleep tasks (4s each) to a queue with capacity 3. **At any point in
time, at most three `[small-3] step …` ranges (START → END) overlap.** Total
wall time ≈ `ceil(count / 3) × sleep_seconds` = 16s.

### `depth_backpressure.py`
Submits 8 sleep tasks (8s each) to a queue that holds at most 5 in flight.
Each submission either succeeds and runs the step, or fails immediately with a
`RESOURCE_EXHAUSTED` error reported in the step's return string. **The total
number of *in-flight* tasks never exceeds 5; the rejected ones return
verbatim error messages instead of executing.** The exact split between
accepted and rejected is timing-dependent: if one task finishes before the
next submission attempt, the next one squeezes in.

This is the right pattern for a producer that needs to slow down when the
downstream queue is full — handle the rejection in the calling task instead of
retrying forever.

### `multi_queue.py`
Submits 8 sleep tasks to `bulk-a` (cap 2) **and** 8 sleep tasks to `bulk-b`
(cap 4) at the same time. The two queues are independent: work on `bulk-b`
should finish in roughly half the wall time of `bulk-a`, even though both
queues are running on the same workers and against the same dispatcher. **At
no point should more than 2 `[bulk-a]` steps or more than 4 `[bulk-b]` steps
overlap; cross-queue overlap is unrestricted.**

### `serial_runs.py`
Targets `runs-1`, which caps concurrent *runs* (root actions / workflow
executions) at 1 but does not cap individual child actions. The script's
`__main__` submits `NUM_RUNS` runs (default 3) concurrently via
`flyte.run.aio` and waits for all of them:

```bash
python examples/queues/serial_runs.py             # 3 runs, 50 children each
NUM_RUNS=5 FAN_OUT=100 python examples/queues/serial_runs.py
```

**All N runs are submitted within a few hundred ms, but only one's `main
START` lands at a time** — the second run's `main START` happens after
the first run's `main END`, and so on. Inside one run, all `FAN_OUT`
children fan out in parallel. This is the right pattern for a job that
internally parallelizes well but must not overlap with itself (training
runs writing a shared checkpoint, batch jobs that mutate global state,
etc.).

You can also submit a single run via the standard `flyte run` CLI:
`flyte run examples/queues/serial_runs.py main`.

## Choosing queue parameters in your own code

The three knobs are:

- **`maxActionConcurrency`** — strict cap on how many tasks pinned to this
  queue run at the same time. Use small values (1, 2, …) when you need to
  serialize access to a finite external resource (a single GPU, a non-thread-
  safe library, a rate-limited API).

- **`maxRunConcurrency`** — strict cap on how many *runs* (root actions /
  workflow executions) pinned to this queue are in flight at once. Children
  of an active run are uncapped (unless `maxActionConcurrency` is also set).
  Use this when a workflow internally parallelizes well but must not overlap
  with itself.

- **`maxDepth`** — total number of in-flight + waiting tasks the queue will
  admit. Defaults to unbounded. Set this when the producer can flood the
  system faster than tasks complete; the resulting `RESOURCE_EXHAUSTED` is
  your back-channel signal to slow down.

There's no penalty for omitting either limit — an unbounded queue is fine when
your workload's natural concurrency is already low. Set a limit when you have
a specific number to enforce, not preemptively.
