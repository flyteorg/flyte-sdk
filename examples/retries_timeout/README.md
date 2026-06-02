# Retries & Timeouts

These examples cover how Flyte tasks handle retries (`RetryStrategy`) and the
three timeout bounds on `flyte.Timeout`:

| Bound             | Scope        | Anchored at                         | Enforced by     |
| ----------------- | ------------ | ----------------------------------- | --------------- |
| `max_runtime`     | per attempt  | first time plugin enters `Running`  | leaseworker     |
| `max_queued_time` | per attempt  | lease enqueued at the leasor        | leaseworker     |
| `deadline`        | all attempts | first time the action was enqueued  | leasor          |

`deadline` is absolute and overrides `retries`: once it fires, no further
attempts run. The other two are per-attempt and respect `retries`.

## Examples

| File                            | What it shows                                                        |
| ------------------------------- | -------------------------------------------------------------------- |
| `backoff.py`                    | Focused demo of exponential backoff between **user** retries (`base`/`factor`/`cap`). Run locally to watch the inter-attempt gaps grow `1s → 2s → 4s → 5s` (capped). |
| `retry_and_timeout.py`          | Full lap of all controls in one task: retries with backoff, all three timeout bounds, `NonRecoverableError` short-circuit. Run locally with `flyte.init()` to observe `Backoff` delays. |
| `non_recoverable_error.py`      | `NonRecoverableError` terminates a task on the first attempt even when `retries=3`. |
| `max_runtime.py`                | Single `max_runtime` task that sleeps past its budget — TIMED_OUT, no retries. |
| `queued_timeout.py`             | Single `max_queued_time` task that requests a GPU the cluster doesn't have, so the pod stays Pending and the budget fires. |
| `deadline.py`                   | Single `deadline` task that sleeps past its absolute budget. |
| `retries_with_timeouts.py`      | Three tasks (one per timeout type) all declaring `retries`, so you can compare how the retry budget interacts with each bound. |

## Running

All timeout examples target a real cluster — the budgets need actual pod
scheduling / worker dispatch to fire. The recommended setup is the local
devbox (see the cloud repo's `devbox/` directory), with the actions service
enabled:

```bash
_U_USE_ACTIONS=1 flyte run examples/retries_timeout/<file>.py [<task>]
```

`backoff.py` and `retry_and_timeout.py` are the exceptions — they're designed
for local execution (`flyte.init()`) so the local controller can demonstrate
`Backoff` between retries inside a single Python process:

```bash
python examples/retries_timeout/backoff.py
```
