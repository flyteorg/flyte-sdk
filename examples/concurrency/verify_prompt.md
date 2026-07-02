# Verify run-level `max_action_concurrency`

Goal: confirm a run never runs more than its `max_action_concurrency` actions at
once, and the leasor tracker drains afterward.

## The invariant

> A limited run's active-action count never exceeds its limit, and
> `run_concurrency_tracked_runs` returns to **0** after the run ends.

## Watch (leave running during any scenario)

```bash
# in another shell: kubectl port-forward <leasor-pod> 10254:10254
LIMIT=3
while true; do
  curl -s localhost:10254/debug/leasor/queues \
    | jq --argjson L $LIMIT '.run_concurrency // {} | to_entries
        | map(select(.value > $L)) as $over
        | "\(now|floor) tracked=\(length) " +
          (if ($over|length)>0 then "VIOLATION \($over)" else "ok" end)'
  sleep 1
done
```

Also: gauge `run_concurrency_tracked_runs` → 0 after drain; counter
`schedule_skip_total{reason="run_at_action_concurrency"}` > 0 while capping; leasor
`Info` logs `RunManager.CreateRun … limited at N` / `OnRootTerminal … released`.

**Post-condition for every scenario: gauge == 0 and `run_concurrency` empty after the run.**

## ⚠️ Prerequisite

Worker/queue capacity must exceed the per-run cap (and, for `multiple_runs.py`, the
**sum** of caps + one root each), or worker saturation masks the per-run cap.

## Scenarios

| File | Stresses | Pass |
|---|---|---|
| `run_max_action_concurrency.py` | baseline cap | waves of ≤ cap; gauge→0 |
| `high_churn.py` | reserve↔1s-snapshot turnover (short actions) | never > cap; throughput ≈ cap/sec; gauge→0 |
| `wide_fanout.py` | capActionsPerRun + byRun scan at scale | never > cap; scheduler/snapshot timers flat; gauge→0 |
| `restart_target.py` | boot-seed rehydration (kill leasor) + abort | no overshoot across restart/abort; gauge→0 |
| `multiple_runs.py` | independent budgets, COW churn | each ≤ its own cap; gauge==#runs then 0 |
| `nested_fanout_deadlock.py` | known deadlock at cap ≤ nesting depth | characterize: cap=2 wedges, cap=3 completes |
| `reject_limit_one.py` | CreateRun rejects `=1` | submission raises InvalidArgument |

## Restart chaos (the path unique to this PR)

While `restart_target.py` is at cap, kill **leasor only** — this exercises RunSpec
reload + the boot `CountActiveByRun` seed (must run before the scheduler starts):

```bash
kubectl delete pod -l app=leasor                                   # single
while true; do kubectl delete pod -l app=leasor; sleep 15; done    # restart loop
```

Fail signals: active jumps past cap after restart (RunSpec didn't persist / boot
seed skipped), or over-dispatch on the first post-restart tick (startup grace
window). Also kill `leaseworker` / `actions` independently. The same run is the
**abort** target — abort mid-flight and confirm the cascade completes + gauge drains.

## Staging only (not on single-cluster devbox)

- Cross-queue global cap: a run's actions across ≥2 clusters stay ≤ cap *in total*.
- min-wins: a queue-level `MaxActionConcurrency` smaller than the run cap binds instead.
- Real scale / multi-shard.
