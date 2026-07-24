


Record of things we tested for queue draining:


## Basic

**Drain an empty, idle queue. This was one of the bugs.**

```
flyte -c ~/.flyte/dogfood.staging.yaml create queue drain-test-a1 --run-concurrency 10 --action-concurrency 20
flyte -vv -c ~/.flyte/dogfood.staging.yaml update queue drain-test-a1 --drain
flyte -vv -c ~/.flyte/dogfood.staging.yaml update queue drain-test-a1 --activate
```


---

**Drain a queue mid-workload**

Make sure it stays DRAINING until last action completes and finalizes → DRAINED.
Watch queue_finalizes go 0 last.

```
flyte -c ~/.flyte/dogfood.staging.yaml run -p flytesnacks -d development network_issues/hello_queue.py parent
flyte -vv -c ~/.flyte/dogfood.staging.yaml update queue drain-test-a1 --drain
```

---

**Enqueue to DRAINING and to DRAINED**

In a drained state
```
flyte -c ~/.flyte/dogfood.staging.yaml run -p flytesnacks -d development network_issues/hello_queue.py child --i 2
```
https://dogfood.cloud-staging.union.ai/v2/domain/development/project/flytesnacks/runs/uhggbqc946m8h4bt2hws
Run is created, but is insta-aborted.  This is okay.  I don't expect an immediate run creation failure necessarily.

There were what claude called "zombie" leases being reported from dogfood-2... set to draining and confirmed that it
doesn't hit drained even after a long time (1 hour).
Zombie leases were really old runs that had GPU requirements that were unsatisfiable and were never terminated - lease worker had been trying to execute these unsuccessfully for weeks now.

After these runs were aborted, dogfood-2 reached drained correctly.

## More Racy scenarios

**TTL Finalize respected**

First reduce the finalize ttl from 4 weeks to 5 mins.
Create a queue
```
flyte -c ~/.flyte/dogfood.staging.yaml create queue drain-test-b4 --run-concurrency 10 --action-concurrency 20 --cluster dogfood-3
```
Ensure nothing's running on dogfood-3. Run a long running run. Kill the cluster's leaseworker, then abort the in-flight
run so a finalize gets created sticky to the dead worker. Then set the queue to drain.

Then restart leasor.

Then let the finalize TTL fire → drops logged, finalize_ttl_drops ticks, drain completes.

---

**Boot leasor while cluster service down**

First create a bunch of long running actions/leases. Then take down clusters service, and restart leasor.
Queue CRUD returns nothing at boot time, every queue with leases becomes a placeholder.
During the placeholder period, run-actions park (skip metric), finalizes still dispatch (they ignore queue state — part 2 PR),
releases land, but new enqueues should be rejected.

Then bring CRUD back → placeholders replaced in place, counters intact, parked work resumes.
This exercises the entire placeholder machinery.

```
flyte -c ~/.flyte/dogfood.staging.yaml run -p flytesnacks -d development network_issues/queue_with_sleepers.py parent
```

All the sleeps are enqueued at start.  So the 100 sleeps are pre-admitted work.
If instead an enqueue happened during the placeholder period, the entire run would fail. This is defensible because
with the cluster service down, leasor doesn't know the difference between an queue that never existed or cluster
service is just down. So retrying is weird.
You could argue that leasor should continue to admit on queues for which it already has leases for, but this weak, and
we don't want to end up being tempted for leasor to save cluster service state (i.e. what was some queue's
action concurrency so leasor knows how many to dispatch).

If cluster service goes down while leasor is running, it's asymmetric, which I also think is okay. Leasor will just
use stale configuration until cluster service comes back up.

---

Additional scenario ideas not yet tested:

Drain/activate flapping under load. Tight loop: --drain / activate
  every 2-5s, 100+ iterations, while a steady workload runs. This attacks
  the CRUD state-machine guard and the one-way mirror. Assert: never
  DRAINED while non-idle; result=rejected completions appear (that's the
  guard working); final state matches the last CLI call; queue never
  wedges.
  2. Drain racing the last terminal. The dangerous window is
  RunAction-delete (releases depth) → finalize outstanding, where only
  the Finalizes count — which trails by ~1s — holds the queue open.
  Script: single-action runs, issue the drain right as the action reports
  terminal, repeat 200+ times. A premature DRAINED here means the
  two-tick confirm is insufficient. This is the sharpest knife in the
  plan.
  3. Drain with retry-backoff load. Tasks that fail with minutes-long
  backoff: queue shows 0 running, looks quiet, but owes work (Unassigned
  lease, depth held since PR 3 releases only on delete). Assert it stays
  DRAINING through the backoff and only drains after retries resolve.
 

  1. Leasor pod kill every ~2m during sustained load + active drains.
  Each boot re-seeds counters from a store scan and rebuilds the two-tick
  state. Assert after each settle: drift zero, drains resume and
  complete, no double-report weirdness.
  
  3. Kill cluster service mid-drain. DrainCompleter gets errors →
  result=error, retries by re-evaluation, DRAINED lands after recovery.
  No stored state to corrupt — verify nothing needed manual kicking.
  4. Leaseworker kill storm during drain. Mass expiry → finalizes recycle
  Sent→Unassigned with generation bumps — exactly the incarnations the
  TTL fence must respect. With TTL=5m and repeated storms,
  recycled-but-live finalizes must survive; only truly dead ones drop.
