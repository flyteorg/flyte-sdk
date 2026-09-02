# cluster_drain

Workloads for the leasor cluster-draining staging regimen
(runbook: https://claude.ai/code/artifact/c9a04c77-50f8-411a-b5c9-8a17a8a81019).

All tasks live in `drain_workloads.py` and take the queue from the command line. `A` below is the cluster being
drained; `Q` is a queue that routes to both `A` and `B` (wildcard or both pinned); `A-queue` is A's co-named queue.

| Step | Command |
|---|---|
| T0 / T1 placement probe, after every `--activate` | `flyte run --queue Q drain_workloads.py quick` |
| T1–T5 long runner | `flyte run --queue Q drain_workloads.py sleep_for --seconds 1800` |
| T6 leasor restart mid-drain | `flyte run --queue Q drain_workloads.py fan_out --n 50 --seconds 300` |
| T7 co-named queue, terminal fail | `flyte run --queue A-queue drain_workloads.py fan_out --n 3 --seconds 1800` |
| T7 parked action | `flyte run --queue A-queue drain_workloads.py sleep_for --seconds 60` after the drain call |

What to read in the task logs:

- `sleep_for` prints `alive Ns` with the pod hostname every 10s. After a force drain on a multi-cluster queue the
  same action reappears with a new hostname (attempt 2) — the old pod's lines stop.
- On the co-named queue nothing reappears; the run ends FAILED with `queue A-queue routes to no other cluster`.
- `fan_out` prints the per-host child count at the end; on T6 it should sum to `--n` with every child on B.

To land work on `A` specifically for T1–T5, use a queue that routes only to `A` (its co-named queue) for the
*first* placement but expect T2/T3 to then fail it terminally — or, simpler, pause B's leaseworker while submitting
so `Q` places on `A`, then resume it before the drain so retries have somewhere to go.
