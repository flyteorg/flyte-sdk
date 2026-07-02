# Run-level `max_action_concurrency` test scenarios

- **`run_max_action_concurrency.py`** — baseline: 10 actions, cap 3. The happy path.
- **`high_churn.py`** — many *fast* actions... slots free faster than the bookkeeping refreshes
- **`wide_fanout.py`** — a larger number of actions. Does the cap hold, and stay cheap? ()
- **`restart_target.py`** — a *long* run you kill leasor under (or abort). Does the cap survive a crash/restart?
- **`multiple_runs.py`** — several capped runs at once, different limits. Does each get its own independent budget?
- **`nested_fanout_deadlock.py`** — fan-out inside fan-out. Shows the sharp edge: too small a cap deadlocks.
- **`reject_limit_one.py`** — cap of 1. Should be rejected outright (it always deadlocks).
