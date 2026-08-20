# Rerun & Recover

Re-launching an *existing* run. `flyte.rerun` / `flyte rerun` fetch the prior run's task spec
and inputs from the platform — no local code is involved, and nothing you edit locally is
picked up.

One verb, two behaviours:

| Call | What happens |
| ---- | ------------ |
| `flyte.rerun(run)` / `flyte rerun <run>` | A whole new run with the same inputs. Every action executes again, subject to global caching. |
| `flyte.rerun(run, recover=True)` / `flyte rerun <run> --recover` | A whole new run with the same inputs, but actions that already succeeded are reused as-is. Only what failed or never ran executes. |

Reused actions land in the `RECOVERED` phase — terminal and success-equivalent — so
`flyte get action <run>` tells you exactly what recovery skipped.

Recovery is durability against *intermittent* failures, not a way to patch a run. It replays
the source run's code and inputs as-is; the run environment (`-e KEY=VALUE`) is the only lever
you get. Replaying with new code or new inputs is `flyte fork`, reserved for
`pip install flyteplugins-union` — so `recover=True` combined with changed inputs raises.

`--force-rerun-action` is the escape hatch: it forces a named action to execute even though it
succeeded in the source run. Action names are deterministic hashes rather than positions, so
list them with `flyte get action <run>` and copy the one you want. A listed parent re-enqueues
its children, so list those too to force a whole subtree; unknown names are ignored.

## Examples

| File | What it shows |
| ---- | ------------- |
| `rerun_and_recover.py` | All four variants end to end — pure rerun, recover, recover with a forced action replay, and rerun with changed inputs — against a pipeline whose join step fails unless `FLAKY_OK=1` is in the run environment. Includes the equivalent `flyte` CLI command for each. |

```bash
uv run python examples/rerun/rerun_and_recover.py
```

Remote only: rerun and recover are not supported in local mode.
