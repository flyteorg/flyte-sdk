# Stress Examples

This directory contains ad hoc stress and failure-mode examples for Flyte and Union dogfood testing.

## Primary Entry Point

Use [sleep_fanout_harness_wrapper.sh](sleep_fanout_harness_wrapper.sh) for multi-run `core-sleep` fanout tests:

```bash
examples/stress/sleep_fanout_harness_wrapper.sh \
  --config ~/.flyte/config-dogfood.yaml \
  --total-runs 10 \
  --submit-concurrency 10 \
  --n-children 1000 \
  --sleep-duration 600 \
  --poll-interval 1 \
  --run-env _F_MAX_QPS=150 \
  --run-env _F_CTRL_WORKERS=20 \
  --run-env _F_P_CNC=1000
```

This wrapper:
- submits many top-level `sleep_fanout` runs through `flyte run`
- tracks aggregate child visibility and running counts
- prints parent-run counts (`p_live`, `p_run`) and child creation rate (`create_rps`, `rps/p`)

The underlying task definitions live in [sleep_fanout.py](sleep_fanout.py), and the local submit helper lives in [sleep_fanout_harness.py](sleep_fanout_harness.py).

## Key Files

- [sleep_fanout.py](sleep_fanout.py): `core-sleep` leaf task, parent fanout task, and swarm submit task definitions.
- [sleep_fanout_harness.py](sleep_fanout_harness.py): local async submit harness used by the wrapper.
- [runs_per_second.py](runs_per_second.py): launch-rate test helper.
- [fanout_concurrency.py](fanout_concurrency.py): simple fanout/concurrency experiment.
- [large_fanout.py](large_fanout.py): wide fanout example.
- [duplicate_action_id.py](duplicate_action_id.py): action-id collision / dedupe behavior probe.
- [crash_recovery_trace.py](crash_recovery_trace.py), [long_recovery.py](long_recovery.py), [fast_crasher.py](fast_crasher.py): controller and recovery failure scenarios.
- [cpu_gremlin.py](cpu_gremlin.py), [network_gremlin.py](network_gremlin.py): fault-injection style workload examples.
- [large_file_io.py](large_file_io.py), [large_dir_io.py](large_dir_io.py), [benchmark/large_io_comparison.py](benchmark/large_io_comparison.py): large I/O stress examples.
- [scale_test_same_image.py](scale_test_same_image.py), [scale_test_varied_images.py](scale_test_varied_images.py), [image_builds.py](image_builds.py): image build and scale tests.

## Notes

- `sleep_fanout` leaves use the `core-sleep` plugin, so the children run in leaseworker instead of creating task pods.
- Parent resource defaults for fanout are controlled in `sleep_fanout.py` via:
  - `FLYTE_STRESS_FANOUT_CPU_REQUEST`
  - `FLYTE_STRESS_FANOUT_CPU_LIMIT`
  - `FLYTE_STRESS_FANOUT_MEMORY_REQUEST`
  - `FLYTE_STRESS_FANOUT_MEMORY_LIMIT`
- Remote image contents come from the built wheel in `dist/`, not directly from local `src/`. If the wrapper warns that `src/flyte` is newer than the wheel, rebuild the wheel before relying on SDK changes in remote runs.
