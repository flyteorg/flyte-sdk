# Weights & Biases Plugin

- Parent and child tasks can log metrics to the same run by setting `new_run=False` in the `wandb_init` decorator.
- Traces can be used to initialize a W&B run independently of the parent task.
- `wandb_run` is added to the Flyte task context and can be accessed via `flyte.ctx().wandb_run`.
- `wandb_run` is set to `None` for tasks that are not initialized with `wandb_init`.
- If a child task needs to use the same run as its parent, `wandb_config` should not be set, as it will overwrite the run name and tags (this must be ensured by the user).
- `wandb_config` can be used to pass configuration to tasks enclosed within the context manager and can also be provided via `with_runcontext`.
- When the context manager exits, the configuration falls back to the parent task's config.
- Arguments passed to `wandb_init` decorator are available only within the current task and are not propagated to child tasks (use `wandb_config` for child tasks).
- At most 20 runs can be active at a time when sharing the same run ID: https://docs.wandb.ai/models/sweeps/existing-project#3-launch-agents i.e. when `new_run=False`
- `wandb_sweep` can be used to initialize a sweep run, and the objective function can be a vanilla Python function decorated with `wandb_init`.

```python

```
