# Weights & Biases Plugin

- Tasks decorated with `@wandb_init` or `@wandb_sweep` automatically get W&B links in the Flyte UI that point directly to the corresponding W&B runs or sweeps. Links retrieve project/entity from runtime context or decorator parameters.
- `@wandb_init` and `@wandb_sweep` must be the **outermost decorators** (applied after `@env.task`). For example:

  ```python
  @wandb_init
  @env.task
  def my_task():
      ...
  ```

- By default (`new_run="auto"`), child tasks automatically reuse their parent's W&B run if one exists, or create a new run if they're top-level tasks. You can override this with `new_run=True` (always create new) or `new_run=False` (always reuse parent).
- `@wandb_init` can only be applied to Flyte tasks. Traces cannot use the decorator but can access the parent task's W&B run via `flyte.ctx().wandb_run`.
- `wandb_run` is added to the Flyte task context and can be accessed via `flyte.ctx().wandb_run`.
- `wandb_run` is set to `None` for tasks that are not initialized with `wandb_init`.
- If a child task needs to use the same run as its parent, `wandb_config` should not be set, as it will overwrite the run name and tags (this must be ensured by the user).
- `wandb_config` can be used to pass configuration to tasks enclosed within the context manager and can also be provided via `with_runcontext`.
- When the context manager exits, the configuration falls back to the parent task's config.
- Arguments passed to `wandb_init` decorator are available only within the current task and traces and are not propagated to child tasks (use `wandb_config` for child tasks).
- At most 20 runs can be active at a time when sharing the same run ID: https://docs.wandb.ai/models/sweeps/existing-project#3-launch-agents i.e. when `new_run="auto"` or `new_run=False`.
- `wandb_sweep` can be used to initialize a sweep run, and the objective function needs to be a vanilla Python function which you can decorate with `@wandb_init` to initialize the run. You can access the run with `wandb.run` since flyte context won't be available during the objective function call.
