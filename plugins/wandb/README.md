# Weights & Biases Plugin

- Tasks decorated with `@wandb_init` or `@wandb_sweep` automatically get W&B links in the Flyte UI that point directly to the corresponding W&B runs or sweeps. Links retrieve project/entity from decorator parameters or context configuration (from `with_runcontext`).
- `@wandb_init` and `@wandb_sweep` must be the **outermost decorators** (applied after `@env.task`). For example:

  ```python
  @wandb_init
  @env.task
  def my_task():
      ...
  ```

- By default (`new_run="auto"`), child tasks automatically reuse their parent's W&B run if one exists, or create a new run if they're top-level tasks. You can override this with `new_run=True` (always create new) or `new_run=False` (always reuse parent).
- `@wandb_init` should be applied to tasks (not traces). Traces can access the parent task's W&B run via `get_wandb_run()`. `@wandb_init` can also be applied to regular Python functions for use in `wandb.agent()` sweep callbacks.
- The wandb run can be accessed via `get_wandb_run()`, which returns the run object or `None` if not within a `@wandb_init` decorated task.
- When using `new_run=False` or `new_run="auto"` (with a parent run), child tasks reuse the parent's run ID. Configuration from `wandb_config()` is merged with decorator parameters.
- `wandb_config` can be used to pass configuration to tasks enclosed within the context manager and can also be provided via `with_runcontext`.
- When the context manager exits, the configuration falls back to the parent task's config.
- Arguments passed to `wandb_init` decorator are available only within the current task and traces and are not propagated to child tasks (use `wandb_config` for child tasks).
- At most 20 sweep agents can be launched at a time: https://docs.wandb.ai/models/sweeps/existing-project#3-launch-agents.
- `@wandb_sweep` creates a W&B sweep and adds a sweep link to the decorated task. The sweep ID is available via `get_wandb_sweep_id()`. For the parent task that creates the sweep, the link points to the project's sweeps list page. For child tasks, the link points to the specific sweep (they inherit the `sweep_id` from the parent's context).
- The objective function passed to `wandb.agent()` should be a vanilla Python function decorated with `@wandb_init` to initialize the run. You can access the run with `wandb.run` since the Flyte context won't be available during the objective function call.
