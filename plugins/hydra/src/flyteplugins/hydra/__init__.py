"""flyteplugins-hydra — Hydra launcher plugin for Flyte.

Provides three entry points for running Flyte tasks via Hydra:

1. **``@hydra.main`` + ``--multirun``** (standard Hydra CLI pattern):

   .. code-block:: bash

       python train.py hydra/launcher=flyte hydra.launcher.mode=remote
       python train.py --multirun hydra/launcher=flyte hydra.launcher.mode=remote \\
           optimizer.lr=0.001,0.01,0.1

2. **``flyte hydra run``** (Flyte CLI extension, no ``@hydra.main`` required):

   .. code-block:: bash

       flyte hydra run --config-path conf --config-name training --mode remote \\
           train.py pipeline --cfg optimizer.lr=0.01

3. **``hydra_run`` / ``hydra_sweep``** (Python SDK):

   .. code-block:: python

       from flyteplugins.hydra import hydra_run, hydra_sweep

       hydra_run(pipeline, config_path="conf", config_name="training",
                 overrides=["optimizer.lr=0.01"], mode="remote")

       runs = hydra_sweep(pipeline, config_path="conf", config_name="training",
                          overrides=["optimizer.lr=0.001,0.01,0.1"], mode="remote")
"""

from flyteplugins.hydra._run import apply_task_env, hydra_run, hydra_sweep

__all__ = ["apply_task_env", "hydra_run", "hydra_sweep"]
