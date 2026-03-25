"""Launcher for neuron_parallel_compile that uses elastic_launch.

neuron_parallel_compile wraps a command, setting NEURON_EXTRACT_GRAPHS_ONLY=1
to extract XLA graphs during a trial run.

This module ensures the trial run uses the same ``elastic_launch`` (fn mode)
as actual training, so XLA graphs match exactly and the cache is hit.

Usage (invoked by _run_neuron_parallel_compile):
    neuron_parallel_compile python -m flyteplugins.pytorch.compile_launcher \
        --ctx-file /path/to/ctx.pkl
"""

import argparse
import os
import pickle

from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from flyteplugins.pytorch.task import launcher_entrypoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctx-file",
        required=True,
        help="Path to pickled (tctx, fn_bytes, kwargs, launch_params) tuple",
    )
    args = parser.parse_args()

    with open(args.ctx_file, "rb") as f:
        tctx, fn_bytes, kwargs, launch_params = pickle.load(f)

    config = LaunchConfig(
        run_id=tctx.action.run_name,
        min_nodes=launch_params["min_nodes"],
        max_nodes=launch_params["max_nodes"],
        nproc_per_node=launch_params["nproc_per_node"],
        rdzv_backend=launch_params["rdzv_backend"],
        rdzv_configs=launch_params["rdzv_configs"],
        rdzv_endpoint=os.environ.get("PET_RDZV_ENDPOINT", "localhost:0"),
        max_restarts=launch_params["max_restarts"],
        monitor_interval=launch_params["monitor_interval"],
    )

    elastic_launch(config=config, entrypoint=launcher_entrypoint)(tctx, fn_bytes, kwargs)


if __name__ == "__main__":
    main()
