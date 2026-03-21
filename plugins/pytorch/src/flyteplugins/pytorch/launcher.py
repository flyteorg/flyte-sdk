"""Standalone launcher script for use with torchrun CLI.

This module is invoked as:
    torchrun --nproc_per_node=N -m flyteplugins.pytorch.launcher --ctx-file /path/to/ctx.pkl

It deserializes the task context and user function from the pickle file,
then runs the same entrypoint logic as elastic_launch would.
"""

import argparse
import pickle

from flyteplugins.pytorch.task import launcher_entrypoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-file", required=True, help="Path to pickled (tctx, fn_bytes, kwargs) tuple")
    args = parser.parse_args()

    with open(args.ctx_file, "rb") as f:
        tctx, fn_bytes, kwargs = pickle.load(f)

    launcher_entrypoint(tctx, fn_bytes, kwargs)


if __name__ == "__main__":
    main()
