"""Which artifact version a deployed app serves, read from the app's recorded inputs.

`flyte materialize app churn-scoring` ends by redeploying the app with each artifact-bound parameter pinned to
the version the run built, and records that version as the app's input. This prints it:

    python apps/served.py --config <flyte config> churn-scoring
    model churn_model 9f2c...

An app deployed with `flyte deploy` resolves `churn_model@latest` when it starts and records the query, not a
version, so it prints `model (resolved at activation)` until a materialization has served it.
"""

from __future__ import annotations

import argparse

import flyte
from flyte.remote import App


def served(name: str) -> dict:
    """parameter -> "<artifact> <version>" for each input pinned to an artifact version, else
    "(resolved at activation)"."""
    out = {}
    for item in App.get(name).pb2.spec.inputs.items:
        if item.HasField("artifact_id") and item.artifact_id.version:
            out[item.name] = f"{item.artifact_id.key.name} {item.artifact_id.version}"
        else:
            out[item.name] = "(resolved at activation)"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="flyte config file (default: the usual lookup)")
    parser.add_argument("app", nargs="?", default="churn-scoring")
    args = parser.parse_args()
    flyte.init_from_config(args.config)
    for param, value in served(args.app).items():
        print(param, value)
