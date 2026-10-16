"""Which artifact version each parameter of a deployed app serves.

    python served.py --config <flyte config> sentiment-api
    model sentiment_model 4f1c...

An app that `flyte deploy` started resolves its artifacts when it starts and prints "(resolved at activation)"
until `flyte materialize app` (or a factory) pins a version.
"""

from __future__ import annotations

import argparse

import flyte
from flyte.remote import App


def served(name: str) -> dict:
    out = {}
    for item in App.get(name).pb2.spec.inputs.items:
        if item.HasField("artifact_id") and item.artifact_id.version:
            out[item.name] = f"{item.artifact_id.key.name} {item.artifact_id.version}"
        else:
            out[item.name] = "(resolved at activation)"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("app", nargs="?", default="sentiment-api")
    args = parser.parse_args()
    flyte.init_from_config(args.config)
    for param, value in served(args.app).items():
        print(param, value)
