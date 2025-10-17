from functools import cache

from .constants import _RUNTIME_CONFIG_FILE


@cache
def get_input(name: str) -> str:
    """Get inputs for application or endpoint."""
    import json
    import os

    config_file = os.getenv(_RUNTIME_CONFIG_FILE)

    with open(config_file, "r") as f:
        inputs = json.load(f)["inputs"]

    return inputs[name]
