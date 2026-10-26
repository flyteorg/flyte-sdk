"""Stand-in for `deep_cluster.hardware`: the device count comes from `STUB_NUM_GPUS`."""

import os


def get_actual_num_gpus() -> int:
    return int(os.environ.get("STUB_NUM_GPUS", "4"))


def accelerator_kind() -> str:
    return "nvidia"
