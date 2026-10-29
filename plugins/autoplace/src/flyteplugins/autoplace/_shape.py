from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from flyte._resources import Resources


@dataclass(frozen=True)
class PodSet:
    """
    `count` pods of a single shape: what a capacity provider is asked to make room for.

    Quantities are Kubernetes strings (`"4"`, `"500m"`, `"16Gi"`) so the value travels unchanged
    from the SDK `Resources` into whatever request the provider files. The accelerator is kept as
    the SDK names it (`gpu_type="H100"`, `gpu_class="GPU"`): each provider maps that to its own
    node labels, and refuses in `validate` what it cannot provision.
    """

    count: int
    cpu: str
    memory: str
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    gpu_class: Optional[str] = None


def quantity(value: Union[int, float, str, Tuple, None], default: str) -> str:
    """
    Render one `Resources` field as a Kubernetes quantity, taking the request side of a
    `(request, limit)` tuple. Floats become millis so `0.5` CPU is `"500m"`, not `"0.5"`.
    """
    if value is None:
        return default
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, bool):
        raise TypeError("bool is not a resource quantity")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{round(value * 1000)}m"
    return str(value)


def podset_from_resources(resources: Resources, *, count: int = 1) -> PodSet:
    """
    Derive the pod set for `count` pods of `resources` (1 for a plain task, `replicas` for a
    clustered one). Cloud-neutral: the provider decides what it can do with the accelerator.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    device = resources.get_device()
    return PodSet(
        count=count,
        cpu=quantity(resources.cpu, "1"),
        memory=quantity(resources.memory, "1Gi"),
        gpu_count=device.quantity if device is not None else 0,
        gpu_type=device.device if device is not None else None,
        gpu_class=device.device_class if device is not None else None,
    )
