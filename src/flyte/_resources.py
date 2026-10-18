import typing
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, get_args

import rich.repr

from flyte._pod import _PRIMARY_CONTAINER_DEFAULT_NAME

if TYPE_CHECKING:
    from kubernetes.client import V1PodSpec

PRIMARY_CONTAINER_DEFAULT_NAME = "primary"

SharedGPUType = Literal["T4 shared", "L4 shared", "A10G shared", "A10 shared", "L40S shared"]
"""
Shared GPU devices. A "<model> shared" device requests time-sliced (or MPS) capacity: a share of one
physical GPU on a node pool that advertises the shared accelerator for that model, so several tasks
can run on the same GPU at once. A plain "<model>" request never lands on such a pool, and a shared
device takes no partition.
"""

GPUType = Literal[
    "A2",
    "A10",
    "A10G",
    "A100",
    "A100_80G",
    "B200",
    "H100",
    "H200",
    "L4",
    "L40s",
    "T4",
    "V100",
    "RTX_PRO_4500",
    "RTX_PRO_6000",
    "GB10",
    SharedGPUType,
]
GPUQuantity = Literal[1, 2, 3, 4, 5, 6, 7, 8]
A100Parts = Literal["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]
"""
Partitions for NVIDIA A100 GPU.
"""

A100_80GBParts = Literal["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
"""
Partitions for NVIDIA A100_80GB GPU.
"""

H100Parts = Literal["1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
"""
Partitions for NVIDIA H100 GPU (80GB HBM3).
"""

H200Parts = Literal["1g.18gb", "1g.35gb", "2g.35gb", "3g.71gb", "4g.71gb", "7g.141gb"]
"""
Partitions for NVIDIA H200 GPU (141GB HBM3e).
"""

RTXPro6000Parts = Literal["1g.24gb", "2g.48gb", "4g.96gb"]
"""
Partitions for NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB GDDR7).

The +gfx and +me profiles are left out because the partition becomes a Kubernetes node label
value, and label values cannot contain "+". The -me profiles are left out until the backend can
parse them.
"""

TPUType = Literal["V5E", "V5P", "V6E"]
V5EParts = Literal["1x1", "2x2", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16"]
"""
Slices for Google Cloud TPU v5e.
"""

V5PParts = Literal[
    "2x2x1", "2x2x2", "2x4x4", "4x4x4", "4x4x8", "4x8x8", "8x8x8", "8x8x16", "8x16x16", "16x16x16", "16x16x24"
]
"""
Slices for Google Cloud TPU v5p.
"""

V6EParts = Literal["1x1", "2x2", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16"]
"""
Slices for Google Cloud TPU v6e.
"""

NeuronType = Literal["Inf1", "Inf2", "Trn1", "Trn1n", "Trn2", "Trn2u"]

AMD_GPUType = Literal["MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300X", "MI325X", "MI350X", "MI355X"]

HABANA_GAUDIType = Literal["Gaudi1"]

AppleSiliconType = Literal["AppleSilicon"]
"""
The Metal GPU of an Apple Silicon (M-series) Mac, on a devbox running on that Mac. A workload that requests
`gpu="AppleSilicon"` runs natively on the Mac, so its code can use Metal directly (`torch.device("mps")`, `mlx`);
this needs a devbox started with `flyte start devbox --apple-silicon`. Everywhere else it fails with
`flyte.errors.AppleSiliconNotAvailableError`.
"""

Accelerators = Literal[
    # A2
    "A2:1",
    # A10
    "A10:1",
    "A10:2",
    "A10:3",
    "A10:4",
    "A10:5",
    "A10:6",
    "A10:7",
    "A10:8",
    # A10G
    "A10G:1",
    "A10G:2",
    "A10G:3",
    "A10G:4",
    "A10G:5",
    "A10G:6",
    "A10G:7",
    "A10G:8",
    # A100
    "A100:1",
    "A100:2",
    "A100:3",
    "A100:4",
    "A100:5",
    "A100:6",
    "A100:7",
    "A100:8",
    # A100_80G
    "A100_80G:1",
    "A100_80G:2",
    "A100_80G:3",
    "A100_80G:4",
    "A100_80G:5",
    "A100_80G:6",
    "A100_80G:7",
    "A100_80G:8",
    # B200
    "B200:1",
    "B200:2",
    "B200:3",
    "B200:4",
    "B200:5",
    "B200:6",
    "B200:7",
    "B200:8",
    # H100
    "H100:1",
    "H100:2",
    "H100:3",
    "H100:4",
    "H100:5",
    "H100:6",
    "H100:7",
    "H100:8",
    # H200
    "H200:1",
    "H200:2",
    "H200:3",
    "H200:4",
    "H200:5",
    "H200:6",
    "H200:7",
    "H200:8",
    # L4
    "L4:1",
    "L4:2",
    "L4:3",
    "L4:4",
    "L4:5",
    "L4:6",
    "L4:7",
    "L4:8",
    # L40s
    "L40s:1",
    "L40s:2",
    "L40s:3",
    "L40s:4",
    "L40s:5",
    "L40s:6",
    "L40s:7",
    "L40s:8",
    # V100
    "V100:1",
    "V100:2",
    "V100:3",
    "V100:4",
    "V100:5",
    "V100:6",
    "V100:7",
    "V100:8",
    # RTX_PRO_4500
    "RTX_PRO_4500:1",
    # RTX_PRO_6000
    "RTX_PRO_6000:1",
    "RTX_PRO_6000:2",
    # GB10
    "GB10:1",
    # T4
    "T4:1",
    "T4:2",
    "T4:3",
    "T4:4",
    "T4:5",
    "T4:6",
    "T4:7",
    "T4:8",
    # Shared (time-sliced or MPS) GPUs
    "T4 shared:1",
    "L4 shared:1",
    "A10G shared:1",
    "A10 shared:1",
    "L40S shared:1",
    # Trn1
    "Trn1:1",
    "Trn1:4",
    "Trn1:8",
    "Trn1:16",
    # Trn1n
    "Trn1n:1",
    "Trn1n:4",
    "Trn1n:8",
    "Trn1n:16",
    # Trn2
    "Trn2:1",
    "Trn2:4",
    "Trn2:8",
    "Trn2:16",
    # Trn2u
    "Trn2u:1",
    "Trn2u:4",
    "Trn2u:8",
    "Trn2u:16",
    # Inf1
    "Inf1:1",
    "Inf1:2",
    "Inf1:3",
    "Inf1:4",
    "Inf1:5",
    "Inf1:6",
    "Inf1:7",
    "Inf1:8",
    "Inf1:9",
    "Inf1:10",
    "Inf1:11",
    "Inf1:12",
    "Inf1:13",
    "Inf1:14",
    "Inf1:15",
    "Inf1:16",
    # Inf2
    "Inf2:1",
    "Inf2:2",
    "Inf2:3",
    "Inf2:4",
    "Inf2:5",
    "Inf2:6",
    "Inf2:7",
    "Inf2:8",
    "Inf2:9",
    "Inf2:10",
    "Inf2:11",
    "Inf2:12",
    # MI100
    "MI100:1",
    # MI210
    "MI210:1",
    # MI250
    "MI250:1",
    # MI250X
    "MI250X:1",
    # MI300A
    "MI300A:1",
    # MI300X
    "MI300X:1",
    # MI325X
    "MI325X:1",
    # MI350X
    "MI350X:1",
    # MI355X
    "MI355X:1",
    # Habana Gaudi
    "Gaudi1:1",
    # Apple Silicon (devbox-local)
    "AppleSilicon",  # the Mac has one GPU, so the count is optional
    "AppleSilicon:1",
]

DeviceClass = Literal["GPU", "TPU", "NEURON", "AMD_GPU", "HABANA_GAUDI", "APPLE_SILICON"]

_DEPRECATED_GPU_ALIASES: Dict[str, str] = {
    "A100 80G": "A100_80G",
    "RTX PRO 6000": "RTX_PRO_6000",
}
"""
Space-separated GPU names that were accepted before the underscore spellings replaced them.
They are still accepted -- normalized to the current name with a deprecation warning -- so
task definitions written against earlier releases keep working.
"""


def _normalize_gpu_type(device: str) -> str:
    current = _DEPRECATED_GPU_ALIASES.get(device)
    if current is None:
        return device
    from flyte._logging import logger

    logger.warning(f"GPU type '{device}' is deprecated, use '{current}' instead.")
    return current


_DeviceClassType: Dict[typing.Any, str] = {
    GPUType: "GPU",
    TPUType: "TPU",
    NeuronType: "NEURON",
    AMD_GPUType: "AMD_GPU",
    HABANA_GAUDIType: "HABANA_GAUDI",
}
# Kept out of `_DeviceClassType`: code outside the SDK walks that table expecting only cluster accelerators,
# and the devbox-local Apple Silicon device has no cluster spelling to map to.
_DEVBOX_DEVICE_CLASSES: Dict[str, str] = dict.fromkeys(get_args(AppleSiliconType), "APPLE_SILICON")


@rich.repr.auto
@dataclass(frozen=True, slots=True)
class Device:
    """
    Represents a device type, its quantity and partition if applicable.

    Args:
        device: The type of device (e.g., "T4", "A100").
        quantity: The number of devices of this type.
        partition: The partition of the device (e.g., "1g.5gb", "2g.10gb" for gpus) or ("1x1", ... for tpus).
    """

    quantity: int
    device_class: DeviceClass
    device: str | None = None
    partition: str | None = None

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError("GPU quantity must be at least 1")
        if (self.device in get_args(AppleSiliconType)) != (self.device_class == "APPLE_SILICON"):
            raise ValueError('The Apple Silicon device is device="AppleSilicon" with device_class="APPLE_SILICON"')
        if self.device_class == "APPLE_SILICON" and self.quantity != 1:
            raise ValueError("An Apple Silicon device is the Mac's single Metal GPU; quantity must be 1")


def GPU(
    device: GPUType,
    quantity: GPUQuantity,
    partition: A100Parts | A100_80GBParts | H100Parts | H200Parts | RTXPro6000Parts | None = None,
) -> Device:
    """
    Create a GPU device instance.

    Args:
        device: The type of GPU (e.g., "T4", "A100"). A "<model> shared" device such as "T4 shared"
            requests time-sliced (or MPS) capacity on a pool that advertises the shared accelerator.
        quantity: The number of GPUs of this type.
        partition: The partition of the GPU (e.g., "1g.5gb", "2g.10gb" for gpus) or ("1x1", ... for tpus).
            A shared device takes no partition.

    Returns:
        Device instance.
    """
    if quantity < 1:
        raise ValueError("GPU quantity must be at least 1")
    device = _normalize_gpu_type(device)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    if device not in get_args(GPUType):
        raise ValueError(f"Invalid GPU type: {device}. Must be one of {get_args(GPUType)}")
    if partition is not None and device in get_args(SharedGPUType):
        raise ValueError(
            f"A shared GPU takes no partition, got {partition!r} for {device}. "
            "Request a plain device to use a MIG partition."
        )
    if partition is not None and device == "A100":
        if partition not in get_args(A100Parts):
            raise ValueError(f"Invalid partition for A100: {partition}. Must be one of {get_args(A100Parts)}")
    elif partition is not None and device == "A100_80G":
        if partition not in get_args(A100_80GBParts):
            raise ValueError(f"Invalid partition for A100_80G: {partition}. Must be one of {get_args(A100_80GBParts)}")
    elif partition is not None and device == "H100":
        if partition not in get_args(H100Parts):
            raise ValueError(f"Invalid partition for H100: {partition}. Must be one of {get_args(H100Parts)}")
    elif partition is not None and device == "H200":
        if partition not in get_args(H200Parts):
            raise ValueError(f"Invalid partition for H200: {partition}. Must be one of {get_args(H200Parts)}")
    elif partition is not None and device == "RTX_PRO_6000":
        if partition not in get_args(RTXPro6000Parts):
            raise ValueError(
                f"Invalid partition for RTX_PRO_6000: {partition}. Must be one of {get_args(RTXPro6000Parts)}"
            )
    return Device(device=device, quantity=quantity, partition=partition, device_class="GPU")


def TPU(device: TPUType, partition: V5EParts | V5PParts | V6EParts | None = None):
    """
    Create a TPU device instance.

    Args:
        device: Device type (e.g., "V5E", "V5P", "V6E").
        partition: Partition of the TPU (e.g., "1x1", "2x2", ...).

    Returns:
        Device instance.
    """
    if device not in get_args(TPUType):
        raise ValueError(f"Invalid TPU type: {device}. Must be one of {get_args(TPUType)}")
    if partition is not None and device == "V5P":
        if partition not in get_args(V5PParts):
            raise ValueError(f"Invalid partition for V5P: {partition}. Must be one of {get_args(V5PParts)}")
    elif partition is not None and device == "V6E":
        if partition not in get_args(V6EParts):
            raise ValueError(f"Invalid partition for V6E: {partition}. Must be one of {get_args(V6EParts)}")
    elif partition is not None and device == "V5E":
        if partition not in get_args(V5EParts):
            raise ValueError(f"Invalid partition for V5E: {partition}. Must be one of {get_args(V5EParts)}")
    return Device(1, "TPU", device, partition)


def Neuron(device: NeuronType) -> Device:
    """
    Create a Neuron device instance.

    Args:
        device: Device type (e.g., "Inf1", "Inf2", "Trn1", "Trn1n", "Trn2", "Trn2u").
        quantity: The number of Neuron devices of this type.

    Returns:
        Device instance.
    """
    if device not in get_args(NeuronType):
        raise ValueError(f"Invalid Neuron type: {device}. Must be one of {get_args(NeuronType)}")
    return Device(device=device, quantity=1, device_class="NEURON")


def AMD_GPU(device: AMD_GPUType) -> Device:
    """
    Create an AMD GPU device instance.

    Args:
        device: Device type (e.g., "MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300X", "MI325X", "MI350X",
            "MI355X").

    Returns:
        Device instance.
    """
    if device not in get_args(AMD_GPUType):
        raise ValueError(f"Invalid AMD GPU type: {device}. Must be one of {get_args(AMD_GPUType)}")
    return Device(device=device, quantity=1, device_class="AMD_GPU")


def HABANA_GAUDI(device: HABANA_GAUDIType) -> Device:
    """
    Create a Habana Gaudi device instance.

    Args:
        device: Device type (e.g., "Gaudi1").

    Returns:
        Device instance.
    """
    if device not in get_args(HABANA_GAUDIType):
        raise ValueError(f"Invalid Habana Gaudi type: {device}. Must be one of {get_args(HABANA_GAUDIType)}")
    return Device(device=device, quantity=1, device_class="HABANA_GAUDI")


def APPLE_SILICON() -> Device:
    """
    Create an Apple Silicon device instance: the Metal GPU of the Mac a devbox runs on.

    The same as `gpu="AppleSilicon"`. The workload runs natively on the Mac, so its code can use Metal
    (`torch.device("mps")`, `mlx`); it needs a devbox started with `flyte start devbox --apple-silicon`.

    Returns:
        Device instance.
    """
    return Device(device="AppleSilicon", quantity=1, device_class="APPLE_SILICON")


CPUBaseType = int | float | str


def _check_quantity_string(
    field_name: str, value: typing.Any, *, allow_pair: bool = False, allow_none: bool = True
) -> None:
    """
    Reject a `Resources` quantity that is not the Kubernetes-style string the field is declared to take.

    `cpu` is the only quantity the serializer stringifies (`_get_cpu_resource_entry` does `str(cpu)`),
    because it legitimately accepts an int or float core count. `memory`, `disk` and `shm` are declared
    `str` and are handed to protobuf string fields untouched, so a non-string reaches protobuf and dies
    there as `TypeError: bad argument type for built-in operation` -- raised in SDK frames, naming
    neither the field that was wrong nor the task it came from.

    Reject rather than coerce. `memory=1024` is a plausible mistake, and `str(1024)` is a valid
    Kubernetes quantity meaning 1024 *bytes*, so coercing would quietly deploy a task with a thousandth
    of the memory the user meant.

    `None` is how the field itself says "unset", so it is accepted there and nowhere else. As one half
    of a request/limit pair it is not a quantity at all: protobuf takes it as the empty string, and
    `memory=(None, "2Gi")` would serialize a request for `''` rather than leaving the request unset.
    """
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{field_name} pair elements must both be set, got None")
    if allow_pair and isinstance(value, tuple):
        for element in value:
            _check_quantity_string(field_name, element, allow_none=False)
        return
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a Kubernetes quantity string such as '1Gi', '512Mi' or '2G', "
            f"got {type(value).__name__}: {value!r}"
        )


def _check_cpu_quantity(value: typing.Any, *, allow_pair: bool = False, allow_none: bool = True) -> None:
    """
    Reject a `cpu` value that is not one of the types the field is declared to take.

    `cpu` is exempt from `_check_quantity_string` because it legitimately accepts an int or float
    core count as well as a Kubernetes quantity string, and `_get_cpu_resource_entry` stringifies
    whatever it is handed. That exemption left the field with no type check at all: `cpu=[1]` and
    `cpu=(None, 2)` are accepted here and reach the backend as the literal CPU quantities `'[1]'`
    and `'None'`, rather than being rejected where the field name is still known.

    The sign check had the same gap in its pair spelling. `cpu=-1` raises, but `cpu=(-1, 2)` was
    accepted and serialized a request for -1 cores, because the check only ever looked at a bare
    value. `memory` validates both halves of its pair -- `_check_quantity_string` recurses when
    `allow_pair` is set -- so before this change the same bad value was treated one way as
    `memory=(...)` and the opposite way as `cpu=(...)`.
    """
    if value is None:
        if allow_none:
            return
        raise ValueError("cpu pair elements must both be set, got None")
    if allow_pair and isinstance(value, tuple):
        for element in value:
            _check_cpu_quantity(element, allow_none=False)
        return
    if not isinstance(value, (int, float, str)):
        raise ValueError(
            f"cpu must be a core count or a Kubernetes quantity string such as 1, 0.5 or '100m', "
            f"got {type(value).__name__}: {value!r}"
        )
    if isinstance(value, (int, float)) and value < 0:
        raise ValueError("cpu must be greater than or equal to 0")


@dataclass
class Resources:
    """
    Resources such as CPU, Memory, and GPU that can be allocated to a task.

    Set via `TaskEnvironment(resources=...)` or `task.override(resources=...)`.

    Examples:

    ```python
    # Simple: 1 CPU, 1 GiB memory, 1 T4 GPU
    Resources(cpu=1, memory="1Gi", gpu="T4:1")

    # Range: request 1 CPU (limit 2), 2 GiB memory, 8 A100 GPUs, 10 GiB disk
    Resources(cpu=(1, 2), memory="2Gi", gpu="A100:8", disk="10Gi")

    # Advanced GPU with partitioning
    Resources(gpu=GPU(device="A100", quantity=1, partition="1g.5gb"))

    # TPU
    Resources(gpu=TPU(device="V5P", partition="2x2x1"))

    # Apple Silicon: on a devbox running on an Apple Silicon Mac, the workload runs natively on the Mac,
    # so its code can use the Metal GPU (torch "mps", mlx)
    Resources(memory="16Gi", gpu="AppleSilicon")  # or gpu=APPLE_SILICON()
    ```

    Args:
        cpu: CPU allocation. Accepts several formats:

            - `int` or `float`: CPU cores (e.g., `1`, `0.5`)
            - `str`: Kubernetes-style (e.g., `"500m"` for 0.5 cores, `"2"` for 2 cores)
            - `tuple`: Request/limit range (e.g., `(1, 4)` requests 1 core, limits to 4)
        memory: Memory allocation using Kubernetes unit conventions:

            - Binary units: `"512Mi"`, `"1Gi"`, `"4Gi"`
            - Decimal units: `"500M"`, `"1G"`
            - `tuple`: Request/limit range (e.g., `("1Gi", "4Gi")`)
        gpu: GPU, TPU, or other accelerator allocation. Accepts:

            - `int`: GPU count, any available type (e.g., `1`, `4`)
            - `str`: Type and quantity (e.g., `"T4:1"`, `"A100:2"`, `"H100:8"`)
            - `Device`: Advanced config via `GPU()`, `TPU()`, or `Device()` for partitioning
              and custom device types. See `GPU`, `TPU`, `Device` for details.

            Supported GPU types include A2, T4, L4, L40s, A10, A10G, A100, A100_80G, B200, H100, H200, V100,
            RTX_PRO_4500, RTX_PRO_6000, and GB10.
            GPU partitioning (MIG) is available on A100, A100_80G, H100, H200, and RTX_PRO_6000.
            Time-sliced (or MPS) shares come from `GPU("T4 shared", 1)` and the other
            `SharedGPUType` devices.
        disk: Ephemeral disk storage as a string with Kubernetes units
            (e.g., `"10Gi"`, `"100Gi"`, `"1Ti"`). Automatically cleaned up when the task completes.
        shm: Shared memory (`/dev/shm`) allocation. Useful for ML data loading
            and inter-process communication:

            - `str`: Size with units (e.g., `"1Gi"`, `"16Gi"`)
            - `"auto"`: Set to the maximum shared memory available on the node
    """

    cpu: Union[CPUBaseType, Tuple[CPUBaseType, CPUBaseType], None] = None
    memory: Union[str, Tuple[str, str], None] = None
    gpu: Union[Accelerators, int, Device, None] = None
    disk: Union[str, None] = None
    shm: Union[str, Literal["auto"], None] = None

    def __post_init__(self):
        if isinstance(self.cpu, tuple):
            if len(self.cpu) != 2:
                raise ValueError("cpu tuple must have exactly two elements")
        if isinstance(self.memory, tuple):
            if len(self.memory) != 2:
                raise ValueError("memory tuple must have exactly two elements")
        _check_cpu_quantity(self.cpu, allow_pair=True)
        _check_quantity_string("memory", self.memory, allow_pair=True)
        _check_quantity_string("disk", self.disk)
        _check_quantity_string("shm", self.shm)
        if self.gpu is not None:
            if isinstance(self.gpu, int):
                if self.gpu < 0:
                    raise ValueError("gpu must be greater than or equal to 0")
            elif isinstance(self.gpu, str):
                device, sep, portion = self.gpu.partition(":")
                if device in _DEPRECATED_GPU_ALIASES:
                    self.gpu = f"{_normalize_gpu_type(device)}{sep}{portion}"  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
                if self.gpu not in get_args(Accelerators):
                    raise ValueError(f"gpu must be one of {Accelerators}, got {self.gpu}")
            elif not isinstance(self.gpu, Device):
                # Anything else is silently carried by `get_device` and only fails much
                # later, in serialization, as an AttributeError against whatever was
                # passed -- a crash in SDK frames rather than a report about the task
                # definition. A `(device, quantity)` tuple is the common way to land
                # here: it is what the accelerator string decomposes into, and what
                # `get_device` used to be documented as returning.
                raise ValueError(
                    "gpu must be an accelerator string such as 'A100:4', a device count, or a "
                    "GPU()/TPU()/Device() instance, got "
                    f"{type(self.gpu).__name__}: {self.gpu!r}"
                )

    def get_device(self) -> Optional[Device]:
        """
        Get the accelerator device for the task.

        Returns:
            A `Device` carrying the quantity, device class and -- for an accelerator string or an
            explicit `GPU()`/`TPU()`/`Device()` -- the device name and partition. `None` when no
            accelerator is requested, which includes `gpu=0`. Partition values follow the default
            cloud provider labels, typically `1g.5gb`, `2g.10gb`, etc.
        """
        if self.gpu is None:
            return None
        if isinstance(self.gpu, int):
            # `gpu=0` is a valid way to say "no accelerator" -- __post_init__ accepts any
            # count >= 0 -- and is what a computed count collapses to. `Device` requires a
            # quantity of at least 1, so report it the same way an unset `gpu` is reported.
            if self.gpu == 0:
                return None
            return Device(quantity=self.gpu, device_class="GPU")
        if isinstance(self.gpu, str):
            device, _, portion = self.gpu.partition(":")
            if device in _DEVBOX_DEVICE_CLASSES:
                # "AppleSilicon" alone means the Mac's one GPU.
                return Device(device=device, device_class=_DEVBOX_DEVICE_CLASSES[device], quantity=int(portion or 1))  # type: ignore
            for cls, cls_name in _DeviceClassType.items():
                if device in get_args(cls):
                    device_class = cls_name
                    break
            else:
                raise ValueError(f"Invalid device type: {device}. Must be one of {list(_DeviceClassType.keys())}")
            return Device(device=device, device_class=device_class, quantity=int(portion))  # type: ignore
        return self.gpu

    def get_shared_memory(self) -> Optional[str]:
        """
        Get the shared memory string for the task.

        Returns:
            The shared memory string.
        """
        if self.shm is None:
            return None
        if self.shm == "auto":
            return ""
        return self.shm


def _check_resource_is_singular(resource: Resources):
    """
    Raise a value error if the resource has a tuple.
    """
    for field in fields(resource):
        value = getattr(resource, field.name)
        if isinstance(value, (tuple, list)):
            raise ValueError(f"{value} can not be a list or tuple")
    return resource


def pod_spec_from_resources(
    primary_container_name: str = _PRIMARY_CONTAINER_DEFAULT_NAME,
    requests: Optional[Resources] = None,
    limits: Optional[Resources] = None,
    k8s_gpu_resource_key: str = "nvidia.com/gpu",
) -> "V1PodSpec":
    from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

    def _construct_k8s_pods_resources(resources: Optional[Resources], k8s_gpu_resource_key: str):
        if resources is None:
            return None

        # Maps a `Resources` field onto its container resource key. `shm` is deliberately
        # absent: shared memory is a volume, surfaced through `ExtendedResources` rather
        # than as a key under container resources, so it is skipped here along with any
        # other field that has no container-level equivalent.
        resources_map = {
            "cpu": "cpu",
            "memory": "memory",
            "gpu": k8s_gpu_resource_key,
            "disk": "ephemeral-storage",
        }

        k8s_pod_resources = {}

        _check_resource_is_singular(resources)
        for resource in fields(resources):
            k8s_resource_name = resources_map.get(resource.name)
            if k8s_resource_name is None:
                continue
            resource_value = getattr(resources, resource.name)
            if resource_value is not None:
                if resource.name == "gpu":
                    device = resources.get_device()
                    if device is not None:
                        resource_value = str(device.quantity)
                k8s_pod_resources[k8s_resource_name] = resource_value

        return k8s_pod_resources

    requests = _construct_k8s_pods_resources(resources=requests, k8s_gpu_resource_key=k8s_gpu_resource_key)
    limits = _construct_k8s_pods_resources(resources=limits, k8s_gpu_resource_key=k8s_gpu_resource_key)
    requests = requests or limits
    limits = limits or requests

    return V1PodSpec(
        containers=[
            V1Container(
                name=primary_container_name,
                resources=V1ResourceRequirements(
                    requests=requests,
                    limits=limits,
                ),
            )
        ]
    )
