import typing
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, get_args

import rich.repr

from flyte._pod import _PRIMARY_CONTAINER_DEFAULT_NAME

if TYPE_CHECKING:
    from kubernetes.client import V1PodSpec

PRIMARY_CONTAINER_DEFAULT_NAME = "primary"

GPUType = Literal["A10", "A10G", "A100", "A100 80G", "B200", "H100", "L4", "L40s", "T4", "V100", "RTX PRO 6000"]
GPUQuantity = Literal[1, 2, 3, 4, 5, 6, 7, 8]
A100Parts = Literal["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]
"""
Partitions for NVIDIA A100 GPU.
"""

A100_80GBParts = Literal["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
"""
Partitions for NVIDIA A100 80GB GPU.
"""

TPUType = Literal["V5P", "V6E"]
V5EParts = Literal["1x1", "2x2", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16"]

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

Accelerators = Literal[
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
    # A100 80G
    "A100 80G:1",
    "A100 80G:2",
    "A100 80G:3",
    "A100 80G:4",
    "A100 80G:5",
    "A100 80G:6",
    "A100 80G:7",
    "A100 80G:8",
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
    # RTX 6000
    "RTX PRO 6000:1",
    # T4
    "T4:1",
    "T4:2",
    "T4:3",
    "T4:4",
    "T4:5",
    "T4:6",
    "T4:7",
    "T4:8",
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
]

DeviceClass = Literal["GPU", "TPU", "NEURON", "AMD_GPU", "HABANA_GAUDI"]

_DeviceClassType: Dict[typing.Any, str] = {
    GPUType: "GPU",
    TPUType: "TPU",
    NeuronType: "NEURON",
    AMD_GPUType: "AMD_GPU",
    HABANA_GAUDIType: "HABANA_GAUDI",
}


@rich.repr.auto
@dataclass(frozen=True, slots=True)
class Device:
    """
     Represents a device type, its quantity and partition if applicable.
    :param device: The type of device (e.g., "T4", "A100").
    :param quantity: The number of devices of this type.
    :param partition: The partition of the device (e.g., "1g.5gb", "2g.10gb" for gpus) or ("1x1", ... for tpus).
    """

    quantity: int
    device_class: DeviceClass
    device: str | None = None
    partition: str | None = None

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError("GPU quantity must be at least 1")


def GPU(device: GPUType, quantity: GPUQuantity, partition: A100Parts | A100_80GBParts | None = None) -> Device:
    """
    Create a GPU device instance.
    :param device: The type of GPU (e.g., "T4", "A100").
    :param quantity: The number of GPUs of this type.
    :param partition: The partition of the GPU (e.g., "1g.5gb", "2g.10gb" for gpus) or ("1x1", ... for tpus).
    :return: Device instance.
    """
    if quantity < 1:
        raise ValueError("GPU quantity must be at least 1")
    if device not in get_args(GPUType):
        raise ValueError(f"Invalid GPU type: {device}. Must be one of {get_args(GPUType)}")
    if partition is not None and device == "A100":
        if partition not in get_args(A100Parts):
            raise ValueError(f"Invalid partition for A100: {partition}. Must be one of {get_args(A100Parts)}")
    elif partition is not None and device == "A100 80G":
        if partition not in get_args(A100_80GBParts):
            raise ValueError(f"Invalid partition for A100 80G: {partition}. Must be one of {get_args(A100_80GBParts)}")
    return Device(device=device, quantity=quantity, partition=partition, device_class="GPU")


def TPU(device: TPUType, partition: V5PParts | V6EParts | None = None):
    """
    Create a TPU device instance.
    :param device: Device type (e.g., "V5P", "V6E").
    :param partition: Partition of the TPU (e.g., "1x1", "2x2", ...).
    :return: Device instance.
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
    :param device: Device type (e.g., "Inf1", "Inf2", "Trn1", "Trn1n", "Trn2", "Trn2u").
    :param quantity: The number of Neuron devices of this type.
    :return: Device instance.
    """
    if device not in get_args(NeuronType):
        raise ValueError(f"Invalid Neuron type: {device}. Must be one of {get_args(NeuronType)}")
    return Device(device=device, quantity=1, device_class="NEURON")


def AMD_GPU(device: AMD_GPUType) -> Device:
    """
    Create an AMD GPU device instance.
    :param device: Device type (e.g., "MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300X", "MI325X", "MI350X",
                   "MI355X").
    :return: Device instance.
    """
    if device not in get_args(AMD_GPUType):
        raise ValueError(f"Invalid AMD GPU type: {device}. Must be one of {get_args(AMD_GPUType)}")
    return Device(device=device, quantity=1, device_class="AMD_GPU")


def HABANA_GAUDI(device: HABANA_GAUDIType) -> Device:
    """
    Create a Habana Gaudi device instance.
    :param device: Device type (e.g., "DL1").
    :return: Device instance.
    """
    if device not in get_args(HABANA_GAUDIType):
        raise ValueError(f"Invalid Habana Gaudi type: {device}. Must be one of {get_args(HABANA_GAUDIType)}")
    return Device(device=device, quantity=1, device_class="HABANA_GAUDI")


CPUBaseType = int | float | str


@dataclass
class Resources:
    """
    Resources such as CPU, Memory, and GPU that can be allocated to a task.

    Example:
    - Single CPU, 1GiB of memory, and 1 T4 GPU:
    ```python
    @task(resources=Resources(cpu=1, memory="1GiB", gpu="T4:1"))
    def my_task() -> int:
        return 42
    ```
    - 1CPU with limit upto 2CPU, 2GiB of memory, and 8 A100 GPUs and 10GiB of disk:
    ```python
    @task(resources=Resources(cpu=(1, 2), memory="2GiB", gpu="A100:8", disk="10GiB"))
    def my_task() -> int:
        return 42
    ```

    :param cpu: The amount of CPU to allocate to the task. This can be a string, int, float, list of ints or strings,
        or a tuple of two ints or strings.
    :param memory: The amount of memory to allocate to the task. This can be a string, int, float, list of ints or
        strings, or a tuple of two ints or strings.
    :param gpu: The amount of GPU to allocate to the task. This can be an Accelerators enum, an int, or None.
    :param disk: The amount of disk to allocate to the task. This is a string of the form "10GiB".
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
        if isinstance(self.cpu, (int, float)):
            if self.cpu < 0:
                raise ValueError("cpu must be greater than or equal to 0")
        if self.gpu is not None:
            if isinstance(self.gpu, int):
                if self.gpu < 0:
                    raise ValueError("gpu must be greater than or equal to 0")
            elif isinstance(self.gpu, str):
                if self.gpu not in get_args(Accelerators):
                    raise ValueError(f"gpu must be one of {Accelerators}, got {self.gpu}")

    def get_device(self) -> Optional[Device]:
        """
        Get the accelerator string for the task.

        :return: If GPUs are requested, return a tuple of the device name, and potentially a partition string.
                 Default cloud provider labels typically use the following values: `1g.5gb`, `2g.10gb`, etc.
        """
        if self.gpu is None:
            return None
        if isinstance(self.gpu, int):
            return Device(quantity=self.gpu, device_class="GPU")
        if isinstance(self.gpu, str):
            device, portion = self.gpu.split(":")
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

        :return: The shared memory string.
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

        resources_map = {
            "cpu": "cpu",
            "memory": "memory",
            "gpu": k8s_gpu_resource_key,
            "ephemeral_storage": "ephemeral-storage",
        }

        k8s_pod_resources = {}

        _check_resource_is_singular(resources)
        for resource in fields(resources):
            resource_value = getattr(resources, resource.name)
            if resource_value is not None:
                k8s_pod_resources[resources_map[resource.name]] = resource_value

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
