from typing import Dict, List, Optional, Tuple

from flyteidl2.core import tasks_pb2

from flyte._resources import CPUBaseType, DeviceClass, Resources

ACCELERATOR_DEVICE_MAP = {
    "A100": "nvidia-tesla-a100",
    "A100 80G": "nvidia-a100-80gb",
    "A10": "nvidia-a10",
    "A10G": "nvidia-a10g",
    "A100G": "nvidia-a100g",
    "L4": "nvidia-l4",
    "L40s": "nvidia-l40s",
    "L4_VWS": "nvidia-l4-vws",
    "K80": "nvidia-tesla-k80",
    "M60": "nvidia-tesla-m60",
    "P4": "nvidia-tesla-p4",
    "P100": "nvidia-tesla-p100",
    "T4": "nvidia-tesla-t4",
    "V100": "nvidia-tesla-v100",
    "V5E": "tpu-v5-lite-podslice",
    "V5P": "tpu-v5p-slice",
    "V6E": "tpu-v6e-slice",
}

_DeviceClassToProto: Dict[DeviceClass, "tasks_pb2.GPUAccelerator.DeviceClass"] = {
    "GPU": tasks_pb2.GPUAccelerator.NVIDIA_GPU,
    "TPU": tasks_pb2.GPUAccelerator.GOOGLE_TPU,
    "NEURON": tasks_pb2.GPUAccelerator.AMAZON_NEURON,
    "AMD_GPU": tasks_pb2.GPUAccelerator.AMD_GPU,
    "HABANA_GAUDI": tasks_pb2.GPUAccelerator.HABANA_GAUDI,
}


def _get_cpu_resource_entry(cpu: CPUBaseType) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.CPU,
        value=str(cpu),
    )


def _get_memory_resource_entry(memory: str) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.MEMORY,
        value=memory,
    )


def _get_gpu_resource_entry(gpu: int) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.GPU,
        value=str(gpu),
    )


def _get_gpu_extended_resource_entry(resources: Resources) -> Optional[tasks_pb2.GPUAccelerator]:
    if resources is None:
        return None
    if resources.gpu is None or isinstance(resources.gpu, int):
        return None
    device = resources.get_device()
    if device is None:
        return None

    device_class = _DeviceClassToProto.get(device.device_class, tasks_pb2.GPUAccelerator.NVIDIA_GPU)
    if device.device is None:
        raise RuntimeError("Device type must be specified for GPU string.")
    else:
        device_type = device.device
    device_type = ACCELERATOR_DEVICE_MAP.get(device_type, device_type)
    return tasks_pb2.GPUAccelerator(
        device=device_type,
        partition_size=device.partition if device.partition else None,
        device_class=device_class,
    )


def _get_disk_resource_entry(disk: str) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.EPHEMERAL_STORAGE,
        value=disk,
    )


def get_proto_extended_resources(resources: Resources | None) -> Optional[tasks_pb2.ExtendedResources]:
    """
    TODO Implement partitioning logic string handling for GPU
    :param resources:
    """
    if resources is None:
        return None
    acc = _get_gpu_extended_resource_entry(resources)
    shm = resources.get_shared_memory()
    if acc is None and shm is None:
        return None
    proto_shm = None
    if shm is not None:
        proto_shm = tasks_pb2.SharedMemory(
            mount_path="/dev/shm",
            mount_name="flyte-shm",
            size_limit=shm,
        )
    return tasks_pb2.ExtendedResources(gpu_accelerator=acc, shared_memory=proto_shm)


def _convert_resources_to_resource_entries(
    resources: Resources | None,
) -> Tuple[List[tasks_pb2.Resources.ResourceEntry], List[tasks_pb2.Resources.ResourceEntry]]:
    request_entries: List[tasks_pb2.Resources.ResourceEntry] = []
    limit_entries: List[tasks_pb2.Resources.ResourceEntry] = []
    if resources is None:
        return request_entries, limit_entries
    if resources.cpu is not None:
        if isinstance(resources.cpu, tuple):
            request_entries.append(_get_cpu_resource_entry(resources.cpu[0]))
            limit_entries.append(_get_cpu_resource_entry(resources.cpu[1]))
        else:
            request_entries.append(_get_cpu_resource_entry(resources.cpu))

    if resources.memory is not None:
        if isinstance(resources.memory, tuple):
            request_entries.append(_get_memory_resource_entry(resources.memory[0]))
            limit_entries.append(_get_memory_resource_entry(resources.memory[1]))
        else:
            request_entries.append(_get_memory_resource_entry(resources.memory))

    if resources.gpu is not None:
        device = resources.get_device()
        if device is not None:
            request_entries.append(_get_gpu_resource_entry(device.quantity))

    if resources.disk is not None:
        request_entries.append(_get_disk_resource_entry(resources.disk))

    return request_entries, limit_entries


def get_proto_resources(resources: Resources | None) -> Optional[tasks_pb2.Resources]:
    """
    Get main resources IDL representation from the resources object

    :param resources: User facing Resources object containing potentially both requests and limits
    :return: The given resources as requests and limits
    """
    if resources is None:
        return None
    request_entries, limit_entries = _convert_resources_to_resource_entries(resources)
    if not request_entries and not limit_entries:
        return None

    return tasks_pb2.Resources(requests=request_entries, limits=limit_entries)
