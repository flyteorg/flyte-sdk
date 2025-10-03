import pytest
from flyteidl2.core import tasks_pb2

from flyte._internal.runtime.resources_serde import (
    ACCELERATOR_DEVICE_MAP,
    _get_cpu_resource_entry,
    _get_gpu_extended_resource_entry,
    get_proto_extended_resources,
    get_proto_resources,
)
from flyte._resources import GPU, Resources


def test_cpu_single():
    entry = _get_cpu_resource_entry(2)
    assert entry.name == tasks_pb2.Resources.ResourceName.CPU
    assert entry.value == "2"


def test_cpu_tuple():
    res = Resources(cpu=(1, 2))
    proto = get_proto_resources(res)
    assert proto.requests[0].value == "1"
    assert proto.limits[0].value == "2"


def test_memory_single():
    res = Resources(memory="2GiB")
    proto = get_proto_resources(res)
    assert proto.requests[0].name == tasks_pb2.Resources.ResourceName.MEMORY
    assert proto.requests[0].value == "2GiB"


def test_memory_tuple():
    res = Resources(memory=("2GiB", "4GiB"))
    proto = get_proto_resources(res)
    assert proto.requests[0].value == "2GiB"
    assert proto.limits[0].value == "4GiB"


def test_gpu_int():
    res = Resources(gpu=1)
    proto = get_proto_resources(res)
    assert proto.requests[0].name == tasks_pb2.Resources.ResourceName.GPU
    assert proto.requests[0].value == "1"


@pytest.mark.parametrize(
    "gpu_str",
    [
        "T4:1",
        "A100:4",
        "A100 80G:2",
        "L4:1",
        "L40s:2",
    ],
)
def test_gpu_accelerator_mapping(gpu_str):
    res = Resources(gpu=gpu_str)  # type: ignore
    acc = _get_gpu_extended_resource_entry(res)
    assert acc is not None
    device, _ = gpu_str.split(":")
    assert acc.device == ACCELERATOR_DEVICE_MAP[device]
    # TODO: implement partition size logic


def test_gpu_invalid_type():
    with pytest.raises(ValueError):
        Resources(gpu="InvalidGPU:1")  # type: ignore


def test_disk():
    res = Resources(disk="10GiB")
    proto = get_proto_resources(res)
    entry = next(e for e in proto.requests if e.name == tasks_pb2.Resources.ResourceName.EPHEMERAL_STORAGE)
    assert entry.value == "10GiB"


def test_shared_memory_auto():
    res = Resources(shm="auto")
    proto = get_proto_extended_resources(res)
    assert proto.shared_memory is not None
    assert proto.shared_memory.size_limit == ""


def test_shared_memory_custom():
    res = Resources(shm="2GiB")
    proto = get_proto_extended_resources(res)
    assert proto.shared_memory.size_limit == "2GiB"


def test_shared_memory_none():
    res = Resources(shm=None)
    proto = get_proto_extended_resources(res)
    assert proto is None


def test_combined_resources():
    res = Resources(cpu=(1, 2), memory=("2GiB", "4GiB"), gpu="T4:1", disk="5GiB", shm="1GiB")
    proto_main = get_proto_resources(res)
    proto_ext = get_proto_extended_resources(res)
    assert proto_main is not None
    assert proto_ext is not None
    assert proto_ext.gpu_accelerator.device == ACCELERATOR_DEVICE_MAP["T4"]
    assert proto_ext.shared_memory.size_limit == "1GiB"


def test_empty_resources():
    res = Resources()
    assert get_proto_resources(res) is None
    assert get_proto_extended_resources(res) is None


def test_gpu_extended_resource_entry():
    res = Resources(gpu="T4:1")
    acc = _get_gpu_extended_resource_entry(res)
    assert acc is not None
    assert acc.device == "nvidia-tesla-t4"
    assert not acc.partition_size


def test_gpu_partition_size():
    res = Resources(gpu=GPU("A100", 4, partition="1g.5gb"))
    acc = _get_gpu_extended_resource_entry(res)
    assert acc is not None
    assert acc.device == "nvidia-tesla-a100"
    assert acc.partition_size == "1g.5gb"


def test_device_class_with_fallback():
    """Test that unknown device_class falls back to NVIDIA_GPU"""
    from flyte._resources import Device

    # Create a Device with an unknown device_class by using Device directly
    # This bypasses the GPU() helper which enforces valid types
    unknown_device = Device(quantity=2, device_class="UNKNOWN_CLASS", device="T4")  # type: ignore
    res = Resources(gpu=unknown_device)
    acc = _get_gpu_extended_resource_entry(res)
    assert acc is not None
    assert acc.device_class == tasks_pb2.GPUAccelerator.NVIDIA_GPU  # Should fallback to NVIDIA_GPU


def test_accelerator_map_with_fallback():
    """Test that unknown accelerator device falls back to original device string"""
    from flyte._resources import Device

    # Create a Device with an unknown device type
    unknown_device = Device(quantity=2, device_class="GPU", device="CustomGPU-X1000")
    res = Resources(gpu=unknown_device)
    acc = _get_gpu_extended_resource_entry(res)
    assert acc is not None
    assert acc.device == "CustomGPU-X1000"  # Should use original device string as fallback


