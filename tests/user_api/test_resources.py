from typing import get_args

import pytest

from flyte._resources import (
    AMD_GPU,
    GPU,
    HABANA_GAUDI,
    TPU,
    Accelerators,
    AMD_GPUType,
    Device,
    GPUType,
    HABANA_GAUDIType,
    Neuron,
    NeuronType,
    Resources,
    TPUType,
)


def test_resources_gpu_with_int():
    res = Resources(gpu=1)
    assert res.gpu == 1
    device = res.get_device()
    assert device is not None
    assert device.quantity == 1
    assert device.device is None


def test_resources_gpu_with_valid_string():
    res = Resources(gpu="A100:2")
    assert res.gpu == "A100:2"
    device = res.get_device()
    assert device is not None
    assert device.device == "A100"
    assert device.quantity == 2


def test_resources_gpu_with_invalid_string():
    with pytest.raises(ValueError, match="gpu must be one of"):
        Resources(gpu="InvalidGPU:1")  # type: ignore


def test_resources_gpu_with_gpu_object():
    gpu = GPU(device="A100", quantity=1, partition="1g.5gb")
    res = Resources(gpu=gpu)
    assert res.gpu == gpu
    device = res.get_device()
    assert device is gpu


def test_resources_gpu_with_invalid_gpu_object():
    with pytest.raises(ValueError, match="Invalid partition for A100"):
        GPU(device="A100", quantity=1, partition="invalid_partition")  # type: ignore


def test_resources_gpu_with_tpu_object():
    tpu = TPU(device="V5P", partition="2x2x1")
    res = Resources(gpu=tpu)
    assert res.gpu == tpu
    device = res.get_device()
    assert device is tpu


def test_resources_gpu_with_negative_int():
    with pytest.raises(ValueError, match="gpu must be greater than or equal to 0"):
        Resources(gpu=-1)


def test_resources_gpu_with_invalid_quantity_in_string():
    with pytest.raises(ValueError, match="gpu must be one of"):
        Resources(gpu="A100:invalid_quantity")  # type: ignore


def test_resources_gpu_with_missing_colon_in_string():
    with pytest.raises(ValueError, match="gpu must be one of"):
        Resources(gpu="A100")  # type: ignore


def test_get_device_with_none_gpu():
    res = Resources(gpu=None)
    assert res.get_device() is None


def test_raw_device():
    res = Resources(gpu=Device(device="A100", device_class="GPU", quantity=1, partition="1g.5gb"))
    assert res.get_device().device == "A100"
    assert res.get_device().quantity == 1
    assert res.get_device().partition == "1g.5gb"


def test_shm():
    res = Resources(shm="1Gi")
    assert res.shm == "1Gi"
    assert res.get_shared_memory() == "1Gi"

    res = Resources(shm="auto")
    assert res.shm == "auto"
    assert res.get_shared_memory() == ""


def test_cpu():
    res = Resources(cpu="1")
    assert res.cpu == "1"

    res = Resources(cpu=1)
    assert res.cpu == 1

    res = Resources(cpu=("1", "2"))
    assert res.cpu == ("1", "2")

    res = Resources(cpu=(1, 2))
    assert res.cpu == (1, 2)

    res = Resources(cpu=0.5)
    assert res.cpu == 0.5

    with pytest.raises(ValueError, match="cpu tuple must have exactly two elements"):
        Resources(cpu=("1", "2", "3"))


def test_mem():
    res = Resources(memory="1Gi")
    assert res.memory == "1Gi"

    res = Resources(memory=("1Gi", "2Gi"))
    assert res.memory == ("1Gi", "2Gi")

    with pytest.raises(ValueError, match="memory tuple must have exactly two elements"):
        Resources(memory=("1Gi", "2Gi", "3Gi"))  # type: ignore


def test_resources_with_various_gpu_combinations():
    res = Resources(gpu=1)
    assert res.gpu == 1

    res = Resources(gpu="A100:2")
    assert res.gpu == "A100:2"
    device = res.get_device()
    assert device is not None
    assert device.device == "A100"
    assert device.quantity == 2

    res = Resources(gpu=GPU(device="A100", quantity=1, partition="1g.5gb"))
    assert res.gpu == GPU(device="A100", quantity=1, partition="1g.5gb")
    device = res.get_device()
    assert device is not None
    assert device.device == "A100"
    assert device.quantity == 1
    assert device.partition == "1g.5gb"

    res = Resources(gpu=TPU(device="V5P", partition="2x2x1"))
    assert res.gpu == TPU(device="V5P", partition="2x2x1")
    device = res.get_device()
    assert device is not None
    assert device.device == "V5P"
    assert device.partition == "2x2x1"
    assert device.quantity == 1


# GPU Accelerator Tests
@pytest.mark.parametrize(
    "gpu_type,quantity",
    [
        ("A10", 1),
        ("A10G", 2),
        ("A100", 4),
        ("A100 80G", 8),
        ("B200", 1),
        ("H100", 2),
        ("L4", 1),
        ("L40s", 2),
        ("T4", 4),
        ("V100", 1),
        ("RTX PRO 6000", 1),
    ],
)
def test_gpu_all_types(gpu_type, quantity):
    """Test all GPU types"""
    gpu = GPU(device=gpu_type, quantity=quantity)  # type: ignore
    assert gpu.device == gpu_type
    assert gpu.quantity == quantity
    assert gpu.device_class == "GPU"

    res = Resources(gpu=gpu)
    device = res.get_device()
    assert device is not None
    assert device.device == gpu_type
    assert device.quantity == quantity
    assert device.device_class == "GPU"


def test_gpu_with_a100_partitions():
    """Test A100 GPU with all valid partitions"""
    partitions = ["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]
    for partition in partitions:
        gpu = GPU(device="A100", quantity=1, partition=partition)  # type: ignore
        assert gpu.partition == partition
        assert gpu.device == "A100"
        assert gpu.device_class == "GPU"


def test_gpu_with_a100_80gb_partitions():
    """Test A100 80GB GPU with all valid partitions"""
    partitions = ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
    for partition in partitions:
        gpu = GPU(device="A100 80G", quantity=1, partition=partition)  # type: ignore
        assert gpu.partition == partition
        assert gpu.device == "A100 80G"
        assert gpu.device_class == "GPU"


def test_gpu_with_h100_partitions():
    """Test H100 GPU with all valid partitions"""
    partitions = ["1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
    for partition in partitions:
        gpu = GPU(device="H100", quantity=1, partition=partition)  # type: ignore
        assert gpu.partition == partition
        assert gpu.device == "H100"
        assert gpu.device_class == "GPU"


def test_gpu_with_h100_invalid_partition():
    """Test H100 GPU with invalid partition"""
    with pytest.raises(ValueError, match="Invalid partition for H100"):
        GPU(device="H100", quantity=1, partition="invalid")  # type: ignore


def test_gpu_invalid_device():
    """Test GPU with invalid device type"""
    with pytest.raises(ValueError, match="Invalid GPU type"):
        GPU(device="InvalidGPU", quantity=1)  # type: ignore


def test_gpu_invalid_quantity():
    """Test GPU with invalid quantity"""
    with pytest.raises(ValueError, match="GPU quantity must be at least 1"):
        GPU(device="A100", quantity=0)


# TPU Accelerator Tests
@pytest.mark.parametrize(
    "tpu_type,partition",
    [
        ("V5P", "2x2x1"),
        ("V5P", "2x2x2"),
        ("V5P", "4x4x4"),
        ("V6E", "1x1"),
        ("V6E", "2x2"),
        ("V6E", "4x4"),
    ],
)
def test_tpu_all_types_with_partitions(tpu_type, partition):
    """Test all TPU types with partitions"""
    tpu = TPU(device=tpu_type, partition=partition)  # type: ignore
    assert tpu.device == tpu_type
    assert tpu.partition == partition
    assert tpu.device_class == "TPU"
    assert tpu.quantity == 1

    res = Resources(gpu=tpu)
    device = res.get_device()
    assert device is not None
    assert device.device == tpu_type
    assert device.partition == partition
    assert device.device_class == "TPU"


def test_tpu_invalid_device():
    """Test TPU with invalid device type"""
    with pytest.raises(ValueError, match="Invalid TPU type"):
        TPU(device="InvalidTPU")  # type: ignore


def test_tpu_invalid_partition_v5p():
    """Test TPU V5P with invalid partition"""
    with pytest.raises(ValueError, match="Invalid partition for V5P"):
        TPU(device="V5P", partition="invalid")  # type: ignore


def test_tpu_invalid_partition_v6e():
    """Test TPU V6E with invalid partition"""
    with pytest.raises(ValueError, match="Invalid partition for V6E"):
        TPU(device="V6E", partition="invalid")  # type: ignore


# Neuron Accelerator Tests
@pytest.mark.parametrize(
    "neuron_type",
    ["Inf1", "Inf2", "Trn1", "Trn1n", "Trn2", "Trn2u"],
)
def test_neuron_all_types(neuron_type):
    """Test all Neuron types"""
    neuron = Neuron(device=neuron_type)  # type: ignore
    assert neuron.device == neuron_type
    assert neuron.quantity == 1
    assert neuron.device_class == "NEURON"

    res = Resources(gpu=neuron)
    device = res.get_device()
    assert device is not None
    assert device.device == neuron_type
    assert device.quantity == 1
    assert device.device_class == "NEURON"


def test_neuron_invalid_device():
    """Test Neuron with invalid device type"""
    with pytest.raises(ValueError, match="Invalid Neuron type"):
        Neuron(device="InvalidNeuron")  # type: ignore


# AMD GPU Accelerator Tests
@pytest.mark.parametrize(
    "amd_gpu_type",
    ["MI100", "MI210", "MI250", "MI250X", "MI300A", "MI300X", "MI325X", "MI350X", "MI355X"],
)
def test_amd_gpu_all_types(amd_gpu_type):
    """Test all AMD GPU types"""
    amd_gpu = AMD_GPU(device=amd_gpu_type)  # type: ignore
    assert amd_gpu.device == amd_gpu_type
    assert amd_gpu.quantity == 1
    assert amd_gpu.device_class == "AMD_GPU"

    res = Resources(gpu=amd_gpu)
    device = res.get_device()
    assert device is not None
    assert device.device == amd_gpu_type
    assert device.quantity == 1
    assert device.device_class == "AMD_GPU"


def test_amd_gpu_invalid_device():
    """Test AMD GPU with invalid device type"""
    with pytest.raises(ValueError, match="Invalid AMD GPU type"):
        AMD_GPU(device="InvalidAMD")  # type: ignore


# Habana Gaudi Accelerator Tests
@pytest.mark.parametrize(
    "gaudi_type",
    ["Gaudi1"],
)
def test_habana_gaudi_all_types(gaudi_type):
    """Test all Habana Gaudi types"""
    gaudi = HABANA_GAUDI(device=gaudi_type)  # type: ignore
    assert gaudi.device == gaudi_type
    assert gaudi.quantity == 1
    assert gaudi.device_class == "HABANA_GAUDI"

    res = Resources(gpu=gaudi)
    device = res.get_device()
    assert device is not None
    assert device.device == gaudi_type
    assert device.quantity == 1
    assert device.device_class == "HABANA_GAUDI"


def test_habana_gaudi_invalid_device():
    """Test Habana Gaudi with invalid device type"""
    with pytest.raises(ValueError, match="Invalid Habana Gaudi type"):
        HABANA_GAUDI(device="InvalidGaudi")  # type: ignore


# String-based accelerator tests
@pytest.mark.parametrize(
    "accelerator_string,expected_device,expected_quantity,expected_class",
    [
        ("A100:4", "A100", 4, "GPU"),
        ("T4:1", "T4", 1, "GPU"),
        ("L4:2", "L4", 2, "GPU"),
        ("Trn1:1", "Trn1", 1, "NEURON"),
        ("Inf2:1", "Inf2", 1, "NEURON"),
        ("MI300X:1", "MI300X", 1, "AMD_GPU"),
        ("Gaudi1:1", "Gaudi1", 1, "HABANA_GAUDI"),
    ],
)
def test_accelerator_strings(accelerator_string, expected_device, expected_quantity, expected_class):
    """Test accelerators using string format"""
    res = Resources(gpu=accelerator_string)  # type: ignore
    device = res.get_device()
    assert device is not None
    assert device.device == expected_device
    assert device.quantity == expected_quantity
    assert device.device_class == expected_class


def test_gpu_type_accelerators_synchronization():
    """
    Test that all GPU types in GPUType exist in Accelerators and vice versa.

    This ensures that when new GPU types are added to one definition,
    they are also added to the other to maintain consistency.
    """
    # Extract all GPU types from GPUType literal
    gpu_types = set(get_args(GPUType))

    # Extract all accelerator strings from Accelerators literal
    accelerators = get_args(Accelerators)

    # Extract unique GPU device names from Accelerators (before the ":" separator)
    # Filter only GPU types (exclude Neuron, AMD_GPU, HABANA_GAUDI, TPU)
    neuron_types = set(get_args(NeuronType))
    amd_gpu_types = set(get_args(AMD_GPUType))
    habana_gaudi_types = set(get_args(HABANA_GAUDIType))
    tpu_types = set(get_args(TPUType))

    # Collect GPU device names from Accelerators
    gpu_accelerators = set()
    for acc in accelerators:
        if ":" in acc:
            device_name = acc.split(":")[0]
            # Only include if it's not a Neuron, AMD_GPU, HABANA_GAUDI, or TPU type
            if (
                device_name not in neuron_types
                and device_name not in amd_gpu_types
                and device_name not in habana_gaudi_types
                and device_name not in tpu_types
            ):
                gpu_accelerators.add(device_name)

    # Check that all GPUType entries exist in Accelerators
    missing_in_accelerators = gpu_types - gpu_accelerators
    assert not missing_in_accelerators, (
        f"GPU types in GPUType but missing in Accelerators: {missing_in_accelerators}. "
        f"Please add entries like '{next(iter(missing_in_accelerators))}:1' through "
        f"'{next(iter(missing_in_accelerators))}:8' to Accelerators."
    )

    # Check that all GPU entries in Accelerators exist in GPUType
    missing_in_gpu_type = gpu_accelerators - gpu_types
    assert not missing_in_gpu_type, (
        f"GPU types in Accelerators but missing in GPUType: {missing_in_gpu_type}. "
        f"Please add these types to the GPUType Literal definition."
    )

    # Verify we found expected types (sanity check)
    assert "H100" in gpu_types, "Expected H100 to be in GPUType"
    assert "H200" in gpu_types, "Expected H200 to be in GPUType"
    assert "H100" in gpu_accelerators, "Expected H100 to be in Accelerators"
    assert "H200" in gpu_accelerators, "Expected H200 to be in Accelerators"


def test_neuron_type_accelerators_synchronization():
    """
    Test that all Neuron types in NeuronType exist in Accelerators and vice versa.
    """
    neuron_types = set(get_args(NeuronType))
    accelerators = get_args(Accelerators)

    # Extract Neuron device names from Accelerators
    neuron_accelerators = set()
    for acc in accelerators:
        if ":" in acc:
            device_name = acc.split(":")[0]
            if device_name in neuron_types:
                neuron_accelerators.add(device_name)

    # Check synchronization
    missing_in_accelerators = neuron_types - neuron_accelerators
    assert not missing_in_accelerators, (
        f"Neuron types in NeuronType but missing in Accelerators: {missing_in_accelerators}"
    )

    missing_in_neuron_type = neuron_accelerators - neuron_types
    assert not missing_in_neuron_type, (
        f"Neuron types in Accelerators but missing in NeuronType: {missing_in_neuron_type}"
    )


def test_amd_gpu_type_accelerators_synchronization():
    """
    Test that all AMD GPU types in AMD_GPUType exist in Accelerators and vice versa.
    """
    amd_gpu_types = set(get_args(AMD_GPUType))
    accelerators = get_args(Accelerators)

    # Extract AMD GPU device names from Accelerators
    amd_gpu_accelerators = set()
    for acc in accelerators:
        if ":" in acc:
            device_name = acc.split(":")[0]
            if device_name in amd_gpu_types:
                amd_gpu_accelerators.add(device_name)

    # Check synchronization
    missing_in_accelerators = amd_gpu_types - amd_gpu_accelerators
    assert not missing_in_accelerators, (
        f"AMD GPU types in AMD_GPUType but missing in Accelerators: {missing_in_accelerators}"
    )

    missing_in_amd_gpu_type = amd_gpu_accelerators - amd_gpu_types
    assert not missing_in_amd_gpu_type, (
        f"AMD GPU types in Accelerators but missing in AMD_GPUType: {missing_in_amd_gpu_type}"
    )


def test_habana_gaudi_type_accelerators_synchronization():
    """
    Test that all Habana Gaudi types in HABANA_GAUDIType exist in Accelerators and vice versa.
    """
    habana_gaudi_types = set(get_args(HABANA_GAUDIType))
    accelerators = get_args(Accelerators)

    # Extract Habana Gaudi device names from Accelerators
    habana_gaudi_accelerators = set()
    for acc in accelerators:
        if ":" in acc:
            device_name = acc.split(":")[0]
            if device_name in habana_gaudi_types:
                habana_gaudi_accelerators.add(device_name)

    # Check synchronization
    missing_in_accelerators = habana_gaudi_types - habana_gaudi_accelerators
    assert not missing_in_accelerators, (
        f"Habana Gaudi types in HABANA_GAUDIType but missing in Accelerators: {missing_in_accelerators}"
    )

    missing_in_habana_gaudi_type = habana_gaudi_accelerators - habana_gaudi_types
    assert not missing_in_habana_gaudi_type, (
        f"Habana Gaudi types in Accelerators but missing in HABANA_GAUDIType: {missing_in_habana_gaudi_type}"
    )
