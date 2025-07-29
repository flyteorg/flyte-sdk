import pytest

from flyte._resources import GPU, TPU, Device, Resources


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
    res = Resources(gpu=Device(device="A100", quantity=1, partition="1g.5gb"))
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
