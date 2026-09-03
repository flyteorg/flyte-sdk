from __future__ import annotations

import pytest
from flyteidl2.core import execution_pb2

import flyte.errors
from flyte._internal.runtime.convert import Error, convert_error_to_native

XID_SENTENCE = "[gpu-health] [CRITICAL] Xid 79 (GPU has fallen off the bus) on GPU 3 GPU-1a2b-3c."
SXID_SENTENCE = "[gpu-health] [CRITICAL] SXid 22 on NVSwitch 0000:3b:00.0."
USER_SENTENCE = "[gpu-health] [USER] Xid 31 (GPU memory page fault) on GPU 0 GPU-abc."


def _err(code: str, kind, message: str = "", worker: str = "worker-0") -> execution_pb2.ExecutionError:
    return execution_pb2.ExecutionError(code=code, kind=kind, message=message, worker=worker)


@pytest.mark.parametrize("code", list(flyte.errors.GPU_FAULT_CODES))
def test_user_kind_gpu_code_selects_user_fault_error(code):
    exc = convert_error_to_native(_err(code, execution_pb2.ExecutionError.USER, USER_SENTENCE))

    assert isinstance(exc, flyte.errors.GPUFaultUserError)
    assert isinstance(exc, flyte.errors.RuntimeUserError)
    assert exc.code == code
    assert exc.kind == "user"
    assert exc.worker == "worker-0"


@pytest.mark.parametrize("code", list(flyte.errors.GPU_FAULT_CODES))
def test_system_kind_gpu_code_selects_system_fault_error(code):
    exc = convert_error_to_native(_err(code, execution_pb2.ExecutionError.SYSTEM, XID_SENTENCE))

    assert isinstance(exc, flyte.errors.GPUFaultSystemError)
    assert isinstance(exc, flyte.errors.RuntimeSystemError)
    assert exc.code == code
    assert exc.kind == "system"


def test_both_kinds_are_caught_by_the_one_base_class():
    for kind in (execution_pb2.ExecutionError.USER, execution_pb2.ExecutionError.SYSTEM):
        exc = convert_error_to_native(_err("GpuXidError", kind, USER_SENTENCE))
        with pytest.raises(flyte.errors.GPUFaultError) as raised:
            raise exc
        assert raised.value is exc


def test_server_injected_code_still_selects_the_gpu_error():
    exc = convert_error_to_native(
        _err("RetriesExhaustedError|GpuEccUncorrectable", execution_pb2.ExecutionError.SYSTEM, XID_SENTENCE)
    )

    assert isinstance(exc, flyte.errors.GPUFaultError)
    assert exc.code == "GpuEccUncorrectable"


def test_error_wrapper_is_unwrapped_like_any_other_failure():
    exc = convert_error_to_native(Error(err=_err("GpuFallenOffBus", execution_pb2.ExecutionError.SYSTEM, XID_SENTENCE)))

    assert isinstance(exc, flyte.errors.GPUFaultSystemError)


def test_fields_from_the_xid_sentence():
    exc = convert_error_to_native(
        _err("GpuFallenOffBus", execution_pb2.ExecutionError.SYSTEM, XID_SENTENCE + " Pod terminated.")
    )

    assert exc.fault_kind == "xid"
    assert exc.fault_code == 79
    assert exc.xid == 79
    assert exc.sxid is None
    assert exc.fault_name == "GPU has fallen off the bus"
    assert exc.severity == "critical"
    assert exc.gpu_index == 3
    assert exc.gpu_uuid == "GPU-1a2b-3c"
    assert str(exc) == XID_SENTENCE + " Pod terminated."


def test_fields_from_the_sxid_sentence():
    exc = convert_error_to_native(_err("GpuNvlinkError", execution_pb2.ExecutionError.SYSTEM, SXID_SENTENCE))

    assert exc.fault_kind == "sxid"
    assert exc.fault_code == 22
    assert exc.sxid == 22
    assert exc.xid is None
    assert exc.severity == "critical"
    assert exc.pci_bus_id == "0000:3b:00.0"
    assert exc.gpu_uuid is None


@pytest.mark.parametrize(
    "sentence, gpu_index, gpu_uuid, pci_bus_id",
    [
        ("[gpu-health] [USER] Xid 31 (GPU memory page fault) on GPU 0 GPU-abc.", 0, "GPU-abc", None),
        ("[gpu-health] [USER] Xid 31 (GPU memory page fault) on GPU GPU-abc.", None, "GPU-abc", None),
        ("[gpu-health] [USER] Xid 31 (GPU memory page fault) on GPU 2.", 2, None, None),
        ("[gpu-health] [USER] Xid 31 (GPU memory page fault) on GPU at PCI 0000:3b:00.0.", None, None, "0000:3b:00.0"),
        ("[gpu-health] [USER] Xid 31 (GPU memory page fault).", None, None, None),
    ],
)
def test_device_is_read_from_every_shape_the_sentence_takes(sentence, gpu_index, gpu_uuid, pci_bus_id):
    exc = convert_error_to_native(_err("GpuXidError", execution_pb2.ExecutionError.USER, sentence))

    assert exc.xid == 31
    assert exc.severity == "user"
    assert exc.gpu_index == gpu_index
    assert exc.gpu_uuid == gpu_uuid
    assert exc.pci_bus_id == pci_bus_id


def test_machine_readable_tail_is_read_when_the_message_carries_one():
    message = (
        "(combined from similar events): [gpu-health] [USER] Xid 13 (Graphics Engine Exception) on GPU 1 GPU-x."
        " xid=13 severity=user gpu_uuid=GPU-x gpu_index=1 pci=0000:3b:00.0 node=ip-10-0-0-1 pid=42 process=python3"
    )
    exc = convert_error_to_native(_err("GpuXidError", execution_pb2.ExecutionError.USER, message))

    assert exc.xid == 13
    assert exc.node == "ip-10-0-0-1"
    assert exc.process == "python3"
    assert exc.pci_bus_id == "0000:3b:00.0"


@pytest.mark.parametrize(
    "message",
    [
        "",
        "container exited with code 137",
        "[gpu-health] [CRITICAL] Xid but no number at all.",
    ],
)
def test_a_message_without_a_readable_sentence_still_converts(message):
    exc = convert_error_to_native(_err("GpuXidError", execution_pb2.ExecutionError.USER, message))

    assert isinstance(exc, flyte.errors.GPUFaultUserError)
    assert exc.xid is None
    assert exc.severity is None
    assert exc.gpu_uuid is None
    assert exc.node is None


@pytest.mark.parametrize(
    "code, kind, expected",
    [
        ("OOMKilled", execution_pb2.ExecutionError.USER, flyte.errors.OOMError),
        ("Interrupted", execution_pb2.ExecutionError.USER, flyte.errors.TaskInterruptedError),
        ("SomeOtherError", execution_pb2.ExecutionError.USER, flyte.errors.RuntimeUserError),
        ("SomeOtherError", execution_pb2.ExecutionError.SYSTEM, flyte.errors.RuntimeSystemError),
        ("SomeOtherError", execution_pb2.ExecutionError.UNKNOWN, flyte.errors.RuntimeUnknownError),
        ("GpuXidError", execution_pb2.ExecutionError.UNKNOWN, flyte.errors.RuntimeUnknownError),
    ],
)
def test_codes_that_are_not_gpu_faults_are_converted_as_before(code, kind, expected):
    exc = convert_error_to_native(_err(code, kind, XID_SENTENCE))

    assert type(exc) is expected
    assert not isinstance(exc, flyte.errors.GPUFaultError)


# ---------------------------------------------------------------------------------------------------------------
# The typed fault the backend attaches to the failure, which is what the attributes are read from whenever it is
# there. The sentence is only the fallback for a failure that arrives without one.
# ---------------------------------------------------------------------------------------------------------------


def test_typed_fault_fills_every_field():
    err = _err("GpuFallenOffBus", execution_pb2.ExecutionError.SYSTEM, XID_SENTENCE)
    err.gpu_fault.CopyFrom(
        execution_pb2.GpuFault(
            kind=execution_pb2.GpuFault.KIND_XID,
            code=79,
            name="GPU has fallen off the bus",
            severity=execution_pb2.GpuFault.SEVERITY_CRITICAL,
            gpu_uuid="GPU-typed",
            gpu_index=0,
            pci_bus_id="0000:3b:00.0",
            node="ip-10-0-0-7",
            pid=42,
            process="train.py",
        )
    )

    exc = convert_error_to_native(err)

    assert isinstance(exc, flyte.errors.GPUFaultSystemError)
    assert exc.fault_kind == "xid"
    assert exc.xid == 79
    assert exc.fault_name == "GPU has fallen off the bus"
    assert exc.severity == "critical"
    assert exc.gpu_uuid == "GPU-typed"
    # Read from the typed fault, not from the sentence on the same failure, which says GPU 3.
    assert exc.gpu_index == 0
    assert exc.pci_bus_id == "0000:3b:00.0"
    assert exc.node == "ip-10-0-0-7"
    assert exc.process == "train.py"


def test_typed_sxid_fault_is_not_reported_as_an_xid():
    err = _err("GpuNvlinkError", execution_pb2.ExecutionError.SYSTEM, SXID_SENTENCE)
    err.gpu_fault.CopyFrom(
        execution_pb2.GpuFault(
            kind=execution_pb2.GpuFault.KIND_SXID,
            code=22,
            severity=execution_pb2.GpuFault.SEVERITY_CRITICAL,
            pci_bus_id="0000:3b:00.0",
        )
    )

    exc = convert_error_to_native(err)

    assert exc.fault_kind == "sxid"
    assert exc.sxid == 22
    assert exc.xid is None
    # An unresolved GPU index is absent rather than zero, which is a GPU of its own.
    assert exc.gpu_index is None


def test_no_typed_fault_falls_back_to_the_sentence():
    exc = convert_error_to_native(_err("GpuXidError", execution_pb2.ExecutionError.USER, USER_SENTENCE))

    assert isinstance(exc, flyte.errors.GPUFaultUserError)
    assert exc.xid == 31
    assert exc.gpu_uuid == "GPU-abc"
    assert exc.node is None


def test_typed_fault_with_nothing_filled_in_reads_as_unknown():
    err = _err("GpuXidError", execution_pb2.ExecutionError.USER, "container exited with code 137")
    err.gpu_fault.CopyFrom(execution_pb2.GpuFault())

    exc = convert_error_to_native(err)

    assert isinstance(exc, flyte.errors.GPUFaultUserError)
    assert exc.fault_kind is None
    assert exc.xid is None
    assert exc.severity is None
