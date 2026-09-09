from __future__ import annotations

import pytest

import flyte.errors
from flyte.errors import parse_gpu_fault_message

XID_SENTENCE = "[gpu-health] [CRITICAL] Xid 79 (GPU has fallen off the bus) on GPU 3 GPU-1a2b-3c."
SXID_SENTENCE = "[gpu-health] [CRITICAL] SXid 22 on NVSwitch 0000:3b:00.0."


def test_fields_from_the_xid_sentence():
    fields = parse_gpu_fault_message(XID_SENTENCE + " Pod terminated.")

    assert fields == {
        "fault_kind": "xid",
        "fault_code": 79,
        "fault_name": "GPU has fallen off the bus",
        "severity": "critical",
        "gpu_index": 3,
        "gpu_uuid": "GPU-1a2b-3c",
    }


def test_fields_from_the_sxid_sentence():
    fields = parse_gpu_fault_message(SXID_SENTENCE)

    assert fields == {
        "fault_kind": "sxid",
        "fault_code": 22,
        "severity": "critical",
        # The trailing full stop of the sentence is not eaten out of the bus id.
        "pci_bus_id": "0000:3b:00.0",
    }


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
    fields = parse_gpu_fault_message(sentence)

    assert fields["fault_kind"] == "xid"
    assert fields["fault_code"] == 31
    assert fields["severity"] == "user"
    assert fields.get("gpu_index") == gpu_index
    assert fields.get("gpu_uuid") == gpu_uuid
    assert fields.get("pci_bus_id") == pci_bus_id


def test_machine_readable_tail_is_read_when_the_message_carries_one():
    message = (
        "(combined from similar events): [gpu-health] [USER] Xid 13 (Graphics Engine Exception) on GPU 1 GPU-x."
        " xid=13 severity=user gpu_uuid=GPU-x gpu_index=1 pci=0000:3b:00.0 node=ip-10-0-0-1 pid=42 process=python3"
    )

    fields = parse_gpu_fault_message(message)

    assert fields["fault_code"] == 13
    # The node and the process are only ever in the tail, the sentence never names them.
    assert fields["node"] == "ip-10-0-0-1"
    assert fields["process"] == "python3"
    assert fields["pci_bus_id"] == "0000:3b:00.0"


@pytest.mark.parametrize(
    "message",
    [
        "",
        "container exited with code 137",
        "[gpu-health] [CRITICAL] Xid but no number at all.",
        "xid=13 severity=user node=ip-10-0-0-1",
    ],
)
def test_a_message_with_no_readable_sentence_returns_nothing(message):
    assert parse_gpu_fault_message(message) is None


def test_the_parsed_fields_are_the_keywords_the_error_takes():
    fields = parse_gpu_fault_message(XID_SENTENCE)

    exc = flyte.errors.GPUFaultSystemError("GpuFallenOffBus", XID_SENTENCE, "worker-0", **fields)

    assert exc.xid == 79
    assert exc.sxid is None
    assert exc.severity == "critical"
    assert exc.gpu_uuid == "GPU-1a2b-3c"
