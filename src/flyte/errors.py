"""
Exceptions raised by Union.

These errors are raised when the underlying task execution fails, either because of a user error, system error or an
unknown error.
"""

import re
from typing import Any, Literal

ErrorKind = Literal["system", "unknown", "user"]


def silence_polling_error(loop, context):
    """
    Suppress specific polling errors in the event loop.
    """
    exc = context.get("exception")
    if isinstance(exc, BlockingIOError):
        return  # suppress
    loop.default_exception_handler(context)


class BaseRuntimeError(RuntimeError):
    """
    Base class for all Union runtime errors. These errors are raised when the underlying task execution fails, either
    because of a user error, system error or an unknown error.
    """

    def __init__(self, code: str, kind: ErrorKind, root_cause_message: str, worker: str | None = None):
        super().__init__(root_cause_message)
        self.code = code
        self.kind = kind
        self.worker = worker

    def _reraise(self, *_args):
        """Re-raise this error when user code mistakenly treats it as a value.

        When `flyte.map` is called with `return_exceptions=True`, exceptions are
        returned as values. If user code then performs arithmetic on them (e.g.
        `sum(results)`), this surfaces the *real* subtask error instead of a
        confusing `TypeError`.
        """
        raise self

    __add__ = _reraise
    __radd__ = _reraise
    __sub__ = _reraise
    __rsub__ = _reraise
    __mul__ = _reraise
    __rmul__ = _reraise
    __truediv__ = _reraise
    __rtruediv__ = _reraise
    __floordiv__ = _reraise
    __rfloordiv__ = _reraise


class InitializationError(BaseRuntimeError):
    """
    This error is raised when the Union system is tried to access without being initialized.
    """


class RuntimeSystemError(BaseRuntimeError):
    """
    This error is raised when the underlying task execution fails because of a system error. This could be a bug in the
    Union system or a bug in the user's code.
    """

    def __init__(self, code: str, message: str, worker: str | None = None):
        super().__init__(code, "system", message, worker)


class UnionRpcError(RuntimeSystemError):
    """
    This error is raised when communication with the Union server fails.
    """


class RuntimeUserError(BaseRuntimeError):
    """
    This error is raised when the underlying task execution fails because of an error in the user's code.
    """

    def __init__(self, code: str, message: str, worker: str | None = None):
        super().__init__(code, "user", message, worker)


class RuntimeUnknownError(BaseRuntimeError):
    """
    This error is raised when the underlying task execution fails because of an unknown error.
    """

    def __init__(self, code: str, message: str, worker: str | None = None):
        super().__init__(code, "unknown", message, worker)


class OOMError(RuntimeUserError):
    """
    This error is raised when the underlying task execution fails because of an out-of-memory error.
    """


GPU_FAULT_CODES: tuple[str, ...] = (
    "GpuXidError",
    "GpuFallenOffBus",
    "GpuEccUncorrectable",
    "GpuRowRemapPending",
    "GpuNvlinkError",
    "GpuGspError",
)
"""
The error codes the backend puts on a failure it attributed to a GPU or NVSwitch fault. GpuXidError is the catch-all
for a fault with no more specific code, the rest name a class of trouble a user or an operator can act on. Any of them
converts to a GPU fault error in the SDK.
"""


class GPUFaultError(BaseRuntimeError):
    """
    This error is raised when the backend attributed the task failure to a GPU or NVSwitch fault that the GPU health
    daemon observed on the node, such as an Xid 31 (a GPU memory page fault) or an Xid 79 (the GPU fell off the bus).

    Catch this class to handle every GPU fault. It is the base of both concrete errors, GPUFaultUserError for a fault
    the workload caused and GPUFaultSystemError for a hardware fault, so one except clause covers both, and the code,
    severity and xid attributes are there to branch on afterwards.

    The two do not reach user code on the same terms. A user severity Xid (13, 31, 43, 45) is the workload's own
    doing, it will fault again if it is replayed unchanged, so the backend charges it to the task's own retry budget
    and this error surfaces as soon as that budget is spent. A critical hardware fault is not the workload's doing, so
    the platform retries it without charging the user's budget and reschedules onto other hardware where it can, which
    means user code sees a critical fault only after platform policy has given up on it. Neither one is a signal to
    retry in place: a user fault has already exhausted its own retries by the time it is raised, and a critical fault
    has already been retried elsewhere.

    This exception and its fields appear when the platform classified the fault and supplied the typed fault data with
    the failure. On a platform or a version that did not, the same failure arrives as a generic runtime error with no
    fault attributes on it, so user code must not depend on this exception firing. Write the handler for the case where
    it does, and keep whatever handles an ordinary task failure for the case where it does not.

    The fault attributes are read from the typed fault and are absent when the failure carried none, so any of them can
    be None and they should be read defensively. For a failure from a backend that predates the typed fault, see
    parse_gpu_fault_message, which recovers what it can from the message text on request.
    """

    def __init__(
        self,
        code: str,
        kind: ErrorKind,
        message: str,
        worker: str | None = None,
        *,
        fault_kind: str | None = None,
        fault_code: int | None = None,
        fault_name: str | None = None,
        severity: str | None = None,
        gpu_uuid: str | None = None,
        gpu_index: int | None = None,
        node: str | None = None,
        pci_bus_id: str | None = None,
        process: str | None = None,
    ):
        # Named explicitly rather than through super(): the concrete errors below mix this class with RuntimeUserError
        # and RuntimeSystemError, whose own initializers fix the kind and take one argument fewer.
        BaseRuntimeError.__init__(self, code, kind, message, worker)
        self.fault_kind = fault_kind
        self.fault_code = fault_code
        self.fault_name = fault_name
        self.severity = severity
        self.gpu_uuid = gpu_uuid
        self.gpu_index = gpu_index
        self.node = node
        self.pci_bus_id = pci_bus_id
        self.process = process

    @property
    def xid(self) -> int | None:
        """
        The NVIDIA Xid number of the fault, or None when the fault was an NVSwitch SXid or when the number could not
        be determined. Xid and SXid numbers share a numbering space but not a meaning, so a number alone never
        identifies a fault, read fault_kind together with fault_code to tell them apart.
        """
        return self.fault_code if self.fault_kind == "xid" else None

    @property
    def sxid(self) -> int | None:
        """
        The NVSwitch SXid number of the fault, or None when the fault was a GPU Xid or when the number could not be
        determined.
        """
        return self.fault_code if self.fault_kind == "sxid" else None


class GPUFaultUserError(GPUFaultError, RuntimeUserError):
    """
    This error is raised when the GPU fault the backend attributed the failure to was the workload's own doing, for
    example an out-of-bounds access that the driver reported as an Xid 31. The GPU itself is fine once the process is
    gone, so the failure was charged to the task's own retry budget.
    """

    def __init__(self, code: str, message: str, worker: str | None = None, **fault):
        GPUFaultError.__init__(self, code, "user", message, worker, **fault)


class GPUFaultSystemError(GPUFaultError, RuntimeSystemError):
    """
    This error is raised when the GPU fault the backend attributed the failure to condemned the device or the node,
    for example an uncorrectable ECC error or a GPU that fell off the bus. The workload did not cause it, so the
    platform retried the task on its own budget before this error reached user code.
    """

    def __init__(self, code: str, message: str, worker: str | None = None, **fault):
        GPUFaultError.__init__(self, code, "system", message, worker, **fault)


# The sentence the GPU health daemon writes when it attributes a failure to a GPU fault, for example
# "[gpu-health] [CRITICAL] Xid 79 (GPU has fallen off the bus) on GPU 3 GPU-1a2b." or
# "[gpu-health] [CRITICAL] SXid 22 on NVSwitch 0000:3b:00.0.". The trailing full stop is matched only where a space or
# the end of the message follows it, so a PCI bus id keeps its own dots.
_GPU_FAULT_SENTENCE = re.compile(
    r"\[gpu-health\]\s+\[(?P<severity>[A-Za-z]+)\]\s+"
    r"(?:Xid\s+(?P<xid>\d+)\s+\((?P<name>[^)]*)\)|SXid\s+(?P<sxid>\d+))"
    r"(?P<where>.*?)\.(?=\s|$)"
)

# The tail of the sentence naming the device the fault happened on, in the four shapes the daemon renders it in. An
# NVSwitch SXid names the switch by bus id, a GPU Xid names the GPU by index and UUID and degrades to whichever of the
# three it could resolve.
_GPU_FAULT_LOCATION = re.compile(
    r"^\s+on\s+(?:NVSwitch\s+(?P<switch_pci>\S+)"
    r"|GPU\s+(?:at\s+PCI\s+(?P<pci>\S+)"
    r"|(?P<index>\d+)\s+(?P<indexed_uuid>\S+)"
    r"|(?P<uuid>\D\S*)"
    r"|(?P<bare_index>\d+)))$"
)

# The keys of the k=v tail the GPU health daemon writes after the sentence. A message that carries one names the node
# and the process, which the sentence never does.
_GPU_FAULT_TAIL_KEYS = frozenset({"xid", "sxid", "severity", "gpu_uuid", "gpu_index", "pci", "node", "process"})


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_gpu_fault_tail(rest: str) -> dict[str, Any]:
    """
    Read the k=v tail that follows the sentence when the whole event message was carried over, ignoring every token
    that is not one of the daemon's own keys.
    """
    tail: dict[str, Any] = {}
    for token in rest.split():
        key, sep, value = token.partition("=")
        if not sep or key not in _GPU_FAULT_TAIL_KEYS or not value:
            continue
        if key in ("xid", "sxid"):
            tail["fault_kind"] = key
            tail["fault_code"] = _int_or_none(value)
        elif key == "gpu_index":
            tail["gpu_index"] = _int_or_none(value)
        elif key == "pci":
            tail["pci_bus_id"] = value
        else:
            tail[key] = value
    return {k: v for k, v in tail.items() if v is not None}


def parse_gpu_fault_message(message: str) -> dict[str, Any] | None:
    """
    Read whatever a human-readable failure message says about the GPU fault behind it, and return it as the keyword
    arguments GPUFaultError takes, for example {"fault_kind": "xid", "fault_code": 79, "severity": "critical"}.
    Returns None when the message says nothing this can read.

    This is best-effort parsing of prose. The wording it looks for is what the GPU health daemon happens to write
    today, it is not a contract, and nothing keeps a future version of the daemon or of the platform from changing it,
    at which point this returns None for messages it used to read. Nothing in the SDK calls it: a GPU fault error gets
    its attributes from the typed fault the platform attaches to the failure, and where there is no typed fault the
    attributes stay None. This exists so that a caller talking to a backend that predates the typed fault can still
    recover the Xid and the device from the message, knowing what it is worth.

        exc = ...  # a GPUFaultError with no attributes on it
        fields = flyte.errors.parse_gpu_fault_message(str(exc)) or {}
        xid = fields.get("fault_code") if fields.get("fault_kind") == "xid" else None
    """
    match = _GPU_FAULT_SENTENCE.search(message or "")
    if match is None:
        return None

    is_sxid = match.group("sxid") is not None
    fields: dict[str, Any] = {
        "fault_kind": "sxid" if is_sxid else "xid",
        "fault_code": _int_or_none(match.group("sxid") if is_sxid else match.group("xid")),
        "fault_name": match.group("name") or None,
        "severity": (match.group("severity") or "").lower() or None,
    }

    location = _GPU_FAULT_LOCATION.match(match.group("where") or "")
    if location is not None:
        fields["gpu_uuid"] = location.group("indexed_uuid") or location.group("uuid")
        fields["gpu_index"] = _int_or_none(location.group("index") or location.group("bare_index"))
        fields["pci_bus_id"] = location.group("pci") or location.group("switch_pci")

    fields.update(_parse_gpu_fault_tail(message[match.end() :]))
    fields = {k: v for k, v in fields.items() if v is not None}
    return fields or None


class TaskInterruptedError(RuntimeUserError):
    """
    This error is raised when the underlying task execution is interrupted.
    """


class PrimaryContainerNotFoundError(RuntimeUserError):
    """
    This error is raised when the primary container is not found.
    """


class TaskTimeoutError(RuntimeUserError):
    """
    This error is raised when the underlying task execution runs for longer than the specified timeout.
    """

    def __init__(self, message: str):
        super().__init__("TaskTimeoutError", message, "user")


class ConditionTimedoutError(RuntimeUserError):
    """
    This error is raised when a condition is not signaled within its specified timeout.
    """

    def __init__(self, message: str):
        super().__init__("ConditionTimedoutError", message, "user")


class RetriesExhaustedError(RuntimeUserError):
    """
    This error is raised when the underlying task execution fails after all retries have been exhausted.
    """


class InvalidImageNameError(RuntimeUserError):
    """
    This error is raised when the image name is invalid.
    """


class ImagePullBackOffError(RuntimeUserError):
    """
    This error is raised when the image cannot be pulled.
    """


class CustomError(RuntimeUserError):
    """
    This error is raised when the user raises a custom error.
    """

    def __init__(self, code: str, message: str):
        super().__init__(code, message, "user")

    @classmethod
    def from_exception(cls, e: Exception):
        """
        Create a CustomError from an exception. The exception's class name is used as the error code and the exception
        message is used as the error message.
        """
        new_exc = cls(e.__class__.__name__, str(e))
        new_exc.__cause__ = e
        return new_exc


class NotInTaskContextError(RuntimeUserError):
    """
    This error is raised when the user tries to access the task context outside of a task.
    """


class ActionNotFoundError(RuntimeError):
    """
    This error is raised when the user tries to access an action that does not exist.
    """


class RemoteTaskNotFoundError(RuntimeUserError):
    """
    This error is raised when the user tries to access a task that does not exist.
    """

    CODE = "RemoteTaskNotFoundError"

    def __init__(self, message: str):
        super().__init__(self.CODE, message, "user")


class RemoteTaskUsageError(RuntimeUserError):
    """
    This error is raised when the user tries to access a task that does not exist.
    """

    CODE = "RemoteTaskUsageError"

    def __init__(self, message: str):
        super().__init__(self.CODE, message, "user")


class LogsNotYetAvailableError(BaseRuntimeError):
    """
    This error is raised when the logs are not yet available for a task.
    """

    def __init__(self, message: str):
        super().__init__("LogsNotYetAvailable", "system", message, None)


class RuntimeDataValidationError(RuntimeUserError):
    """
    This error is raised when the user tries to access a resource that does not exist or is invalid.
    """

    def __init__(self, var: str, e: Exception | str, task_name: str = ""):
        super().__init__(
            "DataValidationError", f"In task {task_name} variable {var}, failed to serialize/deserialize because of {e}"
        )


class DeploymentError(RuntimeUserError):
    """
    This error is raised when the deployment of a task fails, or some preconditions for deployment are not met.
    """

    def __init__(self, message: str):
        super().__init__("DeploymentError", message, "user")


class ImageBuildError(RuntimeUserError):
    """
    This error is raised when the image build fails.
    """

    def __init__(self, message: str):
        super().__init__("ImageBuildError", message, "user")


class ModuleLoadError(RuntimeUserError):
    """
    This error is raised when the module cannot be loaded, either because it does not exist or because of a
     syntax error.
    """

    def __init__(self, message: str):
        super().__init__("ModuleLoadError", message, "user")


class InlineIOMaxBytesBreached(RuntimeUserError):
    """
    This error is raised when the inline IO max bytes limit is breached.
    This can be adjusted per task by setting max_inline_io_bytes in the task definition.
    """

    def __init__(self, message: str):
        super().__init__("InlineIOMaxBytesBreached", message, "user")


class ActionAbortedError(RuntimeUserError):
    """
    This error is raised when an action was aborted, externally. The parent action will raise this error.
    """

    def __init__(self, message: str):
        super().__init__("ActionAbortedError", message, "user")


class SlowDownError(RuntimeUserError):
    """
    This error is raised when the user tries to access a resource that does not exist or is invalid.
    """

    def __init__(self, message: str):
        super().__init__("SlowDownError", message, "user")


class ResourceExhaustedError(SlowDownError):
    pass


class OnlyAsyncIOSupportedError(RuntimeUserError):
    """
    This error is raised when the user tries to use sync IO in an async task.
    """

    def __init__(self, message: str):
        super().__init__("OnlyAsyncIOSupportedError", message, "user")


class ParameterMaterializationError(RuntimeUserError):
    """
    This error is raised when the user tries to use a Parameter in an App, that has delayed Materialization,
    but the materialization fails.
    """

    def __init__(self, message: str):
        super().__init__("ParameterMaterializationError", message, "user")


class RestrictedTypeError(RuntimeUserError):
    """
    This error is raised when the user uses a restricted type, for example current a Tuple is not supported for one
     value.
    """

    def __init__(self, message: str):
        super().__init__("RestrictedTypeUsage", message, "user")


class CodeBundleError(RuntimeUserError):
    """
    This error is raised when the code bundle cannot be created, for example when no files are found to bundle.
    """

    def __init__(self, message: str):
        super().__init__("CodeBundleError", message, "user")


class SyncTaskCallInAsyncContextError(RuntimeUserError):
    """
    This error is raised when a sync task is invoked in a blocking way (`task(...)`) from inside an async
    task. That call would block the event loop that drives the parent task — the same loop the runtime uses
    to watch the controller for failures — so a controller/informer outage would leave the process stuck
    forever. Use `await task.aio(...)` instead.
    """

    def __init__(self, message: str):
        super().__init__("SyncTaskCallInAsyncContextError", message, "user")


class TraceDoesNotAllowNestedTasksError(RuntimeUserError):
    """
    This error is raised when the user tries to use a task from within a trace. Tasks can be nested under tasks
    not traces.
    """

    def __init__(self, message: str):
        super().__init__("TraceDoesNotAllowNestedTasksError", message)


class InvalidPackageError(RuntimeUserError):
    """Raised when an invalid system package is detected during image build."""

    def __init__(self, package_name: str, original_error: str):
        self.package_name = package_name
        self.original_error = original_error
        super().__init__(
            "InvalidPackageError",
            f"Invalid system package detected: '{package_name}'. "
            f"This package does not exist in apt repositories. "
            f"Error: {original_error}",
        )


class NonRecoverableError(RuntimeUserError):
    """
    Raised when an error is encountered that is not recoverable. Retries are irrelevant.
    """

    def __init__(self, message: str, code: str = "NonRecoverableError"):
        super().__init__(code, message)


class ConditionAlreadyExistsError(RuntimeUserError):
    """
    This error is raised when the user tries to create a condition that already exists within the action.
    """

    def __init__(self, message: str):
        super().__init__("ConditionAlreadyExistsError", message, "user")


class ConditionFailedError(RuntimeUserError):
    """
    This error is raised when a condition fails during execution.

    This can happen when the backend encounters an error while processing the condition,
    or when the condition is explicitly marked as failed by the system.
    """

    def __init__(self, message: str):
        super().__init__("ConditionFailedError", message, "user")


class ConditionNotFoundError(RuntimeUserError):
    """
    This error is raised when the user tries to access a condition that does not exist.
    """

    def __init__(self, message: str):
        super().__init__("ConditionNotFoundError", message, "user")
