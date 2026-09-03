"""
Exceptions raised by Union.

These errors are raised when the underlying task execution fails, either because of a user error, system error or an
unknown error.
"""

from typing import Literal

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

    The fault attributes are best effort. They are read from the typed fault the backend attaches to the failure, and
    where there is none, from the sentence the backend prepends to the failure message, which does not carry every
    attribute. Any of them can be None, so read them defensively.
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
