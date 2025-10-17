"""
Exceptions raised by Union.

These errors are raised when the underlying task execution fails, either because of a user error, system error or an
unknown error.
"""

from typing import Literal

ErrorKind = Literal["system", "unknown", "user"]


def silence_grpc_polling_error(loop, context):
    """
    Suppress specific gRPC polling errors in the event loop.
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


class ReferenceTaskError(RuntimeUserError):
    """
    This error is raised when the user tries to access a task that does not exist.
    """

    def __init__(self, message: str):
        super().__init__("ReferenceTaskUsageError", message, "user")


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


class RunAbortedError(RuntimeUserError):
    """
    This error is raised when the run is aborted by the user.
    """

    def __init__(self, message: str):
        super().__init__("RunAbortedError", message, "user")


class SlowDownError(RuntimeUserError):
    """
    This error is raised when the user tries to access a resource that does not exist or is invalid.
    """

    def __init__(self, message: str):
        super().__init__("SlowDownError", message, "user")


class OnlyAsyncIOSupportedError(RuntimeUserError):
    """
    This error is raised when the user tries to use sync IO in an async task.
    """

    def __init__(self, message: str):
        super().__init__("OnlyAsyncIOSupportedError", message, "user")
