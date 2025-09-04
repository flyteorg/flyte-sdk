import concurrent.futures
import threading
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, Tuple, TypeVar

from flyte._task import TaskTemplate
from flyte.models import ActionID, NativeInterface

if TYPE_CHECKING:
    from flyte.remote._task import TaskDetails

from ._trace import TraceInfo

__all__ = ["Controller", "ControllerType", "TraceInfo", "create_controller", "get_controller"]

if TYPE_CHECKING:
    import concurrent.futures

ControllerType = Literal["local", "remote"]

R = TypeVar("R")


class Controller(Protocol):
    """
    Controller interface, that is used to execute tasks. The implementation of this interface,
    can execute tasks in different ways, such as locally, remotely etc.
    """

    async def submit(self, _task: TaskTemplate, *args, **kwargs) -> Any:
        """
        Submit a node to the controller asynchronously and wait for the result. This is async and will block
        the current coroutine until the result is available.
        """
        ...

    def submit_sync(self, _task: TaskTemplate, *args, **kwargs) -> concurrent.futures.Future:
        """
        This should call the async submit method above, but return a concurrent Future object that can be
        used in a blocking wait or wrapped in an async future. This is called when
          a) a synchronous task is kicked off locally,
          b) a running task (of either kind) kicks off a downstream synchronous task.
        """
        ...

    async def submit_task_ref(self, _task: "TaskDetails", *args, **kwargs) -> Any:
        """
        Submit a task reference to the controller asynchronously and wait for the result. This is async and will block
        the current coroutine until the result is available.
        """
        ...

    async def finalize_parent_action(self, action: ActionID):
        """
        Finalize the parent action. This can be called to cleanup the action and should be called after the parent
        task completes
        :param action: Action ID
        :return:
        """
        ...

    async def watch_for_errors(self): ...

    async def get_action_outputs(
        self, _interface: NativeInterface, _func: Callable, *args, **kwargs
    ) -> Tuple[TraceInfo, bool]:
        """
        This method returns the outputs of the action, if it is available.
        :param _interface: NativeInterface
        :param _func: Function name
        :param args: Arguments
        :param kwargs: Keyword arguments
        :return: TraceInfo object and a boolean indicating if the action was found.
        if boolean is False, it means the action is not found and the TraceInfo object will have only min info
        """

    async def record_trace(self, info: TraceInfo):
        """
        Record a trace action. This is used to record the trace of the action and should be called when the action
        is completed.
        :param info: Trace information
        :return:
        """
        ...

    async def stop(self):
        """
        Stops the engine and should be called when the engine is no longer needed.
        """
        ...


# Internal state holder
class _ControllerState:
    controller: Optional[Controller] = None
    lock = threading.Lock()


def get_controller() -> Controller:
    """
    Get the controller instance. Raise an error if it has not been created.
    """
    if _ControllerState.controller is not None:
        return _ControllerState.controller
    raise RuntimeError("Controller is not initialized. Please call create_controller() first.")


def create_controller(
    ct: ControllerType,
    **kwargs,
) -> Controller:
    """
    Create a new instance of the controller, based on the kind and the given configuration.
    """
    controller: Controller
    match ct:
        case "local":
            from ._local_controller import LocalController

            controller = LocalController()
        case "remote" | "hybrid":
            from flyte._internal.controllers.remote import create_remote_controller

            controller = create_remote_controller(**kwargs)
        case _:
            raise ValueError(f"{ct} is not a valid controller type.")

    with _ControllerState.lock:
        _ControllerState.controller = controller
        return controller
