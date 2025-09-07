from dataclasses import dataclass, field
from typing import Any, Optional

from flyteidl2.core import interface_pb2

from flyte.models import ActionID, NativeInterface


@dataclass
class TraceInfo:
    """
    Trace information for the action. This is used to record the trace of the action and should be called when
     the action is completed.
    """

    name: str
    action: ActionID
    interface: NativeInterface
    inputs_path: str
    start_time: float = field(init=False, default=0.0)
    end_time: float = field(init=False, default=0.0)
    output: Optional[Any] = None
    error: Optional[Exception] = None
    typed_interface: Optional[interface_pb2.TypedInterface] = None

    def add_outputs(self, output: Any, start_time: float, end_time: float):
        """
        Add outputs to the trace information.
        :param output: Output of the action
        :param start_time: Start time of the action
        :param end_time: End time of the action
        :return:
        """
        self.output = output
        self.start_time = start_time
        self.end_time = end_time

    def add_error(self, error: Exception, start_time: float, end_time: float):
        """
        Add error to the trace information.
        :param error: Error of the action
        :param start_time: Start time of the action
        :param end_time: End time of the action
        :return:
        """
        self.error = error
        self.start_time = start_time
        self.end_time = end_time
