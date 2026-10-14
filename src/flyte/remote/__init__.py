"""
Remote Entities that are accessible from the Union Server once deployed or created.
"""

from typing import TYPE_CHECKING

__all__ = [
    "Action",
    "ActionDetails",
    "ActionInputs",
    "ActionOutputs",
    "App",
    "Artifact",
    "Condition",
    "PartitionSchema",
    "Project",
    "Run",
    "RunDetails",
    "Secret",
    "SecretTypes",
    "Settings",
    "Task",
    "TaskDetails",
    "TimeFilter",
    "Trigger",
    "TriggerDetails",
    "User",
    "auth_metadata",
    "upload_dir",
    "upload_file",
]

if TYPE_CHECKING:
    from ._action import Action, ActionDetails, ActionInputs, ActionOutputs
    from ._app import App
    from ._artifact import Artifact, PartitionSchema
    from ._auth_metadata import auth_metadata
    from ._common import TimeFilter
    from ._condition import Condition
    from ._data import upload_dir, upload_file
    from ._project import Project
    from ._run import Run, RunDetails
    from ._secret import Secret, SecretTypes
    from ._settings import Settings
    from ._task import Task, TaskDetails
    from ._trigger import Trigger, TriggerDetails
    from ._user import User

# Public name -> submodule that defines it, resolved on first access (PEP 562). Importing anything
# under `flyte.remote` runs this file, and that includes the client every task container builds at
# startup, so the entities are not imported until something asks for them.
_EXPORTS = {
    "Action": "_action",
    "ActionDetails": "_action",
    "ActionInputs": "_action",
    "ActionOutputs": "_action",
    "App": "_app",
    "Artifact": "_artifact",
    "Condition": "_condition",
    "PartitionSchema": "_artifact",
    "Project": "_project",
    "Run": "_run",
    "RunDetails": "_run",
    "Secret": "_secret",
    "SecretTypes": "_secret",
    "Settings": "_settings",
    "Task": "_task",
    "TaskDetails": "_task",
    "TimeFilter": "_common",
    "Trigger": "_trigger",
    "TriggerDetails": "_trigger",
    "User": "_user",
    "auth_metadata": "_auth_metadata",
    "upload_dir": "_data",
    "upload_file": "_data",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
