import importlib.util
import inspect
import pathlib
import sys
from dataclasses import dataclass
from typing import Tuple

import rich.repr

import flyte.app
from flyte.models import SerializationContext

try:
    import fastapi
except ModuleNotFoundError:
    raise ModuleNotFoundError("fastapi is not installed. Please install the 'fastapi', to use FastAPI apps.")


def _extract_fastapi_app_module_and_var(
    app: fastapi.FastAPI, caller_frame: inspect.FrameInfo | None, serialization_context: SerializationContext
) -> Tuple[str, str]:
    """
    Extract the module name and variable name for a FastAPI app instance.

    This function solves the challenge that `inspect.getmodule(app)` returns the
    `fastapi.applications` module (where the FastAPI class is defined) rather than
    the user's module (where the `app` variable is defined). Since FastAPI apps are
    instances rather than classes or functions, we need special handling to locate
    the correct module and variable name.

    The function uses the caller frame (captured when FastAPIAppEnvironment was
    instantiated) to determine which file contains the app definition, then inspects
    that module's globals to find FastAPI instances.

    Args:
        app: The FastAPI application instance to locate.
        caller_frame: Frame information from where FastAPIAppEnvironment was instantiated.
                     If None, falls back to extract_obj_module (which may not work correctly).
        serialization_context: Context containing the root directory for calculating
                              relative module paths.

    Returns:
        A tuple of (module_name, variable_name) where:
        - module_name: Dotted module path (e.g., "examples.apps.single_script_fastapi")
        - variable_name: The name of the variable holding the FastAPI app (e.g., "app")

    Raises:
        RuntimeError: If the module cannot be loaded or the app variable cannot be found.

    Example:
        >>> frame = inspect.getframeinfo(inspect.currentframe().f_back)
        >>> module_name, var_name = _extract_fastapi_app_module_and_var(
        ...     app, frame, serialization_context
        ... )
        >>> # Returns: ("examples.apps.my_app", "app")
        >>> # Can be used as: uvicorn examples.apps.my_app:app
    """
    if caller_frame is None:
        raise RuntimeError("Caller frame cannot be None")

    # Get the file path where the app was defined
    file_path = pathlib.Path(caller_frame.filename)

    # Calculate module name relative to source_dir
    try:
        relative_path = file_path.relative_to(serialization_context.root_dir or pathlib.Path("."))
        module_name = pathlib.Path(relative_path).with_suffix("").as_posix().replace("/", ".")
    except ValueError:
        # File is not relative to source_dir, use the stem
        module_name = file_path.stem

    # Instead of reloading the module, inspect the caller frame's local variables
    # The app variable should be in the frame's globals
    caller_globals = None

    # Try to get globals from the main module if it matches our file
    if hasattr(sys.modules.get("__main__"), "__file__"):
        main_file = pathlib.Path(sys.modules["__main__"].__file__ or ".").resolve()
        if main_file == file_path.resolve():
            caller_globals = sys.modules["__main__"].__dict__

    if caller_globals is None:
        # Load the module to inspect it
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        caller_globals = module.__dict__

    # Extract variable name from module - look for FastAPI instances
    app_var_name = None
    for var_name, obj in caller_globals.items():
        if isinstance(obj, fastapi.FastAPI):
            # Found a FastAPI app - this is likely the one we want
            # Store the first one we find
            if app_var_name is None:
                app_var_name = var_name
            # If the objects match by identity, use this one
            if obj is app:
                app_var_name = var_name
                break

    if app_var_name is None:
        raise RuntimeError("Could not find variable name for FastAPI app in module")

    return module_name, app_var_name


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class FastAPIAppEnvironment(flyte.app.AppEnvironment):
    app: fastapi.FastAPI
    type: str = "FastAPI"
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.app is None:
            raise ValueError("app cannot be None for FastAPIAppEnvironment")
        if not isinstance(self.app, fastapi.FastAPI):
            raise TypeError(f"app must be of type fastapi.FastAPI, got {type(self.app)}")

        # Capture the frame where this environment was instantiated
        # This helps us find the module where the app variable is defined
        frame = inspect.currentframe()
        if frame and frame.f_back:
            # Go up the call stack to find the user's module
            # Skip the dataclass __init__ frame
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """
        Generate the container arguments for running the FastAPI app with uvicorn.

        Returns:
            A list of command arguments in the format:
            ["uvicorn", "<module_name>:<app_var_name>", "--port", "<port>"]
        """
        module_name, app_var_name = _extract_fastapi_app_module_and_var(
            self.app, self._caller_frame, serialization_context
        )

        p = self.port
        assert isinstance(p, flyte.app.Port)
        return ["uvicorn", f"{module_name}:{app_var_name}", "--port", str(p.port)]

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
