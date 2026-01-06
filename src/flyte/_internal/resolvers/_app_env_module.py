import pathlib
import sys

from flyte._logging import logger
from flyte._module import extract_obj_module
from flyte.app._app_environment import AppEnvironment


def extract_app_env_module(app_env: AppEnvironment, /, source_dir: pathlib.Path) -> tuple[str, str]:
    """
    Extract the module name and variable name for a AppEnvironment instance.

    Args:
        app_env: The AppEnvironment instance to locate.
        caller_frame: Frame information from where AppEnvironment was instantiated.
                     If None, falls back to extract_obj_module (which may not work correctly).
        serialization_context: Context containing the root directory for calculating
                              relative module paths.

    Returns:
        A tuple of (module_name, variable_name) where:
        - module_name: Dotted module path (e.g., "examples.apps.single_script_fastapi")
        - variable_name: The name of the variable holding the AppEnvironment (e.g., "env")

    Raises:
        RuntimeError: If the module cannot be loaded or the app variable cannot be found.

    Example:
        >>> frame = inspect.getframeinfo(inspect.currentframe().f_back)
        >>> module_name, var_name = _extract_app_env_module_and_var(
        ...     app_env, frame, serialization_context
        ... )
        >>> # Returns: ("examples.apps.my_app", "env")
        >>> # Can be used as: fserve examples.apps.my_app:env
    """
    if app_env._caller_frame is None:
        raise RuntimeError("Caller frame cannot be None")

    # Get the file path where the app was defined
    file_path = pathlib.Path(app_env._caller_frame.filename)

    # Calculate module name relative to source_dir
    try:
        relative_path = file_path.relative_to(source_dir or pathlib.Path("."))
        logger.info(f"Relative path: {relative_path}, {source_dir} {pathlib.Path('.')}")
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
        # Load the module to inspect it by importing it by name
        # Note: we can't use extract_obj_module here because it uses inspect.getmodule()
        # which returns the module where the CLASS is defined, not where the INSTANCE is created
        import importlib
        entity_module = importlib.import_module(module_name)
        caller_globals = entity_module.__dict__

    # Extract variable name from module - look for AppEnvironment instances
    app_var_name = None
    for var_name, obj in caller_globals.items():
        if isinstance(obj, AppEnvironment):
            # Found a AppEnvironment - this is likely the one we want
            # Store the first one we find
            if app_var_name is None:
                app_var_name = var_name
            # If the objects match by identity, use this one
            if obj is app_env:
                app_var_name = var_name
                break

    if app_var_name is None:
        raise RuntimeError("Could not find variable name for FastAPI app in module")

    return app_var_name, module_name
