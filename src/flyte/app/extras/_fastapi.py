from inspect import getmembers

import flyte.app
from flyte._module import extract_obj_module
from flyte.models import SerializationContext

try:
    import fastapi
except ModuleNotFoundError:
    raise ModuleNotFoundError("fastapi is not installed. Please install the 'fastapi', to use FastAPI apps.")


class FastAPIAppEnvironment(flyte.app.AppEnvironment):
    app: fastapi.FastAPI
    type: str = "FastAPI"

    def __post_init__(self):
        super().__post_init__()
        if self.app is None:
            raise ValueError("app cannot be None for FastAPIAppEnvironment")
        if not isinstance(self.app, fastapi.FastAPI):
            raise TypeError(f"app must be of type fastapi.FastAPI, got {type(self.app)}")

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        module_name, module = extract_obj_module(self.app, source_dir=serialization_context.root_dir)
        # extract variable name from module
        for var_name, obj in getmembers(module):
            if obj is self.app:
                app_var_name = var_name
                break
        else:  # no break
            raise RuntimeError("Could not find variable name for FastAPI app in module")
        p = self.port
        assert isinstance(p, flyte.app.Port)
        return ["uvicorn", f"{module_name}:{app_var_name}", "--port", str(p.port)]

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
