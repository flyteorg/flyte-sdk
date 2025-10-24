import flyte.app
from flyte.models import SerializationContext

try:
    import fastapi
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "fastapi is not installed. Please install the 'fastapi', to use FastAPI apps."
    )


class FastAPIAppEnvironment(flyte.app.AppEnvironment):
    app: fastapi.FastAPI
    type: str = "FastAPI"

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        return ["uvicorn", f"{module_name}:{app_name}", "--port", str(self.port.port)]

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []
