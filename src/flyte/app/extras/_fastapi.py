import flyte.app

try:
    import fastapi
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "fastapi is not installed. Please install the 'fastapi', to use FastAPI apps."
    )

class FastAPIAppEnvironment(flyte.app.AppEnvironment):
    app: fastapi.FastAPI
    type: str = "FastAPI"

    def final_command(self):
        return ["uvicorn", f"{module_name}:{app_name}", "--port", str(self.port.port)]