import fastapi
import uvicorn

import flyte
import flyte.app
import flyte.io
from flyte.app.extras import FastAPIAppEnvironment

image = flyte.Image.from_debian_base().with_pip_packages("scikit-learn", "joblib", "fastapi", "uvicorn")

task_env = flyte.TaskEnvironment(
    name="pickled-model-serving-tasks",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

app = fastapi.FastAPI()
env = FastAPIAppEnvironment(
    name="pickled-fastapi-app",
    app=app,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    port=8080,
    parameters=[
        flyte.app.Parameter(
            name="model",
            value=flyte.app.RunOutput(task_name="pickled-model-serving-tasks.train_model", type="file"),
            download=True,
        )
    ],
)


@task_env.task
def train_model() -> flyte.io.File:
    # Train a model
    import joblib
    import sklearn.datasets
    import sklearn.linear_model

    dummy_data = sklearn.datasets.make_regression(n_features=10, n_samples=100, random_state=42)
    X = dummy_data[0]
    y = dummy_data[1]

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "./model.joblib")
    return flyte.io.File.from_local_sync("./model.joblib")


@env.server
async def fastapi_app_server(model: flyte.io.File):
    import joblib

    model = joblib.load(model.path)
    app.state.model = model

    await uvicorn.Server(uvicorn.Config(app, port=8080)).serve()


@app.post("/predict")
async def predict(x: list[float]) -> float:
    return app.state.model.predict([x])[0]


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)

    run = flyte.run(train_model)
    print(f"training model run: {run.url}")
    run.wait()

    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(app.url)
