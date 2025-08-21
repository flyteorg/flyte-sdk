import os

from flyteplugins.wandb.tracking import WandbConfig

import flyte

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

task_env = flyte.TaskEnvironment(name="hello_wandb")


wandb_env = flyte.TaskEnvironment(
    name="wandb_env",
    plugin_config=WandbConfig(
        wandb_api_key=WANDB_API_KEY,
        project="wandb-example",
    ),
)


@task_env.task()
def train() -> float:
    import wandb
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from wandb.integration.xgboost import WandbCallback
    from xgboost import XGBRegressor

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        callbacks=[WandbCallback(log_model=True)],
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
    )

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)

    # Log custom metrics
    wandb.log({"rmse": rmse})

    return rmse


@wandb_env.task
def hello_wandb() -> float:
    score = train()
    return score


# ## Execute locally
# You can execute the code locally as if it was a normal Python script.

if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(hello_wandb)
    print("run name:", run.name)
    print("run url:", run.url)
    run.wait(run)
