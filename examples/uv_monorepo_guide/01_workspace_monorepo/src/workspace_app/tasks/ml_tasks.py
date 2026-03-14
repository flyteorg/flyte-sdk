from lib_models.baseline import predict, train_mean_predictor

from workspace_app.tasks.envs import ml_env


@ml_env.task
async def train(features: list[float], labels: list[float]) -> dict:
    """Train a simple model."""
    return train_mean_predictor(features, labels)


@ml_env.task
async def evaluate(model: dict, features: list[float]) -> float:
    """Evaluate the model on a set of features."""
    return predict(model, features)


@ml_env.task
async def ml_pipeline(features: list[float], labels: list[float]) -> float:
    """Train and evaluate a model end-to-end."""
    model = await train(features=features, labels=labels)
    return await evaluate(model=model, features=features)
