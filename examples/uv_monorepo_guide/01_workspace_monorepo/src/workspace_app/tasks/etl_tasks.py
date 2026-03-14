from lib_transforms.ops import normalize

from workspace_app.tasks.envs import etl_env


@etl_env.task
async def load_data(n: int) -> list[float]:
    """Simulate loading raw data."""
    return [float(i * 1.5) for i in range(n)]


@etl_env.task
async def transform_data(raw: list[float]) -> list[float]:
    """Normalize raw data."""
    return normalize(raw)


@etl_env.task
async def etl_pipeline(n: int) -> list[float]:
    """Load and normalize data end-to-end."""
    raw = await load_data(n=n)
    return await transform_data(raw=raw)
