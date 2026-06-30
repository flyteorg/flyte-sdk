from tasks.envs import env_data, env_train


@env_train.task
async def train(processed: str) -> float:
    # Runs in /opt/venv — torch and numpy available from Team B's base image
    return float(len(processed))


@env_data.task
async def prepare(raw: str = "Hello World") -> float:
    # Runs in conda env — pandas and pyarrow available from Team A's base image
    processed = raw.strip().lower()
    return await train(processed)
