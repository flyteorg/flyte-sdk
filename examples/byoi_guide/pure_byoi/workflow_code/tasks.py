from workflow_code.envs import env_data, env_train


@env_train.task
async def train(processed: str) -> float:
    # Runs in Team B's container: python 3.10, WORKDIR /workspace
    # torch is available (installed in training/Dockerfile)
    return float(len(processed))


@env_data.task
async def prepare(raw: str = "Hello World") -> float:
    # Runs in Team A's container: python 3.11, WORKDIR /app
    # pandas is available (installed in data_prep/Dockerfile)
    processed = raw.strip().lower()
    return await train(processed)
