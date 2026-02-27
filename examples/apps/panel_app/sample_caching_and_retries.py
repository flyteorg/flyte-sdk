import random

import flyte

env = flyte.TaskEnvironment(name="caching-retries-example")


# Retry the task if it fails.
@env.task(retries=10)
def flaky_lookup(user_id: int) -> str:
    # Fail sometimes to demonstrate automatic retries.
    if random.random() < 0.75:
        msg = f"Transient upstream error for user {user_id}"
        print(msg)
        raise RuntimeError(msg)
    return f"user-{user_id}"


# Cache results to avoid re-running tasks if the inputs are the same.
@env.task(cache="auto")
def cached_compute(user: str) -> dict[str, str]:
    return {
        "user": user,
        "email": f"{user}@example.com",
        "phone": "+1234567890",
    }


@env.task
def main(user_id: int) -> dict[str, str]:
    # Re-running with the same inputs reuses cached results when available.
    return cached_compute(user=flaky_lookup(user_id=user_id))
