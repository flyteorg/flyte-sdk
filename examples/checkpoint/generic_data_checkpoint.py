"""
Generic byte checkpointing
"""

from __future__ import annotations

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_generic_json",
    image=flyte.Image.from_debian_base(),
)

RETRIES = 3


# Define a task to iterate precisely `n_iterations`, checkpoint its state, and recover from simulated failures.
@env.task(retries=RETRIES)
def use_checkpoint(n_iterations: int) -> int:
    assert n_iterations > RETRIES
    checkpoint = flyte.ctx().checkpoint

    path = checkpoint.load_sync()
    prev = None if path is None else path.read_bytes()

    start = 0
    if prev:
        start = int(prev.decode())

    failure_interval = n_iterations // RETRIES
    for index in range(start, n_iterations):
        if index > start and index % failure_interval == 0:
            # Simulate a failure at a regular interval
            raise RuntimeError(f"Failed at iteration {index}, failure_interval {failure_interval}.")
        checkpoint.save_sync(f"{index + 1}".encode())
    return index


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote").run(use_checkpoint, n_iterations=10)
    print(run.url)
