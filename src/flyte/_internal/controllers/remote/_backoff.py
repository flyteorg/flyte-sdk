import random


def jittered(delay: float) -> float:
    """A wait of between half of `delay` and all of it.

    Spreads out retries that would otherwise land together: after a dropped
    connection, every watch and enqueue that failed with it backs off by the
    same steps and, without jitter, all retry at the same instant.
    """
    return random.uniform(delay / 2, delay)
