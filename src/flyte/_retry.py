"""
Retry policy for Flyte tasks.

A retry is a fresh attempt at executing a failed action. ``RetryStrategy.count``
is the number of *user* retries; system retries (network, container, k8s) are
governed by the platform and are not subject to this policy.

User retries can be paced by an optional :class:`Backoff` policy. Without a
backoff, retries fire back-to-back. With a backoff, the n-th retry (0-indexed)
is delayed by ``min(base * factor**n, cap)``.

Retries are *not* triggered when user code raises
:class:`flyte.errors.NonRecoverableError` — that exception is the explicit
opt-out: "this failure is terminal, do not retry, even if attempts remain."
"""

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional


@dataclass(frozen=True)
class Backoff:
    """
    Exponential backoff policy applied between user retries.

    The delay before the n-th retry (0-indexed) is::

        min(base * factor**n, cap)

    :param base: Initial delay before the first retry. Must be >= 0.
    :param factor: Per-retry multiplier. ``1.0`` yields constant delay
                   (``base`` for every retry); ``2.0`` doubles each time. Must
                   be >= 1.0.
    :param cap: Upper bound on the computed delay. Required when ``factor > 1``
                to prevent unbounded growth. Must be >= 0 when set.
    """

    base: timedelta
    factor: float = 1.0
    cap: Optional[timedelta] = None

    def __post_init__(self):
        if self.base < timedelta(0):
            raise ValueError(f"Backoff.base must be >= 0, got {self.base}")
        if not isinstance(self.factor, (int, float)) or math.isnan(self.factor) or math.isinf(self.factor):
            raise ValueError(f"Backoff.factor must be a finite number, got {self.factor!r}")
        if self.factor < 1.0:
            raise ValueError(f"Backoff.factor must be >= 1.0, got {self.factor}")
        if self.cap is not None and self.cap < timedelta(0):
            raise ValueError(f"Backoff.cap must be >= 0, got {self.cap}")
        if self.factor > 1.0 and self.cap is None:
            raise ValueError("Backoff.cap is required when factor > 1.0 to prevent unbounded growth")

    def compute_delay(self, n: int) -> timedelta:
        """
        Returns the delay for the n-th retry (0-indexed).

        Used by the local controller to pace local retries; the remote leasor
        applies the same formula.
        """
        if n < 0:
            raise ValueError(f"Retry index n must be >= 0, got {n}")
        delay = self.base * (self.factor**n)
        if self.cap is not None and delay > self.cap:
            return self.cap
        return delay


@dataclass
class RetryStrategy:
    """
    Retry strategy for a task.

    :param count: Number of user retries. ``count=0`` disables retries.
    :param backoff: Optional :class:`Backoff` policy applied between retries.
                    When unset, retries fire immediately back-to-back.

    Examples::

        # Plain count, no pacing.
        @env.task(retries=5)
        async def call_api(): ...

        # Exponential backoff: 10s, 20s, 40s, 80s, capped at 5m.
        @env.task(
            retries=flyte.RetryStrategy(
                count=5,
                backoff=flyte.Backoff(
                    base=timedelta(seconds=10),
                    factor=2.0,
                    cap=timedelta(minutes=5),
                ),
            ),
        )
        async def call_api_with_backoff(): ...
    """

    count: int
    backoff: Optional[Backoff] = None
