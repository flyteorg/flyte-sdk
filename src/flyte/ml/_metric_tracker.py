"""
Running metric tracking for training loops.

`flyte.ml.MetricTracker` records a metric across training steps/epochs and reports
running statistics (mean, min, max, best-so-far) without deciding whether to stop —
pair it with `flyte.ml.EarlyStopping` when you need that decision too. It has no
framework dependency (no torch/sklearn) — call `.update(value)` with whatever metric
your training loop already computes.

Example:

    tracker = flyte.ml.MetricTracker(mode="min")
    for epoch in range(num_epochs):
        val_loss = train_one_epoch(...)
        tracker.update(val_loss)
        print(f"epoch {epoch}: loss={val_loss:.4f} mean={tracker.mean:.4f} best={tracker.best:.4f}")
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["min", "max"]


class MetricTracker:
    """
    Tracks a metric across steps and reports running statistics.

    Args:
        mode: `"min"` if lower metric values are better (e.g. loss), `"max"` if higher
            is better (e.g. accuracy). Determines what `best` and `best_step` track.
    """

    def __init__(self, mode: Mode = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.mode = mode
        self.history: list[float] = []
        self.best: float | None = None
        self.best_step: int = -1

    @property
    def count(self) -> int:
        """Number of values recorded so far."""
        return len(self.history)

    @property
    def last(self) -> float | None:
        """Most recently recorded value, or None if nothing has been recorded yet."""
        return self.history[-1] if self.history else None

    @property
    def mean(self) -> float | None:
        """Mean of all recorded values, or None if nothing has been recorded yet."""
        return sum(self.history) / len(self.history) if self.history else None

    @property
    def min(self) -> float | None:
        """Minimum of all recorded values, or None if nothing has been recorded yet."""
        return min(self.history) if self.history else None

    @property
    def max(self) -> float | None:
        """Maximum of all recorded values, or None if nothing has been recorded yet."""
        return max(self.history) if self.history else None

    def update(self, value: float) -> None:
        """Record a new metric value and refresh the running statistics."""
        step = len(self.history)
        self.history.append(value)

        if self.best is None:
            self.best = value
            self.best_step = step
        elif self.mode == "min" and value < self.best:
            self.best = value
            self.best_step = step
        elif self.mode == "max" and value > self.best:
            self.best = value
            self.best_step = step

    def reset(self) -> None:
        """Clear all tracked state, as if no `update` had ever been called."""
        self.history = []
        self.best = None
        self.best_step = -1
