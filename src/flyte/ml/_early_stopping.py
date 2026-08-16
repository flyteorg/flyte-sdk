"""
Early stopping for long-running training loops.

`flyte.ml.EarlyStopping` tracks a metric across training steps/epochs and reports when
training has stopped improving, so a task can stop early instead of running to
`total_epochs` unconditionally. It has no framework dependency (no torch/sklearn) —
call `.step(value)` with whatever metric your training loop already computes.

Example:

    early_stop = flyte.ml.EarlyStopping(patience=5, mode="min")
    for epoch in range(num_epochs):
        val_loss = train_one_epoch(...)
        if early_stop.step(val_loss):
            print(f"Stopping at epoch {epoch}, best={early_stop.best} at step {early_stop.best_step}")
            break
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["min", "max"]


class EarlyStopping:
    """
    Tracks a metric across steps and signals when it has stopped improving.

    Args:
        patience: Number of consecutive non-improving steps allowed before stopping.
        mode: `"min"` if lower metric values are better (e.g. loss), `"max"` if higher
            is better (e.g. accuracy).
        min_delta: Minimum change from the current best to count as an improvement.
            Must be non-negative; compared against the raw difference between the new
            value and the current best, regardless of mode.
    """

    def __init__(self, patience: int, mode: Mode = "min", min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best: float | None = None
        self.best_step: int = -1
        self.num_bad_steps: int = 0
        self.should_stop: bool = False
        self._step_count: int = -1

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        """
        Record a new metric value. Returns True if `patience` consecutive steps have
        passed without improvement (i.e. training should stop now).
        """
        self._step_count += 1

        if self._is_improvement(value):
            self.best = value
            self.best_step = self._step_count
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        self.should_stop = self.num_bad_steps >= self.patience
        return self.should_stop

    def reset(self) -> None:
        """Clear all tracked state, as if no `step` had ever been called."""
        self.best = None
        self.best_step = -1
        self.num_bad_steps = 0
        self.should_stop = False
        self._step_count = -1
