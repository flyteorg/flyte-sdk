"""
Gradient accumulation bookkeeping for training loops.

`flyte.ml.GradientAccumulator` tracks how many micro-batches have been processed and
tells you exactly when to call `optimizer.step()`, so you don't hand-roll the modulo
counter (and its off-by-one bugs) in every training script. It has no framework
dependency (no torch) — it only counts; you still call `backward()`/`step()`/
`zero_grad()` yourself.

Example:

    acc = flyte.ml.GradientAccumulator(accumulation_steps=4)
    for micro_batch in loader:
        loss = compute_loss(micro_batch) / acc.accumulation_steps
        loss.backward()
        if acc.step():
            optimizer.step()
            optimizer.zero_grad()
"""

from __future__ import annotations


class GradientAccumulator:
    """
    Tracks micro-batch counts and signals when to run an optimizer step.

    Args:
        accumulation_steps: Number of micro-batches to accumulate gradients over
            before an optimizer step is due. Must be >= 1 (1 means every micro-batch
            triggers a step, i.e. no accumulation).
    """

    def __init__(self, accumulation_steps: int) -> None:
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.accumulation_steps = accumulation_steps
        self.micro_step: int = 0

    def step(self) -> bool:
        """
        Record one micro-batch. Returns True if `accumulation_steps` micro-batches
        have now been accumulated and an optimizer step is due (the internal counter
        resets automatically in that case).
        """
        self.micro_step += 1

        if self.micro_step >= self.accumulation_steps:
            self.micro_step = 0
            return True
        return False

    @property
    def is_accumulating(self) -> bool:
        """True if at least one micro-batch has been accumulated since the last optimizer step."""
        return self.micro_step > 0

    def reset(self) -> None:
        """Clear the micro-batch counter, as if no `step` had ever been called."""
        self.micro_step = 0
