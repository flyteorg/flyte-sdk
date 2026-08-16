"""
Metric Tracker
==============

Track a metric across training steps and report running statistics — mean, min,
max, and the best value seen so far — without deciding whether to stop.
`flyte.ml.MetricTracker` pairs naturally with `flyte.ml.EarlyStopping`: the tracker
reports what happened, the early-stopping helper decides when to act on it.

This example simulates training with a noisy, decaying loss and prints the running
statistics each epoch, then uses `flyte.ml.EarlyStopping` alongside the tracker to
stop once the loss plateaus.
"""

import random

import flyte
import flyte.ml

env = flyte.TaskEnvironment(
    name="metric_tracker_example",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


def simulated_val_loss(epoch: int) -> float:
    """A validation loss that decays then plateaus, with a little noise."""
    plateau_epoch = 8
    base = 1.0 / (1 + min(epoch, plateau_epoch))
    noise = random.uniform(-0.01, 0.01)
    return max(base + noise, 0.01)


@env.task
def train(max_epochs: int = 50, patience: int = 5) -> dict:
    tracker = flyte.ml.MetricTracker(mode="min")
    early_stop = flyte.ml.EarlyStopping(patience=patience, mode="min")

    for epoch in range(max_epochs):
        val_loss = simulated_val_loss(epoch)
        tracker.update(val_loss)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} mean={tracker.mean:.4f} best={tracker.best:.4f}")

        if early_stop.step(val_loss):
            print(f"Stopping early at epoch {epoch}: best={tracker.best:.4f} at epoch {tracker.best_step}")
            break
    else:
        print(f"Completed all {max_epochs} epochs without triggering early stopping")

    return {
        "stopped_early": early_stop.should_stop,
        "epochs_run": tracker.count,
        "mean_val_loss": tracker.mean,
        "best_val_loss": tracker.best,
        "best_epoch": tracker.best_step,
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
