"""
Early Stopping
==============

Stop a training loop as soon as a validation metric stops improving, instead of
always running to `max_epochs`. `flyte.ml.EarlyStopping` tracks the metric you
report each step and tells you when `patience` consecutive steps have passed
without improvement.

This example simulates training with a noisy, decaying loss so it converges
quickly and demonstrates stopping well before `max_epochs` is reached.
"""

import random

import flyte
import flyte.ml

env = flyte.TaskEnvironment(
    name="early_stopping_example",
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
    early_stop = flyte.ml.EarlyStopping(patience=patience, mode="min")

    history = []
    for epoch in range(max_epochs):
        val_loss = simulated_val_loss(epoch)
        history.append({"epoch": epoch, "val_loss": val_loss})
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}")

        if early_stop.step(val_loss):
            print(f"Stopping early at epoch {epoch}: best={early_stop.best:.4f} at epoch {early_stop.best_step}")
            break
    else:
        print(f"Completed all {max_epochs} epochs without triggering early stopping")

    return {
        "stopped_early": early_stop.should_stop,
        "epochs_run": len(history),
        "best_val_loss": early_stop.best,
        "best_epoch": early_stop.best_step,
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
