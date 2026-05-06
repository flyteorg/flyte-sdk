"""
Async Training with Periodic Evaluation and Early Stopping
===========================================================

This example demonstrates a common ML pattern:

1. A long-running **training task** runs asynchronously, writing checkpoints
   periodically to a shared location.
2. The **orchestrator** sleeps (using `flyte.durable.sleep`) between evaluation
   rounds, then reads the latest checkpoint and launches a **batch inference
   (evaluation) task**.
3. If the evaluation detects convergence, the training task is **cancelled**.
4. If the training task fails at any point, the entire pipeline fails and any
   in-progress evaluation is also cancelled.

Key Flyte concepts used:
- `asyncio.create_task` for concurrent execution
- `flyte.durable.sleep.aio` for crash-resilient sleeping
- `@flyte.trace` for checkpointed sub-steps
- `asyncio.CancelledError` for graceful cancellation
- `flyte.io.File` for checkpoint passing between tasks
"""

import asyncio
import json
import math
import random

import flyte
import flyte.durable
import flyte.errors
from flyte.io import File

env = flyte.TaskEnvironment(
    name="async_train_eval",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


# ---------------------------------------------------------------------------
# Training task
# ---------------------------------------------------------------------------
@env.task
async def train(checkpoint_dir: str, total_epochs: int, seconds_per_epoch: float) -> File:
    """Long-running training loop that writes checkpoints to *checkpoint_dir*.

    Each checkpoint is a small JSON file containing the epoch number and the
    current (simulated) loss.  The loss follows a noisy exponential decay so
    that convergence is eventually detected by the evaluator.

    Returns the final checkpoint file so the orchestrator can run one last eval.
    """
    print(f"[train] Starting training for {total_epochs} epochs")

    ckpt_file: File | None = None
    for epoch in range(1, total_epochs + 1):
        # Simulate one epoch of work
        await asyncio.sleep(seconds_per_epoch)

        # Simulated loss: exponential decay + noise
        base_loss = math.exp(-0.08 * epoch)
        noise = random.uniform(-0.02, 0.02)
        loss = max(base_loss + noise, 0.001)

        # Write checkpoint
        checkpoint = {"epoch": epoch, "loss": round(loss, 6)}
        ckpt_file = File.from_existing_remote(f"{checkpoint_dir}/checkpoint.json")
        async with ckpt_file.open("wb") as fh:
            await fh.write(json.dumps(checkpoint).encode())

        print(f"[train] Epoch {epoch}/{total_epochs}  loss={loss:.4f}")

    assert ckpt_file is not None
    print("[train] Training completed all epochs")
    return ckpt_file


# ---------------------------------------------------------------------------
# Evaluation / batch inference task
# ---------------------------------------------------------------------------
@env.task
async def evaluate(checkpoint_file: File, eval_round: int, convergence_loss: float) -> bool:
    """Read a checkpoint and decide whether the model has converged.

    Returns True if the loss is below the convergence threshold.
    """
    async with checkpoint_file.open("rb") as fh:
        raw = bytes(await fh.read())
    checkpoint = json.loads(raw.decode())

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    # Simulate some batch-inference work
    await asyncio.sleep(1)

    converged = loss < convergence_loss
    status = "CONVERGED" if converged else "not yet"
    print(f"[eval round {eval_round}] epoch={epoch}  loss={loss:.4f}  -> {status}")
    return converged


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@env.task
async def main(
    total_epochs: int = 50,
    seconds_per_epoch: float = 0.5,
    convergence_loss: float = 0.05,
    eval_interval_seconds: float = 5.0,
    max_eval_rounds: int = 20,
) -> str:
    """Orchestrate training, periodic evaluation, and early stopping.

    Args:
        checkpoint_dir: Remote path prefix where checkpoints are written.
        total_epochs: Number of training epochs.
        seconds_per_epoch: Simulated duration of one epoch.
        convergence_loss: Loss threshold below which we declare convergence.
        eval_interval_seconds: How long the orchestrator sleeps between evals.
        max_eval_rounds: Safety cap on evaluation rounds.

    Flow:
      1. Launch training as a background asyncio task.
      2. Periodically sleep, read the latest checkpoint, and run evaluation.
      3. If evaluation says "converged" -> cancel training, return success.
      4. If training fails -> cancel any in-progress eval, propagate error.
      5. If training finishes naturally -> run one final eval on the returned
         checkpoint.
    """
    checkpoint_dir = flyte.ctx().run_base_dir
    # Start training in the background
    train_task = asyncio.create_task(train(checkpoint_dir, total_epochs, seconds_per_epoch))

    eval_task: asyncio.Task | None = None

    try:
        for round_num in range(1, max_eval_rounds + 1):
            with flyte.group(f"eval-round-{round_num}"):
                # Durable sleep — survives crashes and restarts.
                # Race it against the training task so we wake up immediately
                # if training finishes (or fails) during the sleep interval.
                sleep_task = asyncio.create_task(flyte.durable.sleep.aio(eval_interval_seconds))
                done, _ = await asyncio.wait(
                    [train_task, sleep_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if train_task in done:
                    sleep_task.cancel()
                    # .result() re-raises if training failed — the except
                    # block below will cancel any in-flight eval and propagate.
                    final_ckpt: File = train_task.result()
                    break

                # Read latest checkpoint
                ckpt_file = File.from_existing_remote(f"{checkpoint_dir}/checkpoint.json")

                # Run evaluation as its own asyncio task so we can cancel it
                # if training fails while eval is running.
                eval_task = asyncio.create_task(
                    evaluate(ckpt_file, eval_round=round_num, convergence_loss=convergence_loss)
                )

                # Wait for eval to complete, but also watch for training failure
                done, _ = await asyncio.wait(
                    [train_task, eval_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if train_task in done:
                    # Training ended while eval was running — cancel eval,
                    # then use the final checkpoint for a clean last eval.
                    eval_task.cancel()
                    final_ckpt = train_task.result()
                    break

                # Eval completed — check result
                converged = eval_task.result()
                eval_task = None

                if converged:
                    print("[main] Convergence detected! Cancelling training.")
                    train_task.cancel()
                    try:
                        await train_task
                    except asyncio.CancelledError:
                        pass
                    return "converged_early"
        else:
            # Exhausted all eval rounds without convergence — wait for training
            print("[main] Max eval rounds reached. Waiting for training to finish.")
            final_ckpt = await train_task

        # Training completed — run one final eval on the last checkpoint
        print("[main] Training done. Running final evaluation.")
        converged = await evaluate(final_ckpt, eval_round=round_num + 1, convergence_loss=convergence_loss)
        return "converged" if converged else "completed_without_convergence"

    except Exception as exc:
        # Training or some other step failed — cancel everything
        print(f"[main] Error detected: {exc}")
        train_task.cancel()
        if eval_task is not None and not eval_task.done():
            eval_task.cancel()
        raise


if __name__ == "__main__":
    flyte.init_from_config()
    result = flyte.run(main)
    print(result.url)
