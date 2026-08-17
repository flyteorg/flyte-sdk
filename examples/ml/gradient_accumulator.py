"""
Gradient Accumulator
=====================

Simulate a large effective batch size on limited hardware by accumulating
gradients over several micro-batches before each optimizer step.
`flyte.ml.GradientAccumulator` tracks the micro-batch count for you, so the
training loop doesn't hand-roll the modulo counter (and its off-by-one bugs)
that shows up in nearly every large-model training script.

This example simulates training without a real model or optimizer — it just
demonstrates exactly which micro-batches trigger an optimizer step.
"""

import flyte
import flyte.ml

env = flyte.TaskEnvironment(
    name="gradient_accumulator_example",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
def train(num_micro_batches: int = 20, accumulation_steps: int = 4) -> dict:
    acc = flyte.ml.GradientAccumulator(accumulation_steps=accumulation_steps)
    optimizer_steps: list[int] = []

    for micro_batch in range(1, num_micro_batches + 1):
        # loss.backward() would happen here in a real training loop.
        if acc.step():
            # optimizer.step() / optimizer.zero_grad() would happen here.
            optimizer_steps.append(micro_batch)
            print(f"[micro-batch {micro_batch}] optimizer step {len(optimizer_steps)}")
        else:
            print(f"[micro-batch {micro_batch}] accumulating ({acc.micro_step}/{accumulation_steps})")

    return {
        "num_micro_batches": num_micro_batches,
        "accumulation_steps": accumulation_steps,
        "optimizer_steps_fired_at": optimizer_steps,
        "total_optimizer_steps": len(optimizer_steps),
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train)
    print(run.url)
