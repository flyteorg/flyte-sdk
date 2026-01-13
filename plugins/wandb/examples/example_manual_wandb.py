"""
Example: Manual W&B integration without decorators.

This example demonstrates:
- Using the Wandb link class directly with custom run IDs
- Manually initializing wandb.init() in tasks
- Adding W&B links to tasks without @wandb_init decorator
"""

import flyte
import wandb

from flyteplugins.wandb import Wandb, wandb_config

env = flyte.TaskEnvironment(
    name="wandb-manual-example",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


# Method 1: Define task first, then add link using override
@env.task
async def train_model_basic(learning_rate: float, batch_size: int) -> str:
    """Basic training task - link added via override() at runtime."""
    ctx = flyte.ctx()
    id = f"{ctx.action.run_name}-{ctx.action.name}"

    # Manually initialize wandb
    run = wandb.init(
        project="flyte-wandb-test",
        entity="samhita-alla",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        tags=["manual-init", "basic"],
        id=id,
    )

    print(f"Training with lr={learning_rate}, batch_size={batch_size}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")

    # Simulate training
    for epoch in range(5):
        loss = 1.0 / (learning_rate * batch_size) + epoch * 0.1
        run.log({"epoch": epoch, "loss": loss})

    # Manually finish the run
    run.finish()

    return run.id


# Method 2: Add link at task definition time with custom run ID
@env.task(
    links=(
        Wandb(
            project="flyte-wandb-test",
            entity="samhita-alla",
            new_run=True,
            id="my-custom-run-id",  # Provide custom run ID to the link
        ),
    )
)
async def train_model_with_link(learning_rate: float, batch_size: int) -> str:
    """Training task with W&B link and custom run ID."""
    # Manually initialize wandb with the same custom run ID
    run = wandb.init(
        project="flyte-wandb-test",
        entity="samhita-alla",
        id="my-custom-run-id",  # Must match the ID in the Wandb link
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        tags=["manual-init", "with-link"],
        resume="allow",
    )

    print(f"Training with lr={learning_rate}, batch_size={batch_size}")
    print(f"Run ID: {run.id}")
    print(f"Run URL: {run.url}")

    # Simulate training
    for epoch in range(5):
        loss = 1.0 / (learning_rate * batch_size) + epoch * 0.1
        accuracy = min(0.95, learning_rate * batch_size * (epoch + 1) * 0.01)
        run.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

    # Manually finish the run
    run.finish()

    return run.id


# Method 3: Child task that accepts custom run ID as parameter
@env.task
async def train_model_child(learning_rate: float, batch_size: int, run_id: str) -> str:
    """Child training task that uses provided run ID."""
    # Manually initialize wandb with provided run ID
    run = wandb.init(
        project="flyte-wandb-test",
        entity="samhita-alla",
        id=run_id,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
        tags=["manual-init", "child"],
    )

    print(f"Child Training with lr={learning_rate}, batch_size={batch_size}")
    print(f"Run ID: {run.id}")

    # Simulate training
    for epoch in range(3):
        loss = 1.0 / (learning_rate * batch_size) + epoch * 0.1
        run.log({"epoch": epoch, "loss": loss})

    run.finish()
    return run.id


# Method 4: Parent task that calls children, all with manual wandb init and custom IDs
@env.task(
    links=(
        Wandb(
            project="flyte-wandb-test",
            entity="samhita-alla",
            new_run=True,
            id="parent-run-custom-id",
        ),
    )
)
async def parent_training_task() -> str:
    """Parent task that orchestrates multiple training runs with custom IDs."""
    # Initialize parent run with custom ID
    parent_run = wandb.init(
        project="flyte-wandb-test",
        entity="samhita-alla",
        id="parent-run-custom-id",
        tags=["manual-init", "parent"],
    )

    print(f"Parent Run ID: {parent_run.id}")
    parent_run.log({"parent_metric": 100})

    # Call child tasks with their own custom IDs
    result1 = await train_model_child.override(
        links=(
            Wandb(
                project="flyte-wandb-test",
                entity="samhita-alla",
                id="child-1-custom-id",
            ),
        )
    )(learning_rate=0.001, batch_size=32, run_id="child-1-custom-id")

    result2 = await train_model_child.override(
        links=(
            Wandb(
                project="flyte-wandb-test",
                entity="samhita-alla",
                id="child-2-custom-id",
            ),
        )
    )(learning_rate=0.01, batch_size=64, run_id="child-2-custom-id")

    parent_run.finish()

    return f"Parent complete: child1={result1}, child2={result2}"


# Method 5: Task with link but no custom ID (auto-generated)
@env.task(
    links=(
        Wandb(
            project="flyte-wandb-test",
            entity="samhita-alla",
            new_run=True,
            # No id parameter - link will auto-generate from run_name-action_name
        ),
    )
)
async def train_model_autoid(learning_rate: float) -> str:
    """Task with link but no custom ID - Flyte auto-generates the run ID."""
    ctx = flyte.ctx()

    # Auto-generate run ID matching what the link will generate
    run_id = f"{ctx.action.run_name}-{ctx.action.name}"

    # Manually initialize wandb with auto-generated ID
    run = wandb.init(
        project="flyte-wandb-test",
        entity="samhita-alla",
        id=run_id,
        config={"learning_rate": learning_rate},
        tags=["manual-init", "auto-id"],
    )

    print(f"Auto-generated Run ID: {run.id}")

    for epoch in range(3):
        loss = 1.0 / learning_rate + epoch * 0.1
        run.log({"epoch": epoch, "loss": loss})

    run.finish()
    return run.id


if __name__ == "__main__":
    flyte.init_from_config()

    print("\n=== Method 1: Basic task with link added via override ===")
    run1 = flyte.with_runcontext(
        custom_context=wandb_config(project="flyte-wandb-test", entity="samhita-alla")
    ).run(
        train_model_basic.override(
            links=(
                Wandb(project="flyte-wandb-test", entity="samhita-alla", new_run=True),
            )
        ),
        learning_rate=0.001,
        batch_size=32,
    )
    print(f"Run URL: {run1.url}\n")

    print("=== Method 2: Task with link and custom run ID ===")
    run2 = flyte.with_runcontext(
        custom_context=wandb_config(project="flyte-wandb-test", entity="samhita-alla")
    ).run(train_model_with_link, learning_rate=0.01, batch_size=64)
    print(f"Run URL: {run2.url}\n")

    print("=== Method 3: Parent task with children (custom IDs) ===")
    run3 = flyte.with_runcontext(
        custom_context=wandb_config(project="flyte-wandb-test", entity="samhita-alla")
    ).run(parent_training_task)
    print(f"Run URL: {run3.url}\n")

    print("=== Method 4: Task with link but auto-generated ID ===")
    run4 = flyte.with_runcontext(
        custom_context=wandb_config(project="flyte-wandb-test", entity="samhita-alla")
    ).run(train_model_autoid, learning_rate=0.005)
    print(f"Run URL: {run4.url}\n")
