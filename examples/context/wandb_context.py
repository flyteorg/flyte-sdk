import os

import wandb

import flyte
from flyte.link import Wandb

env = flyte.TaskEnvironment("wandb-context-example", image=flyte.Image.from_debian_base().with_pip_packages("wandb"))

WANDB_PROJECT = "my_project"
WANDB_ENTITY = "my_entity"


@env.task()
async def leaf_task():
    # Reads run-level context
    with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=flyte.ctx().custom_context.get("wandb_id")):
        wandb.run.log({"test_score": 99})


@env.task(link=Wandb(project="my_project", entity="my_entity"))
async def root():
    action_id = os.getenv("_F_PN", "default_id")
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=action_id,
    )
    run.notes = "hello flyte"
    wandb.run.log({"test_score": 99})

    with flyte.custom_context(wandb_id=action_id):
        await leaf_task()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(root)
    print(run.url)
