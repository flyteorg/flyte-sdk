import flyte
from flyte.link import Wandb

env = flyte.TaskEnvironment("wandb-context-example")


@env.task(link=Wandb(project="my_project", entity="my_entity"))
async def leaf_task() -> str:
    # Reads run-level context
    print("leaf sees:", flyte.ctx().custom_context)
    return flyte.ctx().custom_context.get("trace_id")


@env.task
async def root() -> str:
    return await leaf_task()


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(custom_context={"trace_id": "root-abc", "experiment": "v1"}).run(root)
    print(run.url)
