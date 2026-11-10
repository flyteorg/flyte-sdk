import flyte

# AMD GPUs take 1 to 8 per task, e.g. gpu="MI355X:8" or gpu=flyte.AMD_GPU("MI355X", 8).
acc_env = flyte.TaskEnvironment("amd", resources=flyte.Resources(gpu="MI350X:1"))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
