import flyte

acc_env = flyte.TaskEnvironment("habana", resources=flyte.Resources(gpu="Gaudi1:1"))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":

    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
