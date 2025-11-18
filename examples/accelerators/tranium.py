import flyte

acc_env = flyte.TaskEnvironment("trn", resources=flyte.Resources(gpu="Trn1:1"))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":

    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
