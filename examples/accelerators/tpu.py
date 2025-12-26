import flyte

acc_env = flyte.TaskEnvironment("tpu", resources=flyte.Resources(gpu=flyte.TPU("V5P", "2x2x1")))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
