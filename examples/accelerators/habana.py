import flyte

acc_env = flyte.TaskEnvironment("habana", resources=flyte.Resources(gpu="Gaudi1:1"))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
