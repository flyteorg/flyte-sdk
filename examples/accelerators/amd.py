import flyte

acc_env = flyte.TaskEnvironment("amd", resources=flyte.Resources(gpu="MI350X:1"))


@acc_env.task
async def main() -> str:
    return "Hello from trn"


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
