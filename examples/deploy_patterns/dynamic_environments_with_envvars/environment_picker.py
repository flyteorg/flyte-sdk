import os

import flyte


def create_env(domain: str):
    """
    Deterministically create different environments based on context
    NOTE how we are passing the domain as an environment variable to the environment too, so that at runtime,
    we can see which domain we are running in.
    """
    if domain == "development":
        return flyte.TaskEnvironment(name="dev", image=flyte.Image.from_debian_base(), env_vars={"DOMAIN_NAME": domain})
    return flyte.TaskEnvironment(name="prod", image=flyte.Image.from_debian_base(), env_vars={"DOMAIN_NAME": domain})


env = create_env(os.getenv("DOMAIN_NAME", "development"))


@env.task
async def my_task(n: int) -> int:
    print(f"Environment Variable MY_ENV = {os.environ['DOMAIN_NAME']}", flush=True)
    return n + 1


@env.task
async def entrypoint(n: int) -> int:
    print(f"Environment Variable MY_ENV = {os.environ['DOMAIN_NAME']}", flush=True)
    return await my_task(n)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(entrypoint, n=5)
    print(r.url)
