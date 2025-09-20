import flyte
import flyte.errors

#
# PATH_TO_FASTTASK_WORKER = pathlib.Path("../../../private/flyte/fasttask/worker-v2")
#
# actor_image = (
#     flyte.Image.from_debian_base(install_flyte=False)
#     .with_apt_packages("curl", "build-essential", "ca-certificates", "pkg-config", "libssl-dev")
#     .with_commands(["sh -c 'curl https://sh.rustup.rs -sSf | sh -s -- -y'"])
#     .with_env_vars({"PATH": "/root/.cargo/bin:${PATH}"})
#     .with_source_file(pathlib.Path(".dockerignore"))
#     .with_source_folder(PATH_TO_FASTTASK_WORKER, "/root/fasttask")
#     .with_pip_packages("uv")
#     .with_workdir("/root/fasttask")
#     .with_commands(["uv sync --reinstall --active"])
#     .with_local_v2()
# )

actor_image = flyte.Image.from_debian_base().with_pip_packages("unionai-reuse==0.1.3")

env = flyte.TaskEnvironment(
    name="error_reusable",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    reusable=flyte.ReusePolicy(
        replicas=2,  # Min of 2 replacas are needed to ensure no-starvation of tasks.
        idle_ttl=60,
    ),
    image=actor_image,
)


@env.task
async def raise_err(x: int) -> int:
    if x == 30:
        # Simulate an error for demonstration purposes.
        raise ValueError(x)
    return x


@env.task
async def main(n: int) -> int:
    """
    Run square and cube tasks in parallel for the range of x_list.
    """
    try:
        return await raise_err(n)
    except flyte.errors.RuntimeUserError as e:
        print(f"Caught exception: {e}, of type {type(e)}, {e.code}")
        if e.code == "ValueError":
            print(f"Handling ValueError specifically for {n}")
            # Handle the specific error case here, e.g., retry with a different value.
            return await raise_err(n + 1)
        raise
    return n


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, n=30)  # run remotely inline and pass data.
    print(run.url)
