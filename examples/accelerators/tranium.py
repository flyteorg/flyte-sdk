import flyte

idl2 = "git+https://github.com/flyteorg/flyte.git@jeev/accelerator-device-class#subdirectory=gen/python"
image = (
    flyte.Image.from_debian_base(install_flyte=False).with_apt_packages("git").with_pip_packages(idl2).with_local_v2()
)

trn_env = flyte.TaskEnvironment("trn", resources=flyte.Resources(gpu="Trn1:1"), image=image)
env = flyte.TaskEnvironment("base", depends_on=[trn_env], image=image)


@trn_env.task
async def trn() -> str:
    return "Hello from trn"


@env.task
async def main():
    return await trn()


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
