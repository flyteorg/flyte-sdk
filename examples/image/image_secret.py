import flyte
from flyte import Image, SecretMount, Secret

image = (
    Image.from_debian_base(install_flyte=False)
    .with_apt_packages("vim", secret_mounts=[SecretMount("/tmp/secret1")])
    .with_pip_packages("mypy", secret_mounts=[SecretMount(Secret(group="aws", key="id", as_env_var="AWS_ACCESS_KEY_ID"))])
    .with_local_v2()
)

env = flyte.TaskEnvironment(name="image_secret", image=image)


@env.task
async def t1(data: str = "hello") -> str:
    return f"Hello {data}"


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    run = flyte.run(t1, data="world")
    print(run.name)
    print(run.url)
