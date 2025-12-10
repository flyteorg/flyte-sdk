import flyte

env_1 = flyte.TaskEnvironment(name="env_1", resources=flyte.Resources(cpu=1), image=flyte.Image.from_debian_base().with_commands("echo hello"))
env_2 = flyte.TaskEnvironment(name="env_2", resources=flyte.Resources(cpu=2), image=flyte.Image.from_debian_base().with_commands("echo hello"))
