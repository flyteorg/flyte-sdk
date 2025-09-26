import flyte
import flyte.git

env = flyte.TaskEnvironment(name="live_demo", resources=flyte.Resources(memory="250Mi"))

@env.task
def square(x: int) -> int:
    return x * x

@env.task
def map_square(xs: list[int]) -> list[int]:
    return list(flyte.map(square, xs))

if __name__ == "__main__":

    # print(map_square([1, 2, 3, 4, 5]))

    flyte.init_from_config(flyte.git.config_from_root())
    run = flyte.run(map_square, xs=[1, 2, 3, 4, 5])
    print(run.url)

