import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="file")


@env.task
async def write_file(s: str) -> File:
    """
    This task writes a string to a file and returns the file object.
    """
    f = File.new_remote()
    async with f.open("wb") as fh:
        fh.write(s.encode("utf-8"))
    return f


@env.task
async def print_file(f: File) -> None:
    async with f.open("rb") as fh:
        contents = fh.read()
        print(f"File {f.path} contents: {contents.decode('utf-8')}")


@env.task
async def main() -> None:
    file = await write_file("Hello, Flyte!")
    await print_file(file)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    print(flyte.run(main))
