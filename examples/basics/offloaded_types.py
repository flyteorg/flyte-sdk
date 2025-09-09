import asyncio
import tempfile
from pathlib import Path

import flyte
from flyte.io import Dir, File

env = flyte.TaskEnvironment(name="examples-offloaded")


@env.task
async def write_file(name: str) -> File:
    with open("test.txt", "w") as f:  # noqa: ASYNC230
        f.write(f"hello world {name}")
    # Users must manually handle I/O
    uploaded_file_obj = await File.from_local("test.txt")
    return uploaded_file_obj


@env.task
async def write_and_check_files() -> Dir:
    coros = []
    for name in ["Alice", "Bob", "Eve"]:
        coros.append(write_file(name=name))

    vals = await asyncio.gather(*coros)
    temp_dir = tempfile.mkdtemp()
    for file in vals:
        async with file.open() as fh:
            contents = fh.read()
            print(f"File {file.path} contents: {contents}")
            new_file = Path(temp_dir) / file.name
            with open(new_file, "wb") as out:  # noqa: ASYNC230
                out.write(contents)
    print(f"Files written to {temp_dir}")

    # walk the directory and ls
    for path in Path(temp_dir).iterdir():
        print(f"File: {path.name}")

    my_dir = await Dir.from_local(temp_dir)
    return my_dir


@env.task
async def check_dir(my_dir: Dir):
    print(f"Dir {my_dir.path} contents:")
    async for file in my_dir.walk():
        print(f"File: {file.name}")
        async with file.open() as fh:
            contents = fh.read()
            print(f"Contents: {contents.decode('utf-8')}")


@env.task
async def create_and_check_dir():
    my_dir = await write_and_check_files()
    await check_dir(my_dir=my_dir)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    flyte.run(create_and_check_dir)
