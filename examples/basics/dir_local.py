"""
Run this example with the flyte CLI:

```
flyte run --local examples/basics/dir_local.py process_dir --dir ./examples/basics
```

Or with python:

```
python examples/basics/dir_local.py
```
"""

import flyte
from flyte.io import Dir

env = flyte.TaskEnvironment(
    name="dir-local",
)


@env.task
async def process_dir(dir: Dir) -> dict[str, str]:
    """Process a directory and return file names with their content previews."""
    file_contents = {}
    async for file in dir.walk(recursive=False):
        if file.name.endswith(".py"):
            async with file.open("rb") as f:
                content = bytes(await f.read())
                text_content = content.decode("utf-8")
                # Store first 100 chars as preview
                file_contents[file.name] = text_content[:100]
                print(f"Read {file.name}: {text_content[:50]}...")
    return file_contents


if __name__ == "__main__":
    import logging
    import os
    import tempfile

    flyte.init_from_config(log_level=logging.DEBUG)

    # Create a temporary directory with some test files
    with tempfile.TemporaryDirectory(prefix="flyte_dir_local_example_") as temp_dir:
        with open(os.path.join(temp_dir, "file1.py"), "w") as f:
            f.write("print('Hello from file 1!')")

        with open(os.path.join(temp_dir, "file2.py"), "w") as f:
            f.write("print('Greetings from file 2!')")

        with open(os.path.join(temp_dir, "file3.py"), "w") as f:
            f.write("print('Welcome from file 3!')")

        dir = Dir.from_local_sync(temp_dir)
        run = flyte.with_runcontext(mode="local").run(process_dir, dir=dir)
        print(run.url)
        run.wait()
        print("outputs")
        print(run.outputs()[0])
