"""
Run this example with the flyte CLI:

```
flyte run --local examples/basics/file_local.py process_file --file ./README.md
```

Or with python:

```
python examples/basics/file_local.py
```
"""

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(
    name="file-local",
)


@env.task
async def process_file(file: File) -> str:
    async with file.open("rb") as f:
        content = bytes(await f.read())
        text_content = content.decode("utf-8")
        print("text_content", text_content)
    return text_content


if __name__ == "__main__":
    import logging
    import tempfile

    flyte.init_from_config(log_level=logging.DEBUG)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp:
        temp.write("Hello, Flyte!")
        temp.seek(0)
        temp_path = temp.name
        file = File.from_local_sync(temp_path)
        run = flyte.with_runcontext(mode="local").run(process_file, file=file)
        print(run.url)
        run.wait()
        print("outputs")
        print(run.outputs()[0])
