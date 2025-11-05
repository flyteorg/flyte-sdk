import os
import tempfile

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="inputs_blob_types")


@env.task
async def process_blob(blob: File) -> str:
    """Process a blob input file and return its content summary"""
    async with blob.open("rb") as f:
        content = await f.read()
        text_content = content.decode("utf-8")
        lines = text_content.strip().split("\n")
        return f"Processed blob file: {blob.name}\nTotal lines: {len(lines)}\nFirst line: {lines[0] if lines else 'Empty file'}"


if __name__ == "__main__":
    flyte.init_from_config()

    # Create a local test file and upload it
    test_content = "Hello, Flyte!\nThis is a blob file.\nIt contains multiple lines of text."
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(test_content)
        tmp_path = tmp.name

    try:
        # Upload the local file to create a blob
        test_blob = File.from_local_sync(tmp_path)
        
        # Process the blob
        r = flyte.run(process_blob, blob=test_blob)
        print(r.name)
        print(r.url)
        r.wait()
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

