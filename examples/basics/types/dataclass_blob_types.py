import os
import tempfile
from dataclasses import dataclass

import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="inputs_dataclass_blob_types")


@dataclass
class DataWithBlob:
    """A dataclass that contains a blob property along with other fields"""

    name: str
    description: str
    blob: File
    count: int


@env.task
async def process_data_with_blob(data: DataWithBlob) -> str:
    """Process a dataclass containing a blob and other fields"""
    # Read the blob content
    async with data.blob.open("rb") as f:
        content = await f.read()
        text_content = content.decode("utf-8")
        lines = text_content.strip().split("\n")
        blob_size = len(content)

    result = f"""Processed data: {data.name}
Description: {data.description}
Count: {data.count}
Blob file: {data.blob.name}
Blob size: {blob_size} bytes
Blob lines: {len(lines)}
First line: {lines[0] if lines else "Empty file"}"""
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Create a local test file and upload it
    test_content = "Hello, Flyte!\nThis is a blob file.\nIt contains multiple lines of text.\nUsed in a dataclass example."
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(test_content)
        tmp_path = tmp.name

    try:
        # Upload the local file to create a blob
        test_blob = File.from_local_sync(tmp_path)

        # Create dataclass instance with blob
        data = DataWithBlob(
            name="Example Data",
            description="This is an example dataclass with a blob property",
            blob=test_blob,
            count=42,
        )

        # Process the dataclass
        r = flyte.run(process_data_with_blob, data=data)
        print(r.name)
        print(r.url)
        r.wait()
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

