"""
Example: Using File.from_existing_remote to overwrite an existing file.

This example demonstrates how to:
1. Create a file at a known remote location
2. Reference that file using File.from_existing_remote
3. Overwrite the contents of that file
4. Verify the file was overwritten
"""

import flyte
import flyte.report
from flyte.io import File

env = flyte.TaskEnvironment(
    name="overwrite-file-example",
    image=flyte.Image.from_debian_base(),
)


@env.task
async def create_initial_file() -> File:
    """
    Create an initial file with some content.
    We use File.new_remote() with a specific file name so we know the path.
    """
    f = File.new_remote(file_name="my_data.txt")
    async with f.open("wb") as fh:
        await fh.write(b"Original content - version 1")
    print(f"Created initial file at: {f.path}")
    return f


@env.task
async def overwrite_existing_file(remote_path: str, new_content: str) -> File:
    """
    Demonstrates using File.from_existing_remote() to overwrite an existing file.

    This is useful when you need to update a file at a known location, such as:
    - Updating a configuration file
    - Replacing a model artifact
    - Overwriting a data file with new data

    Args:
        remote_path: The remote path to the existing file (e.g., s3://bucket/path/file.txt)
        new_content: The new content to write to the file
    """
    # Create a reference to the existing remote file
    f = File.from_existing_remote(remote_path)
    print(f"Referencing existing file at: {f.path}")

    # Open the file in write mode to overwrite its contents
    async with f.open("wb") as fh:
        await fh.write(new_content.encode("utf-8"))

    print(f"Successfully overwrote file at: {f.path}")
    return f


@env.task
async def read_file_content(f: File) -> str:
    """
    Read and return the content of a file.
    """
    async with f.open("rb") as fh:
        content = bytes(await fh.read())
        return content.decode("utf-8")


@env.task(report=True)
async def main() -> None:
    """
    Main workflow demonstrating overwriting an existing file.
    """
    print("=== Overwrite Existing File Example ===\n")

    # Step 1: Create an initial file
    print("Step 1: Creating initial file...")
    initial_file = await create_initial_file()

    # Step 2: Read and display original content
    print("\nStep 2: Reading original content...")
    original_content = await read_file_content(initial_file)
    tab1 = flyte.report.get_tab("Original content")
    tab1.log(f"Original content: '{original_content}'")
    await flyte.report.flush.aio()
    print(f"Original content: '{original_content}'")

    # Step 3: Overwrite the file with new content
    print("\nStep 3: Overwriting file with new content...")
    updated_file = await overwrite_existing_file(
        remote_path=initial_file.path, new_content="Updated content - version 2"
    )

    # Step 4: Read and verify the new content
    print("\nStep 4: Verifying new content...")
    new_content = await read_file_content(updated_file)
    tab2 = flyte.report.get_tab("New content")
    tab2.log(f"New content: '{new_content}'")
    await flyte.report.flush.aio()
    print(f"New content: '{new_content}'")

    # Step 5: Also read from the original file reference to show it was updated
    print("\nStep 5: Reading from original file reference...")
    content_from_original_ref = await read_file_content(initial_file)
    tab3 = flyte.report.get_tab("Content from original reference")
    tab3.log(f"Content from original reference: '{content_from_original_ref}'")
    await flyte.report.flush.aio()
    print(f"Content from original reference: '{content_from_original_ref}'")

    print("\n=== Example completed! ===")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
