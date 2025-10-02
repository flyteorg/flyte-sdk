import os
import tempfile

import flyte
from flyte.io import File

env = flyte.TaskEnvironment("sync_file")


@env.task
def create_new_remote_file(content: str) -> File:
    """
    Demonstrates File.new_remote() - creating a new remote file.
    """
    f = File.new_remote()
    with f.open_sync("wb") as fh:
        fh.write(content.encode("utf-8"))
    print(f"Created new remote file at: {f.path}")
    return f


@env.task
def read_file_sync(f: File) -> str:
    """
    Demonstrates reading a file synchronously using open_sync().
    """
    with f.open_sync("rb") as fh:
        contents = fh.read().decode("utf-8")
        print(f"File {f.path} contents: {contents}")
        return contents


@env.task
def download_file_sync(f: File) -> str:
    """
    Demonstrates File.download_sync() - downloading a file to local storage.
    """
    local_path = f.download_sync()
    print(f"Downloaded file to: {local_path}")

    # Verify the download worked
    with open(local_path, "rb") as fh:
        contents = fh.read().decode("utf-8")
        print(f"Downloaded file contents: {contents}")

    return local_path


@env.task
def upload_local_file_sync(content: str) -> File:
    """
    Demonstrates File.from_local_sync() - uploading a local file.
    """
    # Create a temporary local file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Upload the local file
        uploaded_file = File.from_local_sync(tmp_path)
        print(f"Uploaded local file {tmp_path} to remote: {uploaded_file.path}")
        return uploaded_file
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


@env.task
def create_from_existing_remote(remote_path: str) -> File:
    """
    Demonstrates File.from_existing_remote() - referencing an existing remote file.
    """
    f = File.from_existing_remote(remote_path)
    print(f"Created File reference to existing remote file: {f.path}")
    return f


@env.task
def check_file_exists(f: File) -> bool:
    """
    Demonstrates File.exists_sync() - checking if a file exists.
    """
    exists = f.exists_sync()
    print(f"File {f.path} exists: {exists}")
    return exists


@env.task
def copy_file_by_reference(f: File) -> File:
    """
    Demonstrates creating a File by direct reference (no upload/download).
    """
    # This creates a new File object pointing to the same remote location
    copied_file = File(path=f.path)
    print(f"Created file reference to: {copied_file.path}")
    return copied_file


@env.task
def demonstrate_file_properties(f: File) -> None:
    """
    Demonstrates accessing File properties.
    """
    print(f"File path: {f.path}")
    print(f"File name: {f.name}")
    print(f"File format: {f.format}")
    print(f"File hash: {f.hash}")


@env.task
def main():
    """
    Main function demonstrating all File sync APIs.
    """
    print("=== Flyte File Sync API Examples ===\n")

    # 1. Create a new remote file
    print("1. Creating new remote file...")
    file1 = create_new_remote_file("Hello, Flyte Sync API!")

    # 2. Read the file
    print("\n2. Reading file contents...")
    read_file_sync(file1)

    # 3. Check if file exists
    print("\n3. Checking if file exists...")
    check_file_exists(file1)

    # 4. Download the file
    print("\n4. Downloading file...")
    download_file_sync(file1)

    # 5. Upload a local file
    print("\n5. Uploading local file...")
    file2 = upload_local_file_sync("Content from local file!")

    # 6. Create reference to existing file
    print("\n6. Creating reference to existing file...")
    file3 = create_from_existing_remote(file1.path)
    print("\nReferenced file path:", file3.path)

    # 7. Copy file by reference
    print("\n7. Copying file by reference...")
    copy_file_by_reference(file2)

    # 8. Demonstrate file properties
    print("\n8. File properties...")
    demonstrate_file_properties(file1)

    print("\n=== All File sync API examples completed! ===")


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
