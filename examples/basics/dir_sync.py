import os
import tempfile

import flyte
from flyte.io import Dir, File

env = flyte.TaskEnvironment("dir_sync")


def create_test_local_directory() -> str:
    """
    Create a local directory with some test files for demonstration.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="flyte_dir_sync_example_")

    # Create some test files
    with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
        f.write("Content of file 1")

    with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
        f.write("Content of file 2")

    # Create a subdirectory with a file
    sub_dir = os.path.join(temp_dir, "subdir")
    os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, "file3.txt"), "w") as f:
        f.write("Content of file 3 in subdirectory")

    print(f"Created test directory at: {temp_dir}")
    return temp_dir


@env.task
def create_reference_to_existing_remote(remote_path: str) -> Dir:
    """
    Demonstrates Dir.from_existing_remote() - referencing an existing remote directory.
    """
    dir_ref = Dir.from_existing_remote(remote_path)
    print(f"Created Dir reference to existing remote directory: {dir_ref.path}")
    return dir_ref


@env.task
def check_directory_exists(d: Dir) -> bool:
    """
    Demonstrates Dir.exists_sync() - checking if a directory exists synchronously.
    """
    exists = d.exists_sync()
    print(f"Directory {d.path} exists: {exists}")
    return exists


@env.task
def list_files_in_directory_sync(d: Dir) -> list[File]:
    """
    Demonstrates Dir.list_files_sync() - getting a list of files in the directory (non-recursive).
    """
    files = d.list_files_sync()
    print(f"Found {len(files)} files in directory {d.path}:")
    for file in files:
        print(f"  - {file.name}: {file.path}")
    return files


@env.task
def walk_directory_sync(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk_sync() - synchronously walking through the directory.
    """
    all_files = []
    print(f"Walking directory {d.path} (recursive):")
    for file in d.walk_sync(recursive=True):
        print(f"  Found file: {file.name} at {file.path}")
        all_files.append(file)
    return all_files


@env.task
def walk_directory_non_recursive_sync(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk_sync() - walking directory non-recursively.
    """
    files = []
    print(f"Walking directory {d.path} (non-recursive):")
    for file in d.walk_sync(recursive=False):
        print(f"  Found file: {file.name} at {file.path}")
        files.append(file)
    return files


@env.task
def walk_directory_with_max_depth_sync(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk_sync() - walking directory with max depth limit.
    """
    files = []
    print(f"Walking directory {d.path} (max depth 2):")
    for file in d.walk_sync(recursive=True, max_depth=2):
        print(f"  Found file: {file.name} at {file.path}")
        files.append(file)
    return files


@env.task
def walk_directory_with_pattern_sync(d: Dir) -> list[File]:
    """
    Demonstrates Dir.walk_sync() - walking directory with file pattern filtering.
    """
    files = []
    print(f"Walking directory {d.path} with pattern '*.txt':")
    for file in d.walk_sync(recursive=True, file_pattern="*.txt"):
        print(f"  Found file: {file.name} at {file.path}")
        files.append(file)
    return files


@env.task
def get_specific_file_sync(d: Dir, file_name: str) -> File | None:
    """
    Demonstrates Dir.get_file_sync() - getting a specific file from the directory.
    """
    file = d.get_file_sync(file_name)
    if file:
        print(f"Found file {file_name}: {file.path}")
        return file
    else:
        print(f"File {file_name} not found in directory {d.path}")
        return None


@env.task
def read_files_in_directory_sync(d: Dir) -> dict[str, str]:
    """
    Demonstrates reading the contents of files in a directory synchronously.
    """
    file_contents = {}
    for file in d.walk_sync(recursive=False):
        if file.name.endswith(".txt"):  # Only read text files
            try:
                with file.open_sync("rb") as f:
                    content = f.read().decode("utf-8")
                    file_contents[file.name] = content
                    print(f"Read {file.name}: {content}")
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                file_contents[file.name] = f"Error: {e}"

    return file_contents


@env.task
def demonstrate_directory_properties(d: Dir) -> None:
    """
    Demonstrates accessing Dir properties.
    """
    print(f"Directory path: {d.path}")
    print(f"Directory name: {d.name}")
    print(f"Directory format: {d.format}")
    print(f"Directory hash: {d.hash}")


@env.task
def count_files_by_extension(d: Dir) -> dict[str, int]:
    """
    Demonstrates practical use case: counting files by extension.
    """
    extension_counts = {}
    for file in d.walk_sync(recursive=True):
        _, ext = os.path.splitext(file.name)
        ext = ext.lower() if ext else "no_extension"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    print("File counts by extension:")
    for ext, count in extension_counts.items():
        print(f"  {ext}: {count} files")

    return extension_counts


@env.task
def find_largest_files(d: Dir, top_n: int = 3) -> list[tuple[str, int]]:
    """
    Demonstrates practical use case: finding the largest files in a directory.
    """
    file_sizes = []
    for file in d.walk_sync(recursive=True):
        try:
            with file.open_sync("rb") as f:
                content = f.read()
                size = len(content)
                file_sizes.append((file.name, size))
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            file_sizes.append((file.name, 0))

    # Sort by size (descending) and take top N
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_files = file_sizes[:top_n]

    print(f"Top {top_n} largest files:")
    for name, size in largest_files:
        print(f"  {name}: {size} bytes")

    return largest_files


@env.task
def search_files_by_content(d: Dir, search_term: str) -> list[str]:
    """
    Demonstrates practical use case: searching for files containing specific content.
    """
    matching_files = []
    for file in d.walk_sync(recursive=True):
        if file.name.endswith(".txt"):  # Only search text files
            try:
                with file.open_sync("rb") as f:
                    content = f.read().decode("utf-8")
                    if search_term.lower() in content.lower():
                        matching_files.append(file.name)
                        print(f"Found '{search_term}' in {file.name}")
            except Exception as e:
                print(f"Error searching {file.name}: {e}")

    print(f"Found {len(matching_files)} files containing '{search_term}'")
    return matching_files


@env.task
def create_directory_structure_report(d: Dir) -> dict[str, int]:
    """
    Demonstrates practical use case: creating a directory structure report.
    """
    report = {
        "total_files": 0,
        "total_directories": 0,
        "text_files": 0,
        "empty_files": 0,
        "total_size": 0,
    }

    # Count files and calculate sizes
    for file in d.walk_sync(recursive=True):
        report["total_files"] += 1
        if file.name.endswith(".txt"):
            report["text_files"] += 1

        try:
            with file.open_sync("rb") as f:
                content = f.read()
                size = len(content)
                report["total_size"] += size
                if size == 0:
                    report["empty_files"] += 1
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    # Note: We can't easily count directories with the current API
    # since we're iterating over files, not directories
    report["total_directories"] = -1  # Indicates not available

    print("Directory structure report:")
    for key, value in report.items():
        if value == -1:
            print(f"  {key}: Not available with current API")
        else:
            print(f"  {key}: {value}")

    return report


@env.task
def filter_files_by_size(d: Dir, min_size: int = 0, max_size: int = 1000) -> list[str]:
    """
    Demonstrates practical use case: filtering files by size range.
    """
    filtered_files = []
    for file in d.walk_sync(recursive=True):
        try:
            with file.open_sync("rb") as f:
                content = f.read()
                size = len(content)
                if min_size <= size <= max_size:
                    filtered_files.append(file.name)
                    print(f"File {file.name}: {size} bytes (within range)")
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    print(f"Found {len(filtered_files)} files between {min_size} and {max_size} bytes")
    return filtered_files


@env.task
def backup_text_files_content(d: Dir) -> dict[str, str]:
    """
    Demonstrates practical use case: backing up all text file contents.
    """
    backup_data = {}
    for file in d.walk_sync(recursive=True):
        if file.name.endswith(".txt"):
            try:
                with file.open_sync("rb") as f:
                    content = f.read().decode("utf-8")
                    # Use relative path as key to maintain directory structure
                    backup_data[file.path] = content
                    print(f"Backed up content of {file.name}")
            except Exception as e:
                print(f"Error backing up {file.name}: {e}")
                backup_data[file.path] = f"Error: {e}"

    print(f"Backed up {len(backup_data)} text files")
    return backup_data


@env.task
def main():
    """
    Main function demonstrating all Dir sync APIs.
    """
    print("=== Flyte Dir Sync API Examples ===\n")

    # 1. Create a test local directory and upload it
    print("1. Creating test local directory and uploading...")
    local_dir_path = create_test_local_directory()

    remote_dir = Dir.from_local_sync(local_dir_path)
    print(f"Created reference to remote directory: {remote_dir.path}")

    # 2. Create reference to existing remote
    print("\n2. Creating reference to existing remote...")
    dir_ref = create_reference_to_existing_remote(remote_dir.path)

    # 3. Check if directory exists
    print("\n3. Checking if directory exists...")
    exists = check_directory_exists(dir_ref)
    print(f"Directory exists: {exists}")

    # Note: The following operations would work with an actual remote directory
    # For demonstration purposes, we'll show the API usage patterns

    print("\n=== Directory Navigation APIs ===")

    # 4. List files in directory (non-recursive)
    print("\n4. Listing files in directory (non-recursive)...")
    try:
        files = list_files_in_directory_sync(remote_dir)
        print(f"Total files found (non-recursive): {len(files)}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 5. Walk directory recursively
    print("\n5. Walking directory recursively...")
    try:
        all_files = walk_directory_sync(remote_dir)
        print(f"Total files found recursively: {len(all_files)}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 6. Walk directory non-recursively
    print("\n6. Walking directory non-recursively...")
    try:
        walk_directory_non_recursive_sync(remote_dir)
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 7. Walk directory with max depth
    print("\n7. Walking directory with max depth...")
    try:
        walk_directory_with_max_depth_sync(remote_dir)
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 8. Walk directory with pattern
    print("\n8. Walking directory with file pattern...")
    try:
        walk_directory_with_pattern_sync(remote_dir)
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 9. Get specific file
    print("\n9. Getting specific file...")
    try:
        specific_file = get_specific_file_sync(remote_dir, "file1.txt")
        if specific_file:
            print(f"Specific file path: {specific_file.path}")
        else:
            print("Specific file not found.")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    print("\n=== File Content Operations ===")

    # 10. Read file contents
    print("\n10. Reading file contents...")
    try:
        file_contents = read_files_in_directory_sync(remote_dir)
        print(f"File contents: {file_contents}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 11. Demonstrate directory properties
    print("\n11. Directory properties...")
    demonstrate_directory_properties(remote_dir)

    print("\n=== Practical Use Cases ===")

    # 12. Count files by extension
    print("\n12. Counting files by extension...")
    try:
        extension_counts = count_files_by_extension(remote_dir)
        print(f"Extension counts: {extension_counts}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 13. Find largest files
    print("\n13. Finding largest files...")
    try:
        largest_files = find_largest_files(remote_dir, top_n=3)
        print(f"Largest files: {largest_files}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 14. Search files by content
    print("\n14. Searching files by content...")
    try:
        matching_files = search_files_by_content(remote_dir, "Content")
        print(f"Files containing 'Content': {matching_files}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 15. Create directory structure report
    print("\n15. Creating directory structure report...")
    try:
        report = create_directory_structure_report(remote_dir)
        print(f"Directory report: {report}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 16. Filter files by size
    print("\n16. Filtering files by size...")
    try:
        filtered_files = filter_files_by_size(remote_dir, min_size=10, max_size=100)
        print(f"Files in size range: {filtered_files}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    # 17. Backup text files content
    print("\n17. Backing up text files content...")
    try:
        backup_data = backup_text_files_content(remote_dir)
        print(f"Backup data keys: {list(backup_data.keys())}")
    except Exception as e:
        print(f"Note: {e} (expected for demo directory)")

    print("\n=== All Dir sync API examples completed! ===")
    print("\nNote: Some operations may show errors because we're using a demo")
    print("directory path. In practice, these would work with actual remote directories.")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
