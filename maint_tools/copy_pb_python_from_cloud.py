# /// script
# dependencies = ["click"]
# ///

"""Copy over generated proto stubs from union/cloud. To run:

python maint_tools/copy_pb_python_from_cloud.py PATH_TO_UNIONAI_CLOUD
"""

import re
import shutil
from itertools import product
from pathlib import Path

import click

# Add the modules and specific files to copy here.
PROTOS_TO_COPY_CLOUD = (
    ("gen", "pb_python"),
    {
        "app": [
            "app_definition_pb2",
            "app_payload_pb2",
            "app_service_pb2",
            "app_logs_payload_pb2",
            "app_logs_service_pb2",
            "replica_definition_pb2",
        ],
    },
)

PATTERNS_TO_SWITCH = [
    (re.compile(rf"^from {mod}", flags=re.MULTILINE), f"from union._protos.{mod}")
    for mod in [path.replace("/", ".") for path in PROTOS_TO_COPY_CLOUD[1]]
] + [(re.compile(r"^from validate", flags=re.MULTILINE), "from union._protos.validate.validate")]

PROTOS_TO_COPY = [PROTOS_TO_COPY_CLOUD]


@click.command
@click.argument("unionai-cloud-root", type=click.Path(exists=True))
def main(unionai_cloud_root):
    """Copy generated proto stubs from union/cloud to this package."""
    internal_path = Path("src") / "flyte" / "_protos"
    assert internal_path.exists()

    for path_tuple, protos_dict in PROTOS_TO_COPY:
        pb_python_path = Path(unionai_cloud_root).joinpath(*path_tuple)
        assert pb_python_path.exists()

        extensions_to_copy = [".py", ".pyi", "_grpc.py"]

        for module, files in protos_dict.items():
            module_path = pb_python_path / module
            assert module_path.exists()

            dest_dir_path = internal_path / module

            if dest_dir_path.exists():
                shutil.rmtree(dest_dir_path)

            dest_dir_path.mkdir(exist_ok=True, parents=True)

            for file, ext in product(files, extensions_to_copy):
                src_path = module_path / f"{file}{ext}"
                dest_path = dest_dir_path / f"{file}{ext}"

                contents = src_path.read_text()

                # Replace relative imports with absolute imports
                for pattern, replacement in PATTERNS_TO_SWITCH:
                    contents = pattern.sub(replacement, contents)

                print(f"Writing to {dest_path} from {module_path}")
                dest_path.parent.mkdir(exist_ok=True, parents=True)
                dest_path.write_text(contents)


if __name__ == "__main__":
    main()
