import time
import flyte
from pathlib import Path
from flyte.errors import InitializationError
import flyte.storage
import socket

ROOT_DIR = Path(__file__).parent

def safe_init_flyte(max_retries=3):
    for attempt in range(max_retries):
        try:
            flyte.init_from_config(
                Path("/Users/ytong/.flyte/config-k3d.yaml"),
                root_dir=ROOT_DIR,
                storage=flyte.storage.S3.for_sandbox()
            )
            fs = flyte.storage.get_underlying_filesystem("s3")
            print(fs.info("s3://bucket"))
            print(f"Flyte initialized", flush=True)
        except (socket.gaierror, InitializationError) as e:
            print(f"Init attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"  All retries exhausted")
                raise


if __name__ == "__main__":
    safe_init_flyte()
