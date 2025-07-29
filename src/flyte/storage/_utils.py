import os

# This is the default chunk size flyte will use for writing to S3 and GCS. This is set to 25MB by default and is
# configurable by the user if needed. This is used when put() is called on filesystems.
_WRITE_SIZE_CHUNK_BYTES = int(os.environ.get("_F_P_WRITE_CHUNK_SIZE", "26214400"))  # 25 * 2**20
