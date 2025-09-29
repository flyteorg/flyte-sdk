import flyte
import flyte.io
import flyte.storage as storage

env = flyte.TaskEnvironment(
    "rewrite",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)


@env.task(cache="auto")
async def create_large_file(dest_path: str, size_gigabytes: int = 5) -> flyte.io.File:
    f = flyte.io.File(path=dest_path)

    async with f.open("wb") as fp:
        chunk = b"\0" * (1024 * 1024)  # 1 MiB chunk
        for _ in range(size_gigabytes * 1024):
            fp.write(chunk)
    return f


@env.task
async def pathrewrite_read(f: flyte.io.File) -> int:
    remote_path = "s3://union-cloud-oc-canary-playground-persistent/"
    try:
        print(f"Check if remote path exists {remote_path} {await storage.exists(remote_path)}")
    except Exception as e:
        print(f"Failed to check if remote path exists {remote_path} {e}")
    try:
        print(f"Check if file exists: {f.path} {await storage.exists(f.path)}")
    except Exception as e:
        print(f"Failed to check if file exists: {f.path} {e}")

    try:
        print(f"Check if path exists: /mnt/mountpoint/data/ {await storage.exists('/mnt/mountpoint/data/')}")
    except Exception as e:
        print(f"Failed to check if path exists: [/mnt/mountpoint/data/ {e}")

    try:
        print(f"Reading from {f.path}", flush=True)
    except Exception as e:
        print(f"Failed to print reading from {f.path} {e}", flush=True)
    count = 0
    async with f.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            count += 1
    return count


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())

    # Create large data
    # r = flyte.run(create_large_file, "s3://union-cloud-oc-canary-playground-persistent/my_data.dat")
    # print(r.url)

    # Try read the data without acceleration and with acceleration
    r = flyte.with_runcontext(
        env_vars={"_F_PATH_REWRITE": "s3://union-cloud-oc-canary-playground-persistent/->/mnt/mountpoint/data/"},
    ).run(
        pathrewrite_read,
        flyte.io.File.from_existing_remote("s3://union-cloud-oc-canary-playground-persistent/my_data.dat"),
    )
    print(r.url)
