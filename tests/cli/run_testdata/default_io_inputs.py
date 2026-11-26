import flyte
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(
    name="test_default_io",
)


@env.task
def task_with_defaults(
    f: File = File(path="s3://my-bucket/file.txt"),
    d: Dir = Dir(path="s3://my-bucket/dir/"),
    df: DataFrame = DataFrame.from_existing_remote(remote_path="s3://my-bucket/df.parquet"),
) -> str:
    return "success"
