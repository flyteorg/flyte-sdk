from __future__ import annotations

import os
import re
import tempfile
import typing
from typing import TYPE_CHECKING

from flyte._logging import logger
from flyte._resources import Resources
from flyte._task_environment import TaskEnvironment
from flyte.io import Dir
from pydantic import BaseModel

if TYPE_CHECKING:
    from flyte.remote import Run


HF_IMAGE_PACKAGES = [
    "huggingface-hub>=0.27.0",
    "hf-transfer>=0.1.8",
]


class HuggingFaceDatasetInfo(BaseModel):
    repo: str
    name: str | None = None
    split: str | None = None
    revision: str | None = None


def _validate_input_name(value: str | None) -> None:
    if value is not None and not re.match(r"^[a-zA-Z0-9_.-]+$", value):
        raise ValueError(f"'{value}' must only contain alphanumeric characters, underscores, hyphens, and dots")


def _stream_dataset_to_remote(
    repo_id: str,
    config_name: str | None,
    split: str | None,
    revision: str | None,
    token: str | None,
    remote_dir_path: str,
) -> str:
    import flyte.storage as storage
    import huggingface_hub

    hfs = huggingface_hub.HfFileSystem(token=token)
    fs = storage.get_underlying_filesystem(path=remote_dir_path)

    # HF Hub auto-converts datasets to parquet under refs/convert/parquet
    # Structure: datasets/{repo}/{config}/{split}/0000.parquet
    config = config_name or "default"
    base_path = f"datasets/{repo_id}/{config}"

    if split:
        search_paths = [f"{base_path}/{split}"]
    else:
        try:
            entries = hfs.ls(base_path, revision="refs/convert/parquet", detail=True)
            search_paths = [e["name"] for e in entries if e["type"] == "directory"]
        except FileNotFoundError:
            search_paths = [base_path]

    files_streamed = 0
    chunk_size = 64 * 1024 * 1024

    for search_path in search_paths:
        try:
            entries = hfs.ls(search_path, revision="refs/convert/parquet", detail=True)
        except FileNotFoundError:
            logger.warning(f"Path not found: {search_path}")
            continue

        parquet_files = [e for e in entries if e["type"] == "file" and e["name"].endswith(".parquet")]

        for file_info in parquet_files:
            file_name = file_info["name"].split("/")[-1]
            remote_file_path = f"{remote_dir_path}/{file_name}"
            logger.info(f"  Streaming {file_name}...")

            with hfs.open(file_info["name"], "rb", revision="refs/convert/parquet") as src:
                with fs.open(remote_file_path, "wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)

            files_streamed += 1

    if files_streamed == 0:
        raise FileNotFoundError(
            f"No parquet files found for {repo_id} (config={config}, split={split}). "
            f"The dataset may not have been auto-converted to parquet yet."
        )

    logger.info(f"Streamed {files_streamed} parquet files to {remote_dir_path}")
    return remote_dir_path


def _download_dataset_to_local(
    repo_id: str,
    config_name: str | None,
    split: str | None,
    revision: str | None,
    token: str | None,
    local_dir: str,
    flat_dir: str,
) -> str:
    import huggingface_hub

    config = config_name or "default"
    base_pattern = f"{config}/"
    if split:
        base_pattern = f"{config}/{split}/"

    files = huggingface_hub.list_repo_files(repo_id, repo_type="dataset", revision="refs/convert/parquet")
    parquet_files = [f for f in files if f.startswith(base_pattern) and f.endswith(".parquet")]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {repo_id} (config={config}, split={split}).")

    for pf in parquet_files:
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            repo_type="dataset",
            revision="refs/convert/parquet",
            local_dir=local_dir,
            token=token,
        )

    # Flatten: move parquet files to flat_dir root
    for root, _dirs, filenames in os.walk(local_dir):
        for fname in filenames:
            if fname.endswith(".parquet"):
                src = os.path.join(root, fname)
                dst = os.path.join(flat_dir, fname)
                os.rename(src, dst)

    return flat_dir


def _store_hf_dataset_task(info: str, raw_data_path: str | None = None) -> Dir:
    import flyte

    _info = HuggingFaceDatasetInfo.model_validate_json(info)
    token = os.environ.get("HF_TOKEN")

    artifact_name = _info.repo.split("/")[-1].replace(".", "-")
    if _info.split:
        artifact_name = f"{artifact_name}-{_info.split}"

    try:
        logger.info("Attempting direct streaming to remote storage...")

        if raw_data_path is not None:
            remote_path = raw_data_path
        else:
            remote_path = flyte.ctx().raw_data_path.get_random_remote_path(artifact_name)

        _stream_dataset_to_remote(_info.repo, _info.name, _info.split, _info.revision, token, remote_path)
        result_dir = Dir.from_existing_remote(remote_path)
        logger.info(f"Streaming completed to {remote_path}")

    except (OSError, FileNotFoundError) as e:
        logger.error(f"Direct streaming failed: {e}")
        logger.info("Falling back to snapshot download...")

        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as flat_dir:
            _download_dataset_to_local(_info.repo, _info.name, _info.split, _info.revision, token, local_dir, flat_dir)
            result_dir = Dir.from_local_sync(flat_dir, remote_destination=raw_data_path)

    logger.info(f"Dataset stored at {result_dir.path}")
    return result_dir


def hf_dataset(
    repo: str,
    *,
    name: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    raw_data_path: str | None = None,
    hf_token_key: str = "HF_TOKEN",
    resources: Resources = Resources(cpu="2", memory="8Gi", disk="50Gi"),
    force: int = 0,
) -> Run:
    """Prefetch a HuggingFace dataset to remote storage.

    Streams parquet files from HuggingFace Hub directly to Flyte's remote storage,
    returning a Dir that downstream tasks can consume.

    :param repo: HuggingFace dataset repo ID (e.g., 'stanfordnlp/imdb').
    :param name: Dataset configuration name (default: 'default').
    :param split: Dataset split (e.g., 'train', 'test'). None fetches all splits.
    :param revision: Dataset revision/commit. None uses latest.
    :param raw_data_path: Override remote storage path.
    :param hf_token_key: Secret key for HF token. Default: 'HF_TOKEN'.
    :param resources: Resources for the prefetch task.
    :param force: Increment to force re-prefetch.
    :return: A Run object. Call .wait() then .outputs() to get the Dir.
    """
    import flyte
    from flyte import Secret
    from flyte.remote import Run

    _validate_input_name(name)
    _validate_input_name(split)

    info = HuggingFaceDatasetInfo(
        repo=repo,
        name=name,
        split=split,
        revision=revision,
    )

    image = flyte.Image.from_debian_base(name="prefetch-hf-dataset-image").with_pip_packages(*HF_IMAGE_PACKAGES)

    env = TaskEnvironment(
        name="prefetch-hf-dataset",
        image=image,
        resources=resources,
        secrets=[Secret(key=hf_token_key, as_env_var="HF_TOKEN")],
    )
    task = env.task()(_store_hf_dataset_task)
    run = flyte.with_runcontext(interactive_mode=True, disable_run_cache=force > 0).run(
        task, info.model_dump_json(), raw_data_path
    )
    return typing.cast(Run, run)
