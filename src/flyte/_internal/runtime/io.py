"""
This module contains the methods for uploading and downloading inputs and outputs.
It uses the storage module to handle the actual uploading and downloading of files.

"""

import os

from flyteidl2.core import execution_pb2
from flyteidl2.task import common_pb2

import flyte.storage as storage
from flyte._logging import logger
from flyte.models import PathRewrite

from .convert import Inputs, Outputs, _clean_error_code

# ------------------------------- CONSTANTS ------------------------------- #
_INPUTS_FILE_NAME = "inputs.pb"
_OUTPUTS_FILE_NAME = "outputs.pb"
_CHECKPOINT_FILE_NAME = "_flytecheckpoints"
_ERROR_FILE_NAME = "error.pb"
_REPORT_FILE_NAME = "report.html"
_PKL_EXT = ".pkl.gz"


def _is_clustered_worker() -> bool:
    """True for any worker process of a clustered/jobset task (torchrun sets this on every rank)."""
    return bool(os.environ.get("TORCHELASTIC_RUN_ID"))


def _is_nonzero_rank_clustered_worker() -> bool:
    """True only for a non-rank-0 process of a clustered/jobset task.

    torchrun sets both ``TORCHELASTIC_RUN_ID`` and ``RANK`` on every worker, so we gate on the
    torchrun marker rather than ``RANK`` alone — otherwise a regular Python task that happens to
    have ``RANK`` set in its environment would silently skip uploading its outputs/errors.
    """
    return _is_clustered_worker() and os.environ.get("RANK", "0") != "0"


def _get_clustered_restart_attempt() -> int | None:
    raw_attempt = os.environ.get("JOBSET_RESTART_ATTEMPT")
    if raw_attempt is None:
        return None
    try:
        return int(raw_attempt)
    except ValueError:
        logger.warning(f"Ignoring invalid JOBSET_RESTART_ATTEMPT={raw_attempt!r}")
        return None


def _get_clustered_max_restarts() -> int | None:
    raw_max = os.environ.get("JOBSET_MAX_RESTARTS")
    if raw_max is None:
        return None
    try:
        return int(raw_max)
    except ValueError:
        logger.warning(f"Ignoring invalid JOBSET_MAX_RESTARTS={raw_max!r}")
        return None


def _is_terminal_clustered_attempt() -> bool:
    """Whether a failure in this attempt should write error.pb.

    For a clustered/jobset task the JobSet restarts the whole pod set up to ``max_restarts`` times
    within a single Flyte attempt. We only write error.pb on the terminal attempt (budget exhausted)
    so transient restarts don't leave a stale error that a later successful restart would have to
    delete. Returns True (write) for non-clustered tasks, and as a safe fallback whenever the budget
    is unknown — errors must never be silently hidden.
    """
    attempt = _get_clustered_restart_attempt()
    if attempt is None:
        return True  # not a clustered restart context → write normally
    max_restarts = _get_clustered_max_restarts()
    if max_restarts is None:
        return True  # budget unknown (env not injected yet) → write, never hide errors
    return attempt >= max_restarts


def pkl_path(base_path: str, pkl_name: str) -> str:
    return storage.join(base_path, f"{pkl_name}{_PKL_EXT}")


def inputs_path(base_path: str) -> str:
    return storage.join(base_path, _INPUTS_FILE_NAME)


def outputs_path(base_path: str) -> str:
    return storage.join(base_path, _OUTPUTS_FILE_NAME)


def error_path(base_path: str) -> str:
    return storage.join(base_path, _ERROR_FILE_NAME)


def report_path(base_path: str) -> str:
    return storage.join(base_path, _REPORT_FILE_NAME)


# ------------------------------- UPLOAD Methods ------------------------------- #


async def upload_inputs(inputs: Inputs, input_path: str):
    """
    :param Inputs inputs: Inputs
    :param str input_path: The path to upload the input file.
    """
    await storage.put_stream(data_iterable=inputs.proto_inputs.SerializeToString(), to_path=input_path)


async def upload_outputs(outputs: Outputs, output_path: str, max_bytes: int = -1):
    """
    :param outputs: Outputs
    :param output_path: The path to upload the output file.
    :param max_bytes: Maximum number of bytes to write to the output file. Default is -1, which means no limit.
    """
    # In clustered tasks, only rank-0 owns the output; all other ranks skip upload.
    if _is_nonzero_rank_clustered_worker():
        return
    if max_bytes != -1 and outputs.proto_outputs.ByteSize() > max_bytes:
        import flyte.errors

        raise flyte.errors.InlineIOMaxBytesBreached(
            f"Output file at {output_path} exceeds max_bytes limit of {max_bytes},"
            f" size: {outputs.proto_outputs.ByteSize()}"
        )
    output_uri = outputs_path(output_path)
    await storage.put_stream(data_iterable=outputs.proto_outputs.SerializeToString(), to_path=output_uri)
    logger.debug(f"Uploaded {output_uri} to {output_path}")


async def upload_error(err: execution_pb2.ExecutionError, output_prefix: str, recoverable: bool = True) -> str:
    """
    :param err: execution_pb2.ExecutionError
    :param output_prefix: The output prefix of the remote uri.
    :param recoverable: If False, sets ContainerError.kind to NON_RECOVERABLE so the engine skips retries.
    """
    error_uri = error_path(output_prefix)
    # In clustered tasks, only rank-0 owns the error file; other ranks skip the write
    # so they don't race to clobber error.pb.
    if _is_nonzero_rank_clustered_worker():
        return error_uri
    # For a clustered task, only write error.pb once the JobSet has exhausted its restart budget.
    # Transient restarts recover on their own, so writing on every attempt would leave a stale
    # error that a later successful restart would have to delete.
    if not _is_terminal_clustered_attempt():
        logger.info(f"Skipping error.pb on transient JobSet restart (budget remaining): {error_uri}")
        return error_uri
    error_document = execution_pb2.ErrorDocument(
        error=execution_pb2.ContainerError(
            code=err.code,
            message=err.message,
            kind=execution_pb2.ContainerError.RECOVERABLE
            if recoverable
            else execution_pb2.ContainerError.NON_RECOVERABLE,
            origin=err.kind,
        )
    )
    return await storage.put_stream(data_iterable=error_document.SerializeToString(), to_path=error_uri)


# ------------------------------- DOWNLOAD Methods ------------------------------- #
async def load_inputs(path: str, max_bytes: int = -1, path_rewrite_config: PathRewrite | None = None) -> Inputs:
    """
    :param path: Input file to be downloaded
    :param max_bytes: Maximum number of bytes to read from the input file. Default is -1, which means no limit.
    :param path_rewrite_config: If provided, rewrites paths in the input blobs according to the configuration.
    :return: Inputs object
    """
    lm = common_pb2.Inputs()
    if max_bytes == -1:
        proto_str = b"".join([c async for c in storage.get_stream(path=path)])
    else:
        proto_bytes = []
        total_bytes = 0
        async for chunk in storage.get_stream(path=path):
            if total_bytes + len(chunk) > max_bytes:
                import flyte.errors

                raise flyte.errors.InlineIOMaxBytesBreached(
                    f"Input file at {path} exceeds max_bytes limit of {max_bytes}"
                )
            proto_bytes.append(chunk)
            total_bytes += len(chunk)
        proto_str = b"".join(proto_bytes)

    lm.ParseFromString(proto_str)

    if path_rewrite_config is not None:
        for inp in lm.literals:
            if inp.value.HasField("scalar") and inp.value.scalar.HasField("blob"):
                scalar_blob = inp.value.scalar.blob
                if scalar_blob.uri.startswith(path_rewrite_config.old_prefix):
                    scalar_blob.uri = scalar_blob.uri.replace(
                        path_rewrite_config.old_prefix, path_rewrite_config.new_prefix, 1
                    )
            # TODO add check for trigger time speciality here

    return Inputs(proto_inputs=lm)


async def load_outputs(path: str, max_bytes: int = -1) -> Outputs:
    """
    :param path: output file to be loaded
    :param max_bytes: Maximum number of bytes to read from the output file.
                      If -1, reads the entire file.
    :return: Outputs object
    """
    lm = common_pb2.Outputs()

    if max_bytes == -1:
        proto_str = b"".join([c async for c in storage.get_stream(path=path)])
    else:
        proto_bytes = []
        total_bytes = 0
        async for chunk in storage.get_stream(path=path):
            if total_bytes + len(chunk) > max_bytes:
                import flyte.errors

                raise flyte.errors.InlineIOMaxBytesBreached(
                    f"Output file at {path} exceeds max_bytes limit of {max_bytes}"
                )
            proto_bytes.append(chunk)
            total_bytes += len(chunk)
        proto_str = b"".join(proto_bytes)

    lm.ParseFromString(proto_str)
    return Outputs(proto_outputs=lm)


async def load_error(path: str) -> execution_pb2.ExecutionError:
    """
    :param path: error file to be downloaded
    :return: execution_pb2.ExecutionError
    """
    err = execution_pb2.ErrorDocument()
    proto_str = b"".join([c async for c in storage.get_stream(path=path)])
    err.ParseFromString(proto_str)

    if err.error is not None:
        user_code, _server_code = _clean_error_code(err.error.code)
        return execution_pb2.ExecutionError(
            code=user_code,
            message=err.error.message,
            kind=err.error.origin,
            error_uri=path,
        )

    return execution_pb2.ExecutionError(
        code="Unknown",
        message=f"Received unloadable error from path {path}",
        kind=execution_pb2.ExecutionError.SYSTEM,
        error_uri=path,
    )
