"""
This module contains the methods for uploading and downloading inputs and outputs.
It uses the storage module to handle the actual uploading and downloading of files.

TODO: Convert to use streaming apis
"""

from flyteidl.core import errors_pb2
from flyteidl2.core import execution_pb2
from flyteidl2.task import common_pb2

import flyte.storage as storage
from flyte.models import PathRewrite

from .convert import Inputs, Outputs, _clean_error_code

# ------------------------------- CONSTANTS ------------------------------- #
_INPUTS_FILE_NAME = "inputs.pb"
_OUTPUTS_FILE_NAME = "outputs.pb"
_CHECKPOINT_FILE_NAME = "_flytecheckpoints"
_ERROR_FILE_NAME = "error.pb"
_REPORT_FILE_NAME = "report.html"
_PKL_EXT = ".pkl.gz"


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
    if max_bytes != -1 and outputs.proto_outputs.ByteSize() > max_bytes:
        import flyte.errors

        raise flyte.errors.InlineIOMaxBytesBreached(
            f"Output file at {output_path} exceeds max_bytes limit of {max_bytes},"
            f" size: {outputs.proto_outputs.ByteSize()}"
        )
    output_uri = outputs_path(output_path)
    await storage.put_stream(data_iterable=outputs.proto_outputs.SerializeToString(), to_path=output_uri)


async def upload_error(err: execution_pb2.ExecutionError, output_prefix: str) -> str:
    """
    :param err: execution_pb2.ExecutionError
    :param output_prefix: The output prefix of the remote uri.
    """
    # TODO - clean this up + conditionally set kind
    error_document = errors_pb2.ErrorDocument(
        error=errors_pb2.ContainerError(
            code=err.code,
            message=err.message,
            kind=errors_pb2.ContainerError.RECOVERABLE,
            origin=err.kind,
            timestamp=err.timestamp,
            worker=err.worker,
        )
    )
    error_uri = error_path(output_prefix)
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
    err = errors_pb2.ErrorDocument()
    proto_str = b"".join([c async for c in storage.get_stream(path=path)])
    err.ParseFromString(proto_str)

    if err.error is not None:
        user_code, _server_code = _clean_error_code(err.error.code)
        return execution_pb2.ExecutionError(
            code=user_code,
            message=err.error.message,
            kind=err.error.origin,
            error_uri=path,
            timestamp=err.error.timestamp,
            worker=err.error.worker,
        )

    return execution_pb2.ExecutionError(
        code="Unknown",
        message=f"Received unloadable error from path {path}",
        kind=execution_pb2.ExecutionError.SYSTEM,
        error_uri=path,
    )
