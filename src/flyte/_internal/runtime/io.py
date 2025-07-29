"""
This module contains the methods for uploading and downloading inputs and outputs.
It uses the storage module to handle the actual uploading and downloading of files.

TODO: Convert to use streaming apis
"""

import logging

from flyteidl.core import errors_pb2, execution_pb2

import flyte.storage as storage
from flyte._protos.workflow import run_definition_pb2

from ..._logging import log
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


async def upload_outputs(outputs: Outputs, output_path: str):
    """
    :param outputs: Outputs
    :param output_path: The path to upload the output file.
    """
    output_uri = outputs_path(output_path)
    await storage.put_stream(data_iterable=outputs.proto_outputs.SerializeToString(), to_path=output_uri)


async def upload_error(err: execution_pb2.ExecutionError, output_prefix: str):
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
    await storage.put_stream(data_iterable=error_document.SerializeToString(), to_path=error_uri)


# ------------------------------- DOWNLOAD Methods ------------------------------- #
@log(level=logging.INFO)
async def load_inputs(path: str) -> Inputs:
    """
    :param path: Input file to be downloaded
    :return: Inputs object
    """
    lm = run_definition_pb2.Inputs()
    proto_str = b"".join([c async for c in storage.get_stream(path=path)])
    lm.ParseFromString(proto_str)
    return Inputs(proto_inputs=lm)


async def load_outputs(path: str) -> Outputs:
    """
    :param path: output file to be loaded
    :return: Outputs object
    """
    lm = run_definition_pb2.Outputs()
    proto_str = b"".join([c async for c in storage.get_stream(path=path)])
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
        user_code, server_code = _clean_error_code(err.error.code)
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
