"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

import asyncio
import logging
import os
from typing import Tuple

import click

from flyte.models import CodeBundle

logger = logging.getLogger(__name__)

PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
_UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"
_F_PATH_REWRITE = "_F_PATH_REWRITE"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"


async def sync_parameters(serialized_parameters: str, dest: str) -> Tuple[dict, dict]:
    """
    Converts parameters into simple dict of name to value, downloading any files/directories as needed.

    Args:
        serialized_parameters (str): The serialized parameters string.
        dest: Destination to download parameters to

    Returns:
        Tuple[dict, dict]: A tuple containing the output dictionary and the environment variables dictionary.
        The output dictionary maps parameter names to their values.
        The environment variables dictionary maps environment variable names to their values.
    """
    import flyte.storage as storage
    from flyte.app._parameter import SerializableParameterCollection

    print(f"Log level: {logger.getEffectiveLevel()} is set from env {os.environ.get('LOG_LEVEL')}", flush=True)
    logger.info("Reading parameters...")

    user_parameters = SerializableParameterCollection.from_transport(serialized_parameters)

    output = {}
    env_vars = {}

    for parameter in user_parameters.parameters:
        parameter_type = parameter.type
        value = parameter.value

        # download files or directories
        if parameter.download:
            user_dest = parameter.dest or dest
            if parameter_type == "file":
                logger.info(f"Downloading {parameter.name} of type File to {user_dest}...")
                value = await storage.get(value, user_dest)
            elif parameter_type == "directory":
                logger.info(f"Downloading {parameter.name} of type Directory to {user_dest}...")
                value = await storage.get(value, user_dest, recursive=True)
            else:
                raise ValueError("Can only download files or directories")

        output[parameter.name] = value

        if parameter.env_var:
            env_vars[parameter.env_var] = value

    return output, env_vars


async def download_code_parameters(
    serialized_parameters: str, tgz: str, pkl: str, dest: str, version: str
) -> Tuple[dict, dict, CodeBundle | None]:
    from flyte._internal.runtime.entrypoints import download_code_bundle

    user_parameters: dict[str, str] = {}
    env_vars: dict[str, str] = {}
    if serialized_parameters and len(serialized_parameters) > 0:
        user_parameters, env_vars = await sync_parameters(serialized_parameters, dest)
    code_bundle: CodeBundle | None = None
    if tgz or pkl:
        logger.debug(f"Downloading Code bundle: {tgz or pkl} ...")
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
        code_bundle = await download_code_bundle(bundle)

    return user_parameters, env_vars, code_bundle


@click.command()
@click.option("--parameters", "-p", required=False)
@click.option("--version", required=True)
@click.option("--interactive-mode", type=click.BOOL, required=False)
@click.option("--image-cache", required=False)
@click.option("--tgz", required=False)
@click.option("--pkl", required=False)
@click.option("--dest", required=False)
@click.option("--project", envvar=PROJECT_NAME, required=False)
@click.option("--domain", envvar=DOMAIN_NAME, required=False)
@click.option("--org", envvar=ORG_NAME, required=False)
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def main(
    parameters: str | None,
    version: str,
    interactive_mode: bool,
    image_cache: str,
    tgz: str,
    pkl: str,
    dest: str,
    command: Tuple[str, ...] | None = None,
    project: str | None = None,
    domain: str | None = None,
    org: str | None = None,
):
    import json
    import os
    import signal
    from subprocess import Popen

    from flyte.app._parameter import RUNTIME_PARAMETERS_FILE

    logger.info(f"Starting flyte-serve, org: {org}, project: {project}, domain: {domain}")

    materialized_parameters, env_vars, _code_bundle = asyncio.run(
        download_code_parameters(
            serialized_parameters=parameters or "",
            tgz=tgz or "",
            pkl=pkl or "",
            dest=dest or os.getcwd(),
            version=version,
        )
    )

    for key, value in env_vars.items():
        # set environment variables defined in the AppEnvironment Parameters
        logger.info(f"Setting environment variable {key}='{value}'")
        os.environ[key] = value

    parameters_file = os.path.join(os.getcwd(), RUNTIME_PARAMETERS_FILE)
    with open(parameters_file, "w") as f:
        json.dump(materialized_parameters, f)

    os.environ[RUNTIME_PARAMETERS_FILE] = parameters_file

    if command is None or len(command) == 0:
        raise ValueError("No command provided to execute")

    command_list = []
    for arg in command:
        logger.info(f"Processing arg: {arg}")
        if arg.startswith("$"):
            # expand environment variables in the user-defined command
            val = os.getenv(arg[1:])
            if val is None:
                raise ValueError(f"Environment variable {arg[1:]} not found")
            logger.info(f"Found env var {arg}.")
            command_list.append(val)
        else:
            command_list.append(arg)

    command_joined = " ".join(command_list)
    logger.info(f"Serving command: {command_joined}")
    p = Popen(command_joined, env=os.environ, shell=True)

    def handle_sigterm(signum, frame):
        p.send_signal(signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    returncode = p.wait()
    exit(returncode)


if __name__ == "__main__":
    main()
