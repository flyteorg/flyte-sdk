"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

import asyncio
import logging
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


async def sync_inputs(serialized_inputs: str, dest: str) -> Tuple[dict, dict]:
    """
    Converts inputs into simple dict of name to value, downloading any files/directories as needed.

    Args:
        serialized_inputs (str): The serialized inputs string.
        dest: Destination to download inputs to

    Returns:
        Tuple[dict, dict]: A tuple containing the output dictionary and the environment variables dictionary.
        The output dictionary maps input names to their values.
        The environment variables dictionary maps environment variable names to their values.
    """
    import flyte.storage as storage
    from flyte.app._input import SerializableInputCollection

    user_inputs = SerializableInputCollection.from_transport(serialized_inputs)

    output = {}
    env_vars = {}

    for input in user_inputs.inputs:
        if input.download:
            user_dest = input.dest or dest
            if input.type == "file":
                value = await storage.get(input.value, user_dest)
            elif input.type == "directory":
                value = await storage.get(input.value, user_dest, recursive=True)
            else:
                raise ValueError("Can only download files or directories")
        else:
            value = input.value

        output[input.name] = value

        if input.env_var:
            env_vars[input.env_var] = value

    return output, env_vars


async def download_code_inputs(
    serialized_inputs: str, tgz: str, pkl: str, dest: str, version: str
) -> Tuple[dict, dict, CodeBundle | None]:
    from flyte._internal.runtime.entrypoints import download_code_bundle

    user_inputs: dict[str, str] = {}
    env_vars: dict[str, str] = {}
    if serialized_inputs and len(serialized_inputs) > 0:
        user_inputs, env_vars = await sync_inputs(serialized_inputs, dest)
    code_bundle: CodeBundle | None = None
    if tgz or pkl:
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
        code_bundle = await download_code_bundle(bundle)

    return user_inputs, env_vars, code_bundle


@click.command()
@click.option("--inputs", "-i", required=False)
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
    inputs: str | None,
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

    from flyte.app._input import RUNTIME_INPUTS_FILE

    logger.info("Starting flyte-serve")
    # TODO Do we need to init here?
    # from flyte._initialize import init
    # remote_kwargs: dict[str, Any] = {"insecure": False}
    # if api_key := os.getenv(_UNION_EAGER_API_KEY_ENV_VAR):
    #     logger.info("Using api key from environment")
    #     remote_kwargs["api_key"] = api_key
    # else:
    #     ep = os.environ.get(ENDPOINT_OVERRIDE, "host.docker.internal:8090")
    #     remote_kwargs["endpoint"] = ep
    #     if "localhost" in ep or "docker" in ep:
    #         remote_kwargs["insecure"] = True
    #     logger.debug(f"Using controller endpoint: {ep} with kwargs: {remote_kwargs}")
    # init(org=org, project=project, domain=domain, image_builder="remote")  # , **remote_kwargs)

    materialized_inputs, env_vars, _code_bundle = asyncio.run(
        download_code_inputs(
            serialized_inputs=inputs or "",
            tgz=tgz or "",
            pkl=pkl or "",
            dest=dest or os.getcwd(),
            version=version,
        )
    )

    for key, value in env_vars.items():
        # set environment variables defined in the AppEnvironment Inputs
        logger.info(f"Setting environment variable {key}='{value}'")
        os.environ[key] = value

    inputs_file = os.path.join(os.getcwd(), RUNTIME_INPUTS_FILE)
    with open(inputs_file, "w") as f:
        json.dump(materialized_inputs, f)

    os.environ[RUNTIME_INPUTS_FILE] = inputs_file

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
