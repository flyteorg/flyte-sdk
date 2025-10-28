"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

import asyncio
import logging
from typing import List, Tuple

import click

logger = logging.getLogger(__name__)

PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"


async def download_inputs(user_inputs: List[dict], dest: str) -> Tuple[dict, dict]:
    """
    Loads
    Args:
        user_inputs:
        dest:

    Returns:

    """
    import flyte.storage as storage

    output = {}
    env_vars = {}
    for user_input in user_inputs:
        if user_input["auto_download"]:
            user_dest = user_input["dest"] or dest
            if user_input["type"] == "file":
                value = await storage.get(user_input["value"], user_dest)
            elif user_input["type"] == "directory":
                value = await storage.get(user_input["value"], user_dest, recursive=True)
            else:
                raise ValueError("Can only download files or directories")
        else:
            value = user_input["value"]

        output[user_input["name"]] = value

        if user_input["env_name"] is not None:
            env_vars[user_input["env_name"]] = value

    return output, env_vars


@click.command()
@click.option("--inputs", "-i")
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
    inputs: str,
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

    from flyte.app._runtime import RUNTIME_CONFIG_FILE
    from flyte.models import CodeBundle

    serve_config = {}
    env_vars = {}

    logger.info("Starting flyte-serve")

    inputs_json = json.loads(inputs) if inputs else None
    code_bundle = None
    if tgz or pkl:
        from flyte._internal.runtime.entrypoints import download_code_bundle

        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
        code_bundle = download_code_bundle(bundle)

    # download code bundle and inputs
    if inputs_json:
        asyncio.run(download_inputs(inputs_json, os.getcwd()))

    serve_file = os.path.join(os.getcwd(), RUNTIME_CONFIG_FILE)
    with open(serve_file, "w") as f:
        json.dump(serve_config, f)

    os.environ[RUNTIME_CONFIG_FILE] = serve_file

    command_joined = " ".join(command)
    logger.info(f"Serving command: {command_joined}")
    p = Popen(command_joined, env=os.environ, shell=True)

    def handle_sigterm(signum, frame):
        p.send_signal(signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    returncode = p.wait()
    exit(returncode)


if __name__ == "__main__":
    main()
