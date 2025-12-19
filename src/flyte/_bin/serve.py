"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

from __future__ import annotations

import asyncio
import os
import traceback
import typing

import click

from flyte._logging import logger
from flyte.models import CodeBundle

if typing.TYPE_CHECKING:
    from flyte.app import AppEnvironment


PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
_UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"
_F_PATH_REWRITE = "_F_PATH_REWRITE"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"


async def sync_parameters(serialized_parameters: str, dest: str) -> tuple[dict, dict]:
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
) -> tuple[dict, dict, CodeBundle | None]:
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


def load_app_env(
    code_bundle: CodeBundle,
    resolver: str,
    resolver_args: str,
) -> AppEnvironment:
    """
    Load a app environment from a resolver.

    :param resolver: The resolver to use to load the task.
    :param resolver_args: Arguments to pass to the resolver.
    :return: The loaded task.
    """
    from flyte._internal.resolvers.app_env import AppEnvResolver
    from flyte._internal.runtime.entrypoints import load_class

    resolver_class = load_class(resolver)
    resolver_instance: AppEnvResolver = resolver_class()
    try:
        return resolver_instance.load_app_env(resolver_args)
    except ModuleNotFoundError as e:
        cwd = os.getcwd()
        files = []
        try:
            for root, dirs, filenames in os.walk(cwd):
                for name in dirs + filenames:
                    rel_path = os.path.relpath(os.path.join(root, name), cwd)
                    files.append(rel_path)
        except Exception as list_err:
            files = [f"(Failed to list directory: {list_err})"]

        msg = (
            "\n\nFull traceback:\n" + "".join(traceback.format_exc()) + f"\n[ImportError Diagnostics]\n"
            f"Module '{e.name}' not found in either the Python virtual environment or the current working directory.\n"
            f"Current working directory: {cwd}\n"
            f"Files found under current directory:\n" + "\n".join(f"  - {f}" for f in files)
        )
        raise ModuleNotFoundError(msg) from e


def load_pkl_app_env(code_bundle: CodeBundle) -> AppEnvironment:
    import gzip

    import cloudpickle

    if code_bundle.downloaded_path is None:
        raise ValueError("Code bundle downloaded_path is None. Code bundle must be downloaded first.")
    logger.debug(f"Loading app env from pkl: {code_bundle.downloaded_path}")
    try:
        with gzip.open(str(code_bundle.downloaded_path), "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to load pickled app env from {code_bundle.downloaded_path}. Reason: {e!s}")
        raise


@click.command()
@click.option("--parameters", "-p", required=False)
@click.option("--version", required=True)
@click.option("--image-cache", required=False)
@click.option("--tgz", required=False)
@click.option("--pkl", required=False)
@click.option("--dest", required=False)
@click.option("--project", envvar=PROJECT_NAME, required=False)
@click.option("--domain", envvar=DOMAIN_NAME, required=False)
@click.option("--org", envvar=ORG_NAME, required=False)
@click.option("--resolver", required=False)
@click.option("--resolver-args", type=str, required=False)
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def main(
    parameters: str | None,
    version: str,
    resolver: str,
    resolver_args: str,
    image_cache: str,
    tgz: str,
    pkl: str,
    dest: str,
    command: tuple[str, ...] | None = None,
    project: str | None = None,
    domain: str | None = None,
    org: str | None = None,
):
    import json
    import os
    import signal
    from subprocess import Popen

    from flyte._initialize import init_in_cluster
    from flyte.app._parameter import RUNTIME_PARAMETERS_FILE

    init_in_cluster(org=org, project=project, domain=domain)

    logger.info(f"Starting flyte-serve, org: {org}, project: {project}, domain: {domain}")

    materialized_parameters, env_vars, code_bundle = asyncio.run(
        download_code_parameters(
            serialized_parameters=parameters or "",
            tgz=tgz or "",
            pkl=pkl or "",
            dest=dest or os.getcwd(),
            version=version,
        )
    )

    if code_bundle is not None:
        if code_bundle.pkl:
            app_env = load_pkl_app_env(code_bundle)
        elif code_bundle.tgz:
            # TODO: implement a way to extract the app environment object from
            # the tgz bundle, similar to task resolver
            app_env = load_app_env(code_bundle, resolver, resolver_args)
        else:
            raise ValueError("Code bundle did not contain a tgz or pkl file")

    for key, value in env_vars.items():
        # set environment variables defined in the AppEnvironment Parameters
        logger.info(f"Setting environment variable {key}='{value}'")
        os.environ[key] = value

    parameters_file = os.path.join(os.getcwd(), RUNTIME_PARAMETERS_FILE)
    with open(parameters_file, "w") as f:
        json.dump(materialized_parameters, f)

    os.environ[RUNTIME_PARAMETERS_FILE] = parameters_file

    if app_env._startup_fn is not None:
        logger.info("Running app via startup function")
        app_env._startup_fn()

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
