"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

from __future__ import annotations

import asyncio
import os
import traceback
import typing

import click

import flyte.io
from flyte._logging import logger
from flyte.models import CodeBundle

if typing.TYPE_CHECKING:
    from flyte.app import AppEnvironment


PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
_F_PATH_REWRITE = "_F_PATH_REWRITE"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"


async def sync_parameters(serialized_parameters: str, dest: str) -> tuple[dict, dict, dict]:
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

    # these will be serialized to json, the app can fetch these values via
    # env var or with flyte.app.get_input()
    serializable_parameters = {}

    # these will be passed into the AppEnvironment._server function.
    materialized_parameters = {}

    env_vars = {}

    for parameter in user_parameters.parameters:
        parameter_type = parameter.type
        ser_value = parameter.value

        materialized_value: str | flyte.io.File | flyte.io.Dir = ser_value
        # for files and directories, default to remote paths for the materialized value
        if parameter_type == "file":
            materialized_value = flyte.io.File(path=ser_value)
        elif parameter_type == "directory":
            materialized_value = flyte.io.Dir(path=ser_value)

        # download files or directories
        if parameter.download:
            user_dest = parameter.dest or dest

            # replace file and directory inputs with the local paths if download is True
            if parameter_type == "file":
                logger.info(f"Downloading {parameter.name} of type File to {user_dest}...")
                ser_value = await storage.get(ser_value, user_dest)
                materialized_value = flyte.io.File(path=ser_value)

            elif parameter_type == "directory":
                logger.info(f"Downloading {parameter.name} of type Directory to {user_dest}...")
                ser_value = await storage.get(ser_value, user_dest, recursive=True)
                materialized_value = flyte.io.Dir(path=ser_value)
            else:
                raise ValueError("Can only download files or directories")

        serializable_parameters[parameter.name] = ser_value
        materialized_parameters[parameter.name] = materialized_value

        if parameter.env_var:
            env_vars[parameter.env_var] = ser_value

    return serializable_parameters, materialized_parameters, env_vars


async def download_code_parameters(
    serialized_parameters: str, tgz: str, pkl: str, dest: str, version: str
) -> tuple[dict, dict, dict, CodeBundle | None]:
    from flyte._internal.runtime.entrypoints import download_code_bundle

    serializable_parameters: dict[str, str] = {}
    materialized_parameters: dict[str, str | flyte.io.File | flyte.io.Dir] = {}
    env_vars: dict[str, str] = {}
    if serialized_parameters and len(serialized_parameters) > 0:
        serializable_parameters, materialized_parameters, env_vars = await sync_parameters(serialized_parameters, dest)
    code_bundle: CodeBundle | None = None
    if tgz or pkl:
        logger.debug(f"Downloading Code bundle: {tgz or pkl} ...")
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
        code_bundle = await download_code_bundle(bundle)

    return serializable_parameters, materialized_parameters, env_vars, code_bundle


def load_app_env(
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


def _bind_parameters(
    func: typing.Callable,
    materialized_parameters: dict[str, str | flyte.io.File | flyte.io.Dir],
) -> dict[str, str | flyte.io.File | flyte.io.Dir]:
    """Bind materialized_parameters to func based on the argument names of the function.

    If the function has **kwargs, all materialized parameters are passed through.
    Otherwise, only parameters matching the function signature are bound.
    """
    import inspect

    sig = inspect.signature(func)

    # Check if function accepts **kwargs
    has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

    if has_var_keyword:
        # If function has **kwargs, pass all parameters
        return materialized_parameters

    # Otherwise, only bind parameters that match the function signature
    bound_params = {}
    for param_name in sig.parameters:
        if param_name in materialized_parameters:
            bound_params[param_name] = materialized_parameters[param_name]
    return bound_params


async def _serve(
    app_env: AppEnvironment,
    materialized_parameters: dict[str, str | flyte.io.File | flyte.io.Dir],
):
    import signal

    logger.info("Running app via server function")
    assert app_env._server is not None

    # Use the asyncio event loop's add_signal_handler, and ensure all cleanup happens
    # within the running event loop, not from a synchronous signal handler.
    loop = asyncio.get_running_loop()

    async def shutdown():
        logger.info("Received SIGTERM, shutting down server...")
        if app_env._on_shutdown is not None:
            bound_params = _bind_parameters(app_env._on_shutdown, materialized_parameters)
            if asyncio.iscoroutinefunction(app_env._on_shutdown):
                await app_env._on_shutdown(**bound_params)
            else:
                app_env._on_shutdown(**bound_params)
        logger.info("Server shut down")
        # Use loop.stop() to gracefully stop the loop after shutdown
        loop.stop()

    logger.info("Adding signal handler for SIGTERM using signal.signal")
    signal.signal(signal.SIGTERM, lambda signum, frame: asyncio.create_task(shutdown()))

    if app_env._on_startup is not None:
        logger.info("Running on_startup function")
        bound_params = _bind_parameters(app_env._on_startup, materialized_parameters)
        if asyncio.iscoroutinefunction(app_env._on_startup):
            await app_env._on_startup(**bound_params)
        else:
            app_env._on_startup(**bound_params)

    try:
        logger.info("Running server function")
        bound_params = _bind_parameters(app_env._server, materialized_parameters)
        if asyncio.iscoroutinefunction(app_env._server):
            await app_env._server(**bound_params)
        else:
            app_env._server(**bound_params)
    finally:
        await shutdown()


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

    serializable_parameters, materialized_parameters, env_vars, code_bundle = asyncio.run(
        download_code_parameters(
            serialized_parameters=parameters or "",
            tgz=tgz or "",
            pkl=pkl or "",
            dest=dest or os.getcwd(),
            version=version,
        ),
    )

    app_env: AppEnvironment | None = None
    if code_bundle is not None:
        if code_bundle.pkl:
            app_env = load_pkl_app_env(code_bundle)
        elif code_bundle.tgz:
            if resolver is not None and resolver_args is not None:
                logger.info(f"Loading app env from resolver: {resolver}, args: {resolver_args}")
                app_env = load_app_env(resolver, resolver_args)
            else:
                logger.info("Resolver arguments not provided, started server from command.")
        else:
            raise ValueError("Code bundle did not contain a tgz or pkl file")

    for key, value in env_vars.items():
        # set environment variables defined in the AppEnvironment Parameters
        logger.info(f"Setting environment variable {key}='{value}'")
        os.environ[key] = value

    parameters_file = os.path.join(os.getcwd(), RUNTIME_PARAMETERS_FILE)
    with open(parameters_file, "w") as f:
        json.dump(serializable_parameters, f)

    os.environ[RUNTIME_PARAMETERS_FILE] = parameters_file

    if app_env and app_env._server is not None:
        asyncio.run(_serve(app_env, materialized_parameters))
        exit(0)

    if command is None or len(command) == 0:
        raise ValueError("No command provided to execute")

    logger.info(f"Serving app with command: {command}")
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
