"""
Flyte runtime serve module. This is used to serve Apps/serving.
"""

import logging
from typing import Tuple

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--inputs", "-i")
@click.option("--version", required=True)
@click.option("--interactive-mode", type=click.BOOL, required=False)
@click.option("--image-cache", required=False)
@click.option("--tgz", required=False)
@click.option("--pkl", required=False)
@click.option("--dest", required=False)
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
):
    import json
    import os
    import signal
    from subprocess import Popen

    from flyte._app_runtime.constants import _RUNTIME_CONFIG_FILE
    from flyte.models import CodeBundle

    serve_config = {}
    env_vars = {}

    logger.info("Starting flyte-serve")

    inputs_json = json.loads(inputs) if inputs else None
    bundle = None
    if tgz or pkl:
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)

    # download code bundle and inputs
    if inputs_json:
        download_inputs(inputs_json)

    serve_file = os.path.join(os.getcwd(), _RUNTIME_CONFIG_FILE)
    with open(serve_file, "w") as f:
        json.dump(serve_config, f)

    os.environ[_RUNTIME_CONFIG_FILE] = serve_file

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
