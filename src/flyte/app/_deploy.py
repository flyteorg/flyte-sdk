from __future__ import annotations

import typing
from dataclasses import dataclass

import flyte._deployer as deployer
from flyte import Image
from flyte._initialize import ensure_client, get_client
from flyte._logging import logger
from flyte.models import SerializationContext

from ._app_environment import AppEnvironment

if typing.TYPE_CHECKING:
    from flyteidl2.app import app_definition_pb2

FILES_TAR_FILE_NAME = "code_bundle.tgz"


async def upload_include_files(app: AppEnvironment) -> str | None:
    import os
    import tarfile
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import flyte.remote

    with TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, FILES_TAR_FILE_NAME)
        with tarfile.open(tar_path, "w:gz") as tar:
            for resolve_include in app.include_resolved:
                tar.add(resolve_include.src, arcname=resolve_include.dest)

        _, upload_native_url = await flyte.remote.upload_file.aio(Path(tar_path))
        return upload_native_url


@dataclass
class DeployedAppEnvironment(deployer.DeployedEnvironment):
    env: AppEnvironment
    deployed_app: app_definition_pb2.App

    def get_name(self) -> str:
        """
        Returns the name of the deployed environment.
        Returns:
        """
        return self.env.name

    def env_repr(self) -> typing.List[typing.Tuple[str, ...]]:
        return [
            ("environment", self.env.name),
            ("image", self.env.image.uri if isinstance(self.env.image, Image) else self.env.image or ""),
        ]

    def table_repr(self) -> typing.List[typing.List[typing.Tuple[str, ...]]]:
        return [
            [
                ("type", "App"),
                ("name", self.deployed_app.metadata.id.name),
                ("version", self.deployed_app.spec.runtime_metadata.version),
                (
                    "state",
                    self.deployed_app.spec.desired_state.DESIRED_STATE.Name(self.deployed_app.spec.desired_state),
                ),
            ],
        ]

    def summary_repr(self) -> str:
        return f"Deployed App[{self.deployed_app.metadata.id.name}] in environment {self.env.name}"


async def _deploy_app(
    app: AppEnvironment, serialization_context: SerializationContext, dryrun: bool = False
) -> app_definition_pb2.App:
    """
    Deploy the given app.
    """
    import grpc.aio
    from flyteidl2.app import app_payload_pb2

    import flyte.errors
    import flyte.remote
    from flyte.app._runtime import translate_app_env_to_idl

    # TODO We need to handle uploading include files, ideally this is part of code bundle
    # The reason is at this point, we already have a code bundle created.
    # additional_distribution = await upload_include_files(app)
    # materialized_inputs = {}

    image_uri = app.image.uri if isinstance(app.image, Image) else app.image
    try:
        app_idl = translate_app_env_to_idl(app, serialization_context)
        if dryrun:
            return app_idl
        ensure_client()
        msg = f"Deploying app {app.name}, with image {image_uri} version {serialization_context.version}"
        if app_idl.spec.HasField("container") and app_idl.spec.container.args:
            msg += f" from {app_idl.spec.container.args[-3]}.{app_idl.spec.container.args[-1]}"
        logger.info(msg)

        try:
            await get_client().app_service.Create(app_payload_pb2.CreateRequest(app=app_idl))
            logger.info(f"Deployed app {app.name} with version {app_idl.spec.runtime_metadata.version}")
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                logger.info(f"App {app.name} with image {image_uri} already exists, skipping deployment.")
                return app_idl
            raise

        return app_idl
    except Exception as exc:
        logger.error(f"Failed to deploy app {app.name} with image {image_uri}: {exc}")
        raise flyte.errors.DeploymentError(
            f"Failed to deploy app {app.name} with image {image_uri}, Error: {exc!s}"
        ) from exc


async def _deploy_app_env(context: deployer.DeploymentContext) -> deployer.DeployedEnvironment:
    if not isinstance(context.environment, AppEnvironment):
        raise TypeError(f"Expected AppEnvironment, got {type(context.environment)}")

    app_env = context.environment
    deployed_app = await _deploy_app(app_env, context.serialization_context, dryrun=context.dryrun)

    return DeployedAppEnvironment(env=app_env, deployed_app=deployed_app)
