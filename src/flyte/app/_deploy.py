from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import flyte._deployer as deployer
from flyte import Image
from flyte._code_bundle.bundle import build_code_bundle_from_relative_paths
from flyte._initialize import ensure_client, get_client
from flyte._logging import logger
from flyte.models import SerializationContext

from ._app_environment import AppEnvironment

if typing.TYPE_CHECKING:
    from flyteidl2.app import app_definition_pb2

    from flyte._deployer import DeployedEnvironment

FILES_TAR_FILE_NAME = "code_bundle.tgz"


@dataclass
class DeployedAppEnvironment:
    env: AppEnvironment
    deployed_app: app_definition_pb2.App

    def get_name(self) -> str:
        """
        Returns the name of the deployed environment.
        """
        return self.env.name

    def env_repr(self) -> typing.List[typing.Tuple[str, ...]]:
        return [
            ("environment", self.env.name),
            ("image", self.env.image.uri if isinstance(self.env.image, Image) else self.env.image or ""),
        ]

    def table_repr(self) -> typing.List[typing.List[typing.Tuple[str, ...]]]:
        from flyteidl2.app import app_definition_pb2

        return [
            [
                ("type", "App"),
                ("name", self.deployed_app.metadata.id.name),
                ("version", self.deployed_app.spec.runtime_metadata.version),
                (
                    "state",
                    app_definition_pb2.Spec.DesiredState.Name(self.deployed_app.spec.desired_state),
                ),
                (
                    "public_url",
                    self.deployed_app.status.ingress.public_url,
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
    from flyteidl2.app import app_definition_pb2, app_payload_pb2

    import flyte.errors
    from flyte.app._runtime import translate_app_env_to_idl

    if app.include:
        app_file = Path(app._app_filename)
        app_root_dir = app_file.parent
        files = (app_file.name, *app.include)
        code_bundle = await build_code_bundle_from_relative_paths(files, from_dir=app_root_dir)
        serialization_context.code_bundle = code_bundle

    image_uri = app.image.uri if isinstance(app.image, Image) else app.image
    try:
        app_idl = translate_app_env_to_idl(app, serialization_context)
        if dryrun:
            return app_idl
        ensure_client()
        msg = f"Deploying app {app.name}, with image {image_uri} version {serialization_context.version}"
        if app_idl.spec.HasField("container") and app_idl.spec.container.args:
            msg += f" with args {app_idl.spec.container.args}"
        logger.info(msg)

        try:
            await get_client().app_service.Create(app_payload_pb2.CreateRequest(app=app_idl))
            logger.info(f"Deployed app {app.name} with version {app_idl.spec.runtime_metadata.version}")
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                logger.warning(f"App {app.name} with image {image_uri} already exists, updating...")
                resp = await get_client().app_service.Get(app_payload_pb2.GetRequest(app_id=app_idl.metadata.id))
                # Update the revision to match the existing app
                updated_app = app_definition_pb2.App(
                    metadata=resp.app.metadata, spec=app_idl.spec, status=resp.app.status
                )
                logger.info(f"Updating app {app.name} to revision {updated_app.metadata.revision}")
                update_resp = await get_client().app_service.Update(app_payload_pb2.UpdateRequest(app=updated_app))
                return update_resp.app
            raise

        return app_idl
    except Exception as exc:
        logger.error(f"Failed to deploy app {app.name} with image {image_uri}: {exc}")
        raise flyte.errors.DeploymentError(
            f"Failed to deploy app {app.name} with image {image_uri}, Error: {exc!s}"
        ) from exc


async def _deploy_app_env(context: deployer.DeploymentContext) -> DeployedEnvironment:
    if not isinstance(context.environment, AppEnvironment):
        raise TypeError(f"Expected AppEnvironment, got {type(context.environment)}")

    app_env = context.environment
    deployed_app = await _deploy_app(app_env, context.serialization_context, dryrun=context.dryrun)

    return DeployedAppEnvironment(env=app_env, deployed_app=deployed_app)
