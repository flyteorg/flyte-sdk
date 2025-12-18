from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import flyte._deployer as deployer
from flyte import Image
from flyte._code_bundle.bundle import build_code_bundle_from_relative_paths
from flyte._initialize import ensure_client
from flyte._logging import logger
from flyte.models import SerializationContext

from ._app_environment import AppEnvironment
from ._input import Input

if typing.TYPE_CHECKING:
    from flyte._deployer import DeployedEnvironment
    from flyte.remote import App

FILES_TAR_FILE_NAME = "code_bundle.tgz"


@dataclass
class DeployedAppEnvironment:
    env: AppEnvironment
    deployed_app: "App"

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
                ("name", f"[link={self.deployed_app.url}]{self.deployed_app.name}[/link]"),
                ("revision", str(self.deployed_app.revision)),
                (
                    "desired state",
                    app_definition_pb2.Spec.DesiredState.Name(self.deployed_app.desired_state),
                ),
                (
                    "current state",
                    app_definition_pb2.Status.DeploymentStatus.Name(self.deployed_app.deployment_status),
                ),
                (
                    "public_url",
                    self.deployed_app.endpoint,
                ),
            ],
        ]

    def summary_repr(self) -> str:
        return f"Deployed App[{self.deployed_app.name}] in environment {self.env.name}"


async def _deploy_app(
    app: AppEnvironment,
    serialization_context: SerializationContext,
    input_overrides: list[Input] | None = None,
    dryrun: bool = False,
) -> "App":
    """
    Deploy the given app.
    """
    import flyte.errors
    from flyte.app._runtime import translate_app_env_to_idl
    from flyte.remote import App

    if app.include:
        app_file = Path(app._app_filename)
        app_root_dir = app_file.parent
        assert serialization_context.code_bundle is not None
        _files = serialization_context.code_bundle.files or []
        files = (*_files, *[f for f in app.include if f not in _files])
        code_bundle = await build_code_bundle_from_relative_paths(files, from_dir=app_root_dir)
        serialization_context.code_bundle = code_bundle

    image_uri = app.image.uri if isinstance(app.image, Image) else app.image
    try:
        app_idl = await translate_app_env_to_idl.aio(app, serialization_context, input_overrides=input_overrides)

        if dryrun:
            return app_idl
        ensure_client()
        msg = f"Deploying app {app.name}, with image {image_uri} version {serialization_context.version}"
        if app_idl.spec.HasField("container") and app_idl.spec.container.args:
            msg += f" with args {app_idl.spec.container.args}"
        logger.info(msg)

        return await App.create.aio(app_idl)
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
