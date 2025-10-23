import os
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from flyteidl2.app import app_definition_pb2
from flyteidl2.core import tasks_pb2

from flyte._image import Image
from flyte._pod import PodTemplate
from flyte.app import AppEnvironment
from flyte.app._types import Scaling
from flyte.models import SerializationContext

FILES_TAR_FILE_NAME = "code_bundle.tgz"


async def upload_include_files(app: AppEnvironment) -> str | None:
    import flyte.remote

    with TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, FILES_TAR_FILE_NAME)
        with tarfile.open(tar_path, "w:gz") as tar:
            for resolve_include in app.include_resolved:
                tar.add(resolve_include.src, arcname=resolve_include.dest)

        _, upload_native_url = await flyte.remote.upload_file.aio(Path(tar_path))
        return upload_native_url


def translate_app_to_wire(app: AppEnvironment, settings: SerializationContext) -> app_definition_pb2.App:
    if app.config is not None:
        app.config.before_to_union_idl(app, settings)

    security_context_kwargs = {}
    security_context = None
    if app.secrets:
        security_context_kwargs["secrets"] = [s.to_flyte_idl() for s in app.secrets]
    if not app.requires_auth:
        security_context_kwargs["allow_anonymous"] = True

    if security_context_kwargs:
        security_context = app_definition_pb2.SecurityContext(**security_context_kwargs)

    scaling_metric = Scaling._to_union_idl(app.scaling_metric)

    dur = None
    if app.scaledown_after:
        from google.protobuf.duration_pb2 import Duration

        dur = Duration()
        dur.FromTimedelta(app.scaledown_after)

    autoscaling = app_definition_pb2.AutoscalingConfig(
        replicas=app_definition_pb2.Replicas(min=app.min_replicas, max=app.max_replicas),
        scaledown_period=dur,
        scaling_metric=scaling_metric,
    )

    spec_kwargs = {}
    if isinstance(app.image, (str, Image)):
        container_ports = [tasks_pb2.ContainerPort(container_port=app._port.port, name=app._port.name)]
        spec_kwargs["container"] = tasks_pb2.Container(
            image=settings.image_uri,
            command=app._get_command(settings),
            args=app._get_args(),
            resources=app._get_resources(),
            ports=container_ports,
            env=[literals_pb2.KeyValuePair(key=k, value=v) for k, v in app._get_env(settings).items()],
        )
    elif isinstance(app.image, PodTemplate):
        spec_kwargs["pod"] = app._get_k8s_pod(app.image, settings)
    else:
        msg = "container_image must be a str, ImageSpec or PodTemplate"
        raise ValueError(msg)

    return app_definition_pb2.App(
        metadata=app_definition_pb2.Meta(
            id=app_definition_pb2.Identifier(
                org=settings.org,
                project=settings.project,
                domain=settings.domain,
                name=app.name,
            ),
        ),
        spec=app_definition_pb2.Spec(
            desired_state=settings.desired_state,
            ingress=app_definition_pb2.IngressConfig(
                private=False,
                subdomain=app.subdomain if app.subdomain else None,
                cname=app.custom_domain if app.custom_domain else None,
            ),
            autoscaling=autoscaling,
            security_context=security_context,
            cluster_pool=app.cluster_pool,
            extended_resources=app._get_extended_resources(),
            runtime_metadata=tasks_pb2.RuntimeMetadata(
                type=tasks_pb2.RuntimeMetadata.RuntimeType.FLYTE_SDK,
                version=settings.version,
                flavor="python",
            ),
            profile=app_definition_pb2.Profile(
                type=app.type,
                short_description=app.description,
            ),
            links=(
                [
                    app_definition_pb2.Link(path=link.path, title=link.title, is_relative=link.is_relative)
                    for link in app.links
                ]
                if app.links
                else None
            ),
            **spec_kwargs,
        ),
    )
