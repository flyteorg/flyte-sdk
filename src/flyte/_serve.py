import typing

from flyte.models import SerializationContext
from flyte.syncify import syncify

if typing.TYPE_CHECKING:
    from flyte.app import AppEnvironment
    from flyte.remote import App


@syncify
async def serve(app_env: "AppEnvironment") -> "App":
    """
    This method can be used to serve a flyte app using a flyte app environment.

    TODO: add support for with_servecontext, dryrun, copy_style, pickled serve etc

    Args:
        app_env: The app environment to serve.

    Returns:

    """

    import hashlib

    import cloudpickle

    from flyte.app import _deploy

    from ._code_bundle import build_code_bundle
    from ._deploy import build_images, plan_deploy
    from ._initialize import get_init_config

    cfg = get_init_config()

    deployments = plan_deploy(app_env)
    assert deployments
    app_deployment = deployments[0]
    image_cache = await build_images.aio(app_env)
    assert image_cache

    code_bundle = await build_code_bundle(from_dir=cfg.root_dir, dryrun=False, copy_style="loaded_modules")
    if app_deployment.version:
        version = app_deployment.version
    else:
        h = hashlib.md5()
        h.update(cloudpickle.dumps(app_deployment.envs))
        h.update(code_bundle.computed_version.encode("utf-8"))
        h.update(cloudpickle.dumps(image_cache))
        version = h.hexdigest()

    sc = SerializationContext(
        project=cfg.project,
        domain=cfg.domain,
        org=cfg.org,
        code_bundle=code_bundle,
        version=version,
        image_cache=image_cache,
        root_dir=cfg.root_dir,
    )

    deployed_app = await _deploy._deploy_app(app_env, sc)
    assert deployed_app
    return await deployed_app.watch.aio(wait_for="activated")
