from __future__ import annotations

import asyncio
import pathlib
import sys
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

import flyte.errors
from flyte._context import contextual_run, internal_ctx
from flyte._environment import Environment
from flyte._initialize import (
    _get_init_config,
    get_client,
    get_init_config,
    get_storage,
    requires_initialization,
    requires_storage,
)
from flyte._logging import LogFormat, logger
from flyte._task import F, P, R, TaskTemplate
from flyte.models import (
    ActionID,
    Checkpoints,
    CodeBundle,
    RawDataPath,
    SerializationContext,
    TaskContext,
)
from flyte.syncify import syncify

from ._constants import FLYTE_SYS_PATH

if TYPE_CHECKING:
    from flyte.remote import Run
    from flyte.remote._task import LazyEntity

    from ._code_bundle import CopyFiles
    from ._internal.imagebuild.image_builder import ImageCache

Mode = Literal["local", "remote", "hybrid"]
CacheLookupScope = Literal["global", "project-domain"]


@dataclass(frozen=True)
class _CacheKey:
    obj_id: int
    dry_run: bool


@dataclass(frozen=True)
class _CacheValue:
    code_bundle: CodeBundle | None
    image_cache: Optional[ImageCache]


_RUN_CACHE: Dict[_CacheKey, _CacheValue] = {}


async def _get_code_bundle_for_run(name: str) -> CodeBundle | None:
    """
    Get the code bundle for the run with the given name.
    This is used to get the code bundle for the run when running in hybrid mode.
    """
    from flyte._internal.runtime.task_serde import extract_code_bundle
    from flyte.remote import Run

    run = await Run.get.aio(name=name)
    if run:
        run_details = await run.details.aio()
        spec = run_details.action_details.pb2.resolved_task_spec
        return extract_code_bundle(spec)
    return None


class _Runner:
    def __init__(
        self,
        force_mode: Mode | None = None,
        name: Optional[str] = None,
        service_account: Optional[str] = None,
        version: Optional[str] = None,
        copy_style: CopyFiles = "loaded_modules",
        dry_run: bool = False,
        copy_bundle_to: pathlib.Path | None = None,
        interactive_mode: bool | None = None,
        raw_data_path: str | None = None,
        metadata_path: str | None = None,
        run_base_dir: str | None = None,
        overwrite_cache: bool = False,
        project: str | None = None,
        domain: str | None = None,
        env_vars: Dict[str, str] | None = None,
        labels: Dict[str, str] | None = None,
        annotations: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        log_level: int | None = None,
        log_format: LogFormat = "console",
        disable_run_cache: bool = False,
        queue: Optional[str] = None,
        custom_context: Dict[str, str] | None = None,
        cache_lookup_scope: CacheLookupScope = "global",
    ):
        from flyte._tools import ipython_check

        init_config = _get_init_config()
        client = init_config.client if init_config else None
        if not force_mode and client is not None:
            force_mode = "remote"
        force_mode = force_mode or "local"
        logger.debug(f"Effective run mode: `{force_mode}`, client configured: `{client is not None}`")
        self._mode = force_mode
        self._name = name
        self._service_account = service_account
        self._version = version
        self._copy_files = copy_style
        self._dry_run = dry_run
        self._copy_bundle_to = copy_bundle_to
        self._interactive_mode = interactive_mode if interactive_mode else ipython_check()
        self._raw_data_path = raw_data_path
        self._metadata_path = metadata_path
        self._run_base_dir = run_base_dir
        self._overwrite_cache = overwrite_cache
        self._project = project
        self._domain = domain
        self._env_vars = env_vars
        self._labels = labels
        self._annotations = annotations
        self._interruptible = interruptible
        self._log_level = log_level
        self._log_format = log_format
        self._disable_run_cache = disable_run_cache
        self._queue = queue
        self._custom_context = custom_context or {}
        self._cache_lookup_scope = cache_lookup_scope

    @requires_initialization
    async def _run_remote(self, obj: TaskTemplate[P, R, F] | LazyEntity, *args: P.args, **kwargs: P.kwargs) -> Run:
        import grpc
        from flyteidl2.common import identifier_pb2
        from flyteidl2.core import literals_pb2, security_pb2
        from flyteidl2.task import run_pb2
        from flyteidl2.workflow import run_definition_pb2, run_service_pb2
        from google.protobuf import wrappers_pb2

        from flyte.remote import Run
        from flyte.remote._task import LazyEntity

        from ._code_bundle import build_code_bundle, build_pkl_bundle
        from ._deploy import build_images
        from ._internal.runtime.convert import convert_from_native_to_inputs
        from ._internal.runtime.task_serde import translate_task_to_wire

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        if isinstance(obj, LazyEntity):
            task = await obj.fetch.aio()
            task_spec = task.pb2.spec
            inputs = await convert_from_native_to_inputs(
                task.interface, *args, custom_context=self._custom_context, **kwargs
            )
            version = task.pb2.task_id.version
            code_bundle = None
        else:
            task = cast(TaskTemplate[P, R, F], obj)
            if obj.parent_env is None:
                raise ValueError("Task is not attached to an environment. Please attach the task to an environment")

            if (
                not self._disable_run_cache
                and _RUN_CACHE.get(_CacheKey(obj_id=id(obj), dry_run=self._dry_run)) is not None
            ):
                cached_value = _RUN_CACHE[_CacheKey(obj_id=id(obj), dry_run=self._dry_run)]
                code_bundle = cached_value.code_bundle
                image_cache = cached_value.image_cache
            else:
                if not self._dry_run:
                    image_cache = await build_images.aio(cast(Environment, obj.parent_env()))
                else:
                    image_cache = None

                if self._interactive_mode:
                    code_bundle = await build_pkl_bundle(
                        obj,
                        upload_to_controlplane=not self._dry_run,
                        copy_bundle_to=self._copy_bundle_to,
                    )
                else:
                    if self._copy_files != "none":
                        code_bundle = await build_code_bundle(
                            from_dir=cfg.root_dir,
                            dryrun=self._dry_run,
                            copy_bundle_to=self._copy_bundle_to,
                            copy_style=self._copy_files,
                        )
                    else:
                        code_bundle = None
            if not self._disable_run_cache:
                _RUN_CACHE[_CacheKey(obj_id=id(obj), dry_run=self._dry_run)] = _CacheValue(
                    code_bundle=code_bundle, image_cache=image_cache
                )

            version = self._version or (
                code_bundle.computed_version if code_bundle and code_bundle.computed_version else None
            )
            if not version:
                raise ValueError("Version is required when running a task")
            s_ctx = SerializationContext(
                code_bundle=code_bundle,
                version=version,
                image_cache=image_cache,
                root_dir=cfg.root_dir,
            )
            task_spec = translate_task_to_wire(obj, s_ctx)
            inputs = await convert_from_native_to_inputs(
                obj.native_interface, *args, custom_context=self._custom_context, **kwargs
            )

        env = self._env_vars or {}
        if env.get("LOG_LEVEL") is None:
            if self._log_level:
                env["LOG_LEVEL"] = str(self._log_level)
            else:
                env["LOG_LEVEL"] = str(logger.getEffectiveLevel())
        env["LOG_FORMAT"] = self._log_format

        # These paths will be appended to sys.path at runtime.
        if cfg.sync_local_sys_paths:
            env[FLYTE_SYS_PATH] = ":".join(
                f"./{pathlib.Path(p).relative_to(cfg.root_dir)}" for p in sys.path if p.startswith(str(cfg.root_dir))
            )

        if not self._dry_run:
            if get_client() is None:
                # This can only happen, if the user forces flyte.run(mode="remote") without initializing the client
                raise flyte.errors.InitializationError(
                    "ClientNotInitializedError",
                    "user",
                    "flyte.run requires client to be initialized. "
                    "Call flyte.init() with a valid endpoint or api-key before using this function.",
                )
            run_id = None
            project_id = None
            if self._name:
                run_id = identifier_pb2.RunIdentifier(
                    project=project,
                    domain=domain,
                    org=cfg.org,
                    name=self._name if self._name else None,
                )
            else:
                project_id = identifier_pb2.ProjectIdentifier(
                    name=project,
                    domain=domain,
                    organization=cfg.org,
                )
            # Fill in task id inside the task template if it's not provided.
            # Maybe this should be done here, or the backend.
            if task_spec.task_template.id.project == "":
                task_spec.task_template.id.project = project if project else ""
            if task_spec.task_template.id.domain == "":
                task_spec.task_template.id.domain = domain if domain else ""
            if task_spec.task_template.id.org == "":
                task_spec.task_template.id.org = cfg.org if cfg.org else ""
            if task_spec.task_template.id.version == "":
                task_spec.task_template.id.version = version

            kv_pairs: List[literals_pb2.KeyValuePair] = []
            for k, v in env.items():
                if not isinstance(v, str):
                    raise ValueError(f"Environment variable {k} must be a string, got {type(v)}")
                kv_pairs.append(literals_pb2.KeyValuePair(key=k, value=v))

            env_kv = run_pb2.Envs(values=kv_pairs)
            annotations = run_pb2.Annotations(values=self._annotations)
            labels = run_pb2.Labels(values=self._labels)
            raw_data_storage = (
                run_pb2.RawDataStorage(raw_data_prefix=self._raw_data_path) if self._raw_data_path else None
            )
            security_context = (
                security_pb2.SecurityContext(run_as=security_pb2.Identity(k8s_service_account=self._service_account))
                if self._service_account
                else None
            )

            def _to_cache_lookup_scope(scope: CacheLookupScope | None = None) -> run_pb2.CacheLookupScope:
                if scope == "global":
                    return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_GLOBAL
                elif scope == "project-domain":
                    return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_PROJECT_DOMAIN
                elif scope is None:
                    return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_UNSPECIFIED
                else:
                    raise ValueError(f"Unknown cache lookup scope: {scope}")

            try:
                resp = await get_client().run_service.CreateRun(
                    run_service_pb2.CreateRunRequest(
                        run_id=run_id,
                        project_id=project_id,
                        task_spec=task_spec,
                        inputs=inputs.proto_inputs,
                        run_spec=run_pb2.RunSpec(
                            overwrite_cache=self._overwrite_cache,
                            interruptible=wrappers_pb2.BoolValue(value=self._interruptible)
                            if self._interruptible is not None
                            else None,
                            annotations=annotations,
                            labels=labels,
                            envs=env_kv,
                            cluster=self._queue or task.queue,
                            raw_data_storage=raw_data_storage,
                            security_context=security_context,
                            cache_config=run_pb2.CacheConfig(
                                overwrite_cache=self._overwrite_cache,
                                cache_lookup_scope=_to_cache_lookup_scope(self._cache_lookup_scope)
                                if self._cache_lookup_scope
                                else None,
                            ),
                        ),
                    ),
                )
                return Run(pb2=resp.run)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise flyte.errors.RuntimeSystemError(
                        "SystemUnavailableError",
                        "Flyte system is currently unavailable. check your configuration, or the service status.",
                    ) from e
                elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                    raise flyte.errors.RuntimeUserError("InvalidArgumentError", e.details())
                elif e.code() == grpc.StatusCode.ALREADY_EXISTS:
                    # TODO maybe this should be a pass and return existing run?
                    raise flyte.errors.RuntimeUserError(
                        "RunAlreadyExistsError",
                        f"A run with the name '{self._name}' already exists. Please choose a different name.",
                    )
                else:
                    raise flyte.errors.RuntimeSystemError(
                        "RunCreationError",
                        f"Failed to create run: {e.details()}",
                    ) from e

        class DryRun(Run):
            def __init__(self, _task_spec, _inputs, _code_bundle):
                super().__init__(
                    pb2=run_definition_pb2.Run(
                        action=run_definition_pb2.Action(
                            id=identifier_pb2.ActionIdentifier(
                                name="a0",
                                run=identifier_pb2.RunIdentifier(name="dry-run"),
                            )
                        )
                    )
                )
                self.task_spec = _task_spec
                self.inputs = _inputs
                self.code_bundle = _code_bundle

        return DryRun(_task_spec=task_spec, _inputs=inputs, _code_bundle=code_bundle)

    @requires_storage
    @requires_initialization
    async def _run_hybrid(self, obj: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Run a task in hybrid mode. This means that the parent action will be run locally, but the child actions will be
        run in the cluster remotely. This is currently only used for testing,
        over the longer term we will productize this.
        """
        import flyte.report
        from flyte._code_bundle import build_code_bundle, build_pkl_bundle
        from flyte._deploy import build_images
        from flyte.models import RawDataPath
        from flyte.storage import ABFS, GCS, S3

        from ._internal import create_controller
        from ._internal.runtime.taskrunner import run_task

        cfg = get_init_config()

        if obj.parent_env is None:
            raise ValueError("Task is not attached to an environment. Please attach the task to an environment.")

        image_cache = await build_images.aio(cast(Environment, obj.parent_env()))

        code_bundle = None
        if self._name is not None:
            # Check if remote run service has this run name already and if exists, then extract the code bundle from it.
            code_bundle = await _get_code_bundle_for_run(name=self._name)

        if not code_bundle:
            if self._interactive_mode:
                code_bundle = await build_pkl_bundle(
                    obj,
                    upload_to_controlplane=not self._dry_run,
                    copy_bundle_to=self._copy_bundle_to,
                )
            else:
                if self._copy_files != "none":
                    code_bundle = await build_code_bundle(
                        from_dir=cfg.root_dir,
                        dryrun=self._dry_run,
                        copy_bundle_to=self._copy_bundle_to,
                        copy_style=self._copy_files,
                    )
                else:
                    code_bundle = None

        version = self._version or (
            code_bundle.computed_version if code_bundle and code_bundle.computed_version else None
        )
        if not version:
            raise ValueError("Version is required when running a task")

        project = cfg.project
        domain = cfg.domain
        org = cfg.org
        action_name = "a0"
        run_name = self._name
        random_id = str(uuid.uuid4())[:6]

        controller = create_controller("remote", endpoint="localhost:8090", insecure=True)
        action = ActionID(name=action_name, run_name=run_name, project=project, domain=domain, org=org)

        inputs = obj.native_interface.convert_to_kwargs(*args, **kwargs)
        # TODO: Ideally we should get this from runService
        # The API should be:
        # create new run, from run, in mode hybrid -> new run id, output_base, raw_data_path, inputs_path
        storage = get_storage()
        if type(storage) not in (S3, GCS, ABFS):
            raise ValueError(f"Unsupported storage type: {type(storage)}")
        if self._run_base_dir is None:
            raise ValueError(
                "Raw data path is required when running task, please set it in the run context:",
                " flyte.with_runcontext(run_base_dir='s3://bucket/metadata/outputs')",
            )
        output_path = self._run_base_dir
        run_base_dir = self._run_base_dir
        raw_data_path = f"{output_path}/rd/{random_id}"
        raw_data_path_obj = RawDataPath(path=raw_data_path)
        checkpoint_path = f"{raw_data_path}/checkpoint"
        prev_checkpoint = f"{raw_data_path}/prev_checkpoint"
        checkpoints = Checkpoints(checkpoint_path, prev_checkpoint)

        async def _run_task() -> Tuple[Any, Optional[Exception]]:
            ctx = internal_ctx()
            tctx = TaskContext(
                action=action,
                checkpoints=checkpoints,
                code_bundle=code_bundle,
                output_path=output_path,
                version=version if version else "na",
                raw_data_path=raw_data_path_obj,
                compiled_image_cache=image_cache,
                run_base_dir=run_base_dir,
                report=flyte.report.Report(name=action.name),
                custom_context=self._custom_context,
            )
            async with ctx.replace_task_context(tctx):
                return await run_task(tctx=tctx, controller=controller, task=obj, inputs=inputs)

        outputs, err = await contextual_run(_run_task)
        if err:
            raise err
        return outputs

    async def _run_local(self, obj: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Run:
        from flyteidl2.common import identifier_pb2

        from flyte._internal.controllers import create_controller
        from flyte._internal.controllers._local_controller import LocalController
        from flyte.remote import Run
        from flyte.report import Report

        controller = cast(LocalController, create_controller("local"))

        if self._name is None:
            action = ActionID.create_random()
        else:
            action = ActionID(name=self._name)

        metadata_path = self._metadata_path
        if metadata_path is None:
            metadata_path = pathlib.Path("/") / "tmp" / "flyte" / "metadata" / action.name
        else:
            metadata_path = pathlib.Path(metadata_path) / action.name
        output_path = metadata_path / "a0"
        if self._raw_data_path is None:
            path = pathlib.Path("/") / "tmp" / "flyte" / "raw_data" / action.name
            raw_data_path = RawDataPath(path=str(path))
        else:
            raw_data_path = RawDataPath(path=self._raw_data_path)

        ctx = internal_ctx()
        tctx = TaskContext(
            action=action,
            checkpoints=Checkpoints(
                prev_checkpoint_path=internal_ctx().raw_data.path,
                checkpoint_path=internal_ctx().raw_data.path,
            ),
            code_bundle=None,
            output_path=str(output_path),
            run_base_dir=str(metadata_path),
            version="na",
            raw_data_path=raw_data_path,
            compiled_image_cache=None,
            report=Report(name=action.name),
            mode="local",
            custom_context=self._custom_context,
        )

        with ctx.replace_task_context(tctx):
            # make the local version always runs on a different thread, returns a wrapped future.
            if obj._call_as_synchronous:
                fut = controller.submit_sync(obj, *args, **kwargs)
                awaitable = asyncio.wrap_future(fut)
                outputs = await awaitable
            else:
                outputs = await controller.submit(obj, *args, **kwargs)

        class _LocalRun(Run):
            def __init__(self, outputs: Tuple[Any, ...] | Any):
                from flyteidl2.workflow import run_definition_pb2

                self._outputs = outputs
                super().__init__(
                    pb2=run_definition_pb2.Run(
                        action=run_definition_pb2.Action(
                            id=identifier_pb2.ActionIdentifier(
                                name="a0",
                                run=identifier_pb2.RunIdentifier(name="dry-run"),
                            )
                        )
                    )
                )

            @property
            def url(self) -> str:
                return str(metadata_path)

            def wait(
                self,
                quiet: bool = False,
                wait_for: Literal["terminal", "running"] = "terminal",
            ):
                pass

            def outputs(self) -> R:
                return cast(R, self._outputs)

        return _LocalRun(outputs)

    @syncify
    async def run(
        self,
        task: TaskTemplate[P, Union[R, Run], F] | LazyEntity,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Union[R, Run]:
        """
        Run an async `@env.task` or `TaskTemplate` instance. The existing async context will be used.

        Example:
        ```python
        import flyte
        env = flyte.TaskEnvironment("example")

        @env.task
        async def example_task(x: int, y: str) -> str:
            return f"{x} {y}"

        if __name__ == "__main__":
            flyte.run(example_task, 1, y="hello")
        ```

        :param task: TaskTemplate instance `@env.task` or `TaskTemplate`
        :param args: Arguments to pass to the Task
        :param kwargs: Keyword arguments to pass to the Task
        :return: Run instance or the result of the task
        """
        from flyte.remote._task import LazyEntity

        if isinstance(task, LazyEntity) and self._mode != "remote":
            raise ValueError("Remote task can only be run in remote mode.")

        if not isinstance(task, TaskTemplate) and not isinstance(task, LazyEntity):
            raise TypeError(f"On Flyte tasks can be run, not generic functions or methods '{type(task)}'.")

        if self._mode == "remote":
            return await self._run_remote(task, *args, **kwargs)
        task = cast(TaskTemplate, task)
        if self._mode == "hybrid":
            return await self._run_hybrid(task, *args, **kwargs)

        # TODO We could use this for remote as well and users could simply pass flyte:// or s3:// or file://
        with internal_ctx().new_raw_data_path(
            raw_data_path=RawDataPath.from_local_folder(local_folder=self._raw_data_path)
        ):
            return await self._run_local(task, *args, **kwargs)


def with_runcontext(
    mode: Mode | None = None,
    *,
    name: Optional[str] = None,
    service_account: Optional[str] = None,
    version: Optional[str] = None,
    copy_style: CopyFiles = "loaded_modules",
    dry_run: bool = False,
    copy_bundle_to: pathlib.Path | None = None,
    interactive_mode: bool | None = None,
    raw_data_path: str | None = None,
    run_base_dir: str | None = None,
    overwrite_cache: bool = False,
    project: str | None = None,
    domain: str | None = None,
    env_vars: Dict[str, str] | None = None,
    labels: Dict[str, str] | None = None,
    annotations: Dict[str, str] | None = None,
    interruptible: bool | None = None,
    log_level: int | None = None,
    log_format: LogFormat = "console",
    disable_run_cache: bool = False,
    queue: Optional[str] = None,
    custom_context: Dict[str, str] | None = None,
    cache_lookup_scope: CacheLookupScope = "global",
) -> _Runner:
    """
    Launch a new run with the given parameters as the context.

    Example:
    ```python
    import flyte
    env = flyte.TaskEnvironment("example")

    @env.task
    async def example_task(x: int, y: str) -> str:
        return f"{x} {y}"

    if __name__ == "__main__":
        flyte.with_runcontext(name="example_run_id").run(example_task, 1, y="hello")
    ```

    :param mode: Optional The mode to use for the run, if not provided, it will be computed from flyte.init
    :param version: Optional The version to use for the run, if not provided, it will be computed from the code bundle
    :param name: Optional The name to use for the run
    :param service_account: Optional The service account to use for the run context
    :param copy_style: Optional The copy style to use for the run context
    :param dry_run: Optional If true, the run will not be executed, but the bundle will be created
    :param copy_bundle_to: When dry_run is True, the bundle will be copied to this location if specified
    :param interactive_mode: Optional, can be forced to True or False.
         If not provided, it will be set based on the current environment. For example Jupyter notebooks are considered
         interactive mode, while scripts are not. This is used to determine how the code bundle is created.
    :param raw_data_path: Use this path to store the raw data for the run for local and remote, and can be used to
         store raw data in specific locations.
    :param run_base_dir: Optional The base directory to use for the run. This is used to store the metadata for the run,
     that is passed between tasks.
    :param overwrite_cache: Optional If true, the cache will be overwritten for the run
    :param project: Optional The project to use for the run
    :param domain: Optional The domain to use for the run
    :param env_vars: Optional Environment variables to set for the run
    :param labels: Optional Labels to set for the run
    :param annotations: Optional Annotations to set for the run
    :param interruptible: Optional If true, the run can be scheduled on interruptible instances and false implies
        that all tasks in the run should only be scheduled on non-interruptible instances. If not specified the
        original setting on all tasks is retained.
    :param log_level: Optional Log level to set for the run. If not provided, it will be set to the default log level
        set using `flyte.init()`
    :param log_format: Optional Log format to set for the run. If not provided, it will be set to the default log format
    :param disable_run_cache: Optional If true, the run cache will be disabled. This is useful for testing purposes.
    :param queue: Optional The queue to use for the run. This is used to specify the cluster to use for the run.
    :param custom_context: Optional global input context to pass to the task. This will be available via
        get_custom_context() within the task and will automatically propagate to sub-tasks.
        Acts as base/default values that can be overridden by context managers in the code.
    :param cache_lookup_scope: Optional Scope to use for the run. This is used to specify the scope to use for cache
        lookups. If not specified, it will be set to the default scope (global unless overridden at the system level).

    :return: runner
    """
    if mode == "hybrid" and not name and not run_base_dir:
        raise ValueError("Run name and run base dir are required for hybrid mode")
    if copy_style == "none" and not version:
        raise ValueError("Version is required when copy_style is 'none'")
    return _Runner(
        force_mode=mode,
        name=name,
        service_account=service_account,
        version=version,
        copy_style=copy_style,
        dry_run=dry_run,
        copy_bundle_to=copy_bundle_to,
        interactive_mode=interactive_mode,
        raw_data_path=raw_data_path,
        run_base_dir=run_base_dir,
        overwrite_cache=overwrite_cache,
        env_vars=env_vars,
        labels=labels,
        annotations=annotations,
        interruptible=interruptible,
        project=project,
        domain=domain,
        log_level=log_level,
        log_format=log_format,
        disable_run_cache=disable_run_cache,
        queue=queue,
        custom_context=custom_context,
        cache_lookup_scope=cache_lookup_scope,
    )


@syncify
async def run(task: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Union[R, Run]:
    """
    Run a task with the given parameters
    :param task: task to run
    :param args: args to pass to the task
    :param kwargs: kwargs to pass to the task
    :return: Run | Result of the task
    """
    # using syncer causes problems
    return await _Runner().run.aio(task, *args, **kwargs)  # type: ignore
