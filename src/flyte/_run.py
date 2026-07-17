from __future__ import annotations

import asyncio
import contextvars
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from flyte._context import Context, contextual_run, internal_ctx
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
    ActionPhase,
    CheckpointPaths,
    CodeBundle,
    RawDataPath,
    SerializationContext,
    TaskContext,
)
from flyte.syncify import syncify

# ``flyte.storage.join`` is imported lazily inside the one method that needs it so
# ``import flyte`` does not eagerly pull fsspec/obstore/etc. into the startup path.
from ._constants import FLYTE_SYS_PATH

if TYPE_CHECKING:
    from flyte.notify import NamedRule, Notification
    from flyte.remote import Run
    from flyte.remote._task import LazyEntity

    from ._code_bundle import CopyFiles
    from ._internal.imagebuild.image_builder import ImageCache

Mode = Literal["local", "remote", "hybrid"]
CacheLookupScope = Literal["global", "project-domain"]


# ContextVar for run mode - thread-safe and coroutine-safe alternative to a global variable.
# This allows offloaded types (files, directories, dataframes) to be aware of the run mode
# for controlling auto-uploading behavior (only enabled in remote mode).
_run_mode_var: contextvars.ContextVar[Mode | None] = contextvars.ContextVar("run_mode", default=None)


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
        spec = run_details.action_details.pb2.task
        return extract_code_bundle(spec)
    return None


def _get_main_run_mode() -> Mode | None:
    """Get the current run mode from the context variable."""
    return _run_mode_var.get()


def _ambient_image_cache() -> ImageCache | None:
    """Image cache transported into this process by the run that launched it, if any.

    Inside a task pod, the parent run's deploy already built every environment in its plan
    and shipped the resolved URIs here (``TaskContext.compiled_image_cache``). A nested
    ``flyte.run(...)`` submitted from task code seeds image resolution with it so
    already-built environments are never re-resolved in-cluster — where the predicted URI
    can differ from where the builder actually pushed (e.g. the remote builder's system
    registry), and where no builder may be available at all. Same-run child calls already
    reuse this cache via the controller; this extends that behavior to nested runs.
    Returns None on the driver (no task context), leaving behavior unchanged there.
    """
    tctx = internal_ctx().data.task_context
    return tctx.compiled_image_cache if tctx else None


def _to_cache_lookup_scope(scope: CacheLookupScope | None = None):
    """Map the SDK cache-lookup-scope literal onto its RunSpec enum value."""
    from flyteidl2.task import run_pb2

    if scope == "global":
        return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_GLOBAL
    elif scope == "project-domain":
        return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_PROJECT_DOMAIN
    elif scope is None:
        return run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_UNSPECIFIED
    else:
        raise ValueError(f"Unknown cache lookup scope: {scope}")


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
        run_start_time: Optional[datetime] = None,
        overwrite_cache: bool = False,
        project: str | None = None,
        domain: str | None = None,
        env_vars: Dict[str, str] | None = None,
        labels: Dict[str, str] | None = None,
        annotations: Dict[str, str] | None = None,
        interruptible: bool | None = None,
        log_level: int | None = None,
        log_format: LogFormat = "console",
        user_log_level: int | None = None,
        reset_root_logger: bool = False,
        disable_run_cache: bool = False,
        queue: Optional[str] = None,
        max_action_concurrency: int | None = None,
        custom_context: Dict[str, str] | None = None,
        notifications: NamedRule | Notification | Tuple[Notification, ...] | None = None,
        cache_lookup_scope: CacheLookupScope = "global",
        preserve_original_types: bool | None = None,
        debug: bool = False,
        recover: bool | str | None = False,
        recover_force_rerun_actions: Sequence[str] | None = None,
        _tracker: Any = None,
        _bundle_relative_paths: tuple[str, ...] | None = None,
        _bundle_from_dir: pathlib.Path | None = None,
    ):
        from flyte._tools import ipython_check

        self._tracker = _tracker
        self._bundle_relative_paths = _bundle_relative_paths
        self._bundle_from_dir = _bundle_from_dir
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
        self._interactive_mode = interactive_mode or ipython_check()
        self._raw_data_path = raw_data_path
        self._metadata_path = metadata_path
        self._run_base_dir = run_base_dir
        self._run_start_time = run_start_time
        self._overwrite_cache = overwrite_cache
        self._project = project
        self._domain = domain
        self._env_vars = env_vars
        self._labels = labels
        self._annotations = annotations
        self._interruptible = interruptible
        self._log_level = log_level
        self._log_format = log_format
        self._user_log_level = user_log_level
        self._reset_root_logger = reset_root_logger
        self._disable_run_cache = disable_run_cache
        self._queue = queue
        self._max_action_concurrency = max_action_concurrency
        self._notifications = notifications
        self._custom_context = custom_context or {}
        self._cache_lookup_scope = cache_lookup_scope
        self._preserve_original_types = (
            preserve_original_types if preserve_original_types is not None else self._interactive_mode
        )
        self._debug = debug
        # Recover (reuse a prior run's succeeded actions). `True` = recover from the run being rerun;
        # a run-name string = recover from that named run (the only form valid on a plain run()).
        # Carried on RunSpec.relation/recover; remote-only; gated in _apply_overrides until the
        # flyteidl2 field + backend ship. See _resolve_recover_ref.
        self._recover = recover
        # Escape hatch: actions that must re-execute in the recovery run even if they succeeded
        # in the source run (RunSpec.recover.force_rerun_actions). Only valid with recover.
        self._recover_force_rerun_actions = tuple(recover_force_rerun_actions or ())
        if self._recover_force_rerun_actions and not self._recover:
            raise ValueError("recover_force_rerun_actions requires recover to be set")

    def _resolve_recover_ref(self, rerun_run_name: str | None) -> str | None:
        """Resolve `self._recover` to the reference run name to recover from (or None).

        `False`/`None` -> no recover. `True` -> the run being rerun (`rerun_run_name`); invalid on a
        plain `run()` where there is no rerun target. A string -> that named run.
        """
        r = self._recover
        if not r:
            return None
        if r is True:
            if rerun_run_name is None:
                raise ValueError(
                    "recover=True is only valid with rerun() (it recovers from the run being rerun). "
                    "To recover a fresh run() from a prior run, pass its name: "
                    "with_runcontext(recover='<run-name>').run(...)"
                )
            return rerun_run_name
        return r  # explicit run-name string

    def _resolve_related_to(self, source_run: Any = None) -> Any | None:
        """Resolve the provenance pointer (``RunSpec.related_to``) for the run being created.

        An explicit ``source_run`` (the rerun path) wins; otherwise, when invoked from inside a
        running remote task container (``TaskContext.is_in_cluster()``), the current run is the
        parent. The effective source scope is the source's ids with empty fields inherited from
        the init config; the pointer is stamped only when that scope equals the new run's target
        scope exactly and all four id fields are non-empty (the server requires min_len=1 on
        each, and related_to is same-org/project/domain as the new run by contract). Returns
        None otherwise — a provenance pointer must never fail run creation. Pure resolution,
        no I/O.
        """
        from flyteidl2.common import identifier_pb2

        if source_run is not None:
            org, project, domain, name = source_run.org, source_run.project, source_run.domain, source_run.name
        else:
            tctx = internal_ctx().data.task_context
            if tctx is None or not tctx.is_in_cluster():
                return None
            action = tctx.action
            org, project, domain, name = action.org or "", action.project or "", action.domain or "", action.run_name

        cfg = get_init_config()
        org = org or cfg.org or ""
        project = project or cfg.project or ""
        domain = domain or cfg.domain or ""

        target = (cfg.org or "", self._project or cfg.project or "", self._domain or cfg.domain or "")
        if (org, project, domain) != target:
            logger.debug(f"Skipping RunSpec.related_to: source scope {(org, project, domain)} != target {target}")
            return None
        if not (org and project and domain and name):
            logger.debug("Skipping RunSpec.related_to: incomplete source run identifier")
            return None
        return identifier_pb2.RunIdentifier(org=org, project=project, domain=domain, name=name)

    async def _build_task_spec_from_template(self, obj: TaskTemplate[P, R, F]) -> Tuple[Any, Any, str]:
        """Build ``(task_spec, code_bundle, version)`` from a local ``TaskTemplate``.

        Shared by ``_run_remote`` (local-task branch) and ``rerun`` with substitute code, so both
        get identical fidelity (copy_files / dry_run / interactive_mode / include-files). Heavy
        imports stay function-local to keep ``import flyte`` cheap. The built ``image_cache`` is
        folded into the returned ``task_spec`` via the serialization context, so it is not returned.
        """
        import flyte.report
        from flyte._image import Image, resolve_code_bundle_layer

        from ._code_bundle import build_code_bundle, build_code_bundle_from_relative_paths, build_pkl_bundle
        from ._code_bundle._includes import collect_env_include_files
        from ._deploy import build_images, plan_deploy
        from ._internal.runtime.task_serde import translate_task_to_wire

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        if obj.parent_env is None:
            raise ValueError("Task is not attached to an environment. Please attach the task to an environment")

        # Resolve any CodeBundleLayer layers before building images.
        # Must cover the parent env AND all depends_on envs (recursively)
        # so that _build_images can compute the content hash for every image.
        parent_env = cast(Environment, obj.parent_env())
        plan_envs = list(plan_deploy(parent_env)[0].envs.values())
        for _env in plan_envs:
            if isinstance(_env.image, Image):
                _env.image = resolve_code_bundle_layer(_env.image, self._copy_files, pathlib.Path(cfg.root_dir))

        if not self._dry_run:
            # Seed with the cache transported from the launching run (if we're inside a task
            # pod) so already-built environments reuse their pushed URIs instead of being
            # re-resolved in-cluster. No-op on the driver.
            image_cache = await build_images.aio(parent_env, seed_cache=_ambient_image_cache())
        else:
            image_cache = None

        include_files = collect_env_include_files(plan_envs)
        skip_cache = self._disable_run_cache

        if self._interactive_mode:
            if include_files:
                raise ValueError(
                    "Environment.include is not supported in interactive/pkl runs. "
                    "Run from a file or remove `include` from the environment."
                )
            code_bundle = await build_pkl_bundle(
                obj,
                upload_to_controlplane=not self._dry_run,
                copy_bundle_to=self._copy_bundle_to,
            )
        elif self._copy_files == "custom":
            if not self._bundle_relative_paths or not self._bundle_from_dir:
                raise ValueError("copy_style='custom' requires _bundle_relative_paths and _bundle_from_dir")
            merged_paths = tuple(self._bundle_relative_paths) + include_files
            code_bundle = await build_code_bundle_from_relative_paths(
                merged_paths,
                from_dir=self._bundle_from_dir,
                dryrun=self._dry_run,
                copy_bundle_to=self._copy_bundle_to,
                skip_cache=skip_cache,
            )
        elif self._copy_files != "none":
            code_bundle = await build_code_bundle(
                from_dir=cfg.root_dir,
                dryrun=self._dry_run,
                copy_bundle_to=self._copy_bundle_to,
                copy_style=self._copy_files,
                additional_files=include_files,
                skip_cache=skip_cache,
            )
        elif include_files:
            code_bundle = await build_code_bundle_from_relative_paths(
                include_files,
                from_dir=pathlib.Path(cfg.root_dir),
                dryrun=self._dry_run,
                copy_bundle_to=self._copy_bundle_to,
                skip_cache=skip_cache,
            )
        else:
            code_bundle = None

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
        action = ActionID(name="{{.actionName}}", run_name="{{.runName}}", project=project, domain=domain, org=cfg.org)
        tctx = TaskContext(
            action=action,
            code_bundle=code_bundle,
            output_path="",
            version=version or "na",
            raw_data_path=RawDataPath(path=""),
            compiled_image_cache=image_cache,
            run_base_dir="",
            report=flyte.report.Report(name=action.name),
            custom_context=self._custom_context,
        )
        task_spec = translate_task_to_wire(obj, s_ctx, default_inputs=None, task_context=tctx)
        return task_spec, code_bundle, version

    def _build_env_dict(self) -> Dict[str, str]:
        """Assemble the runtime env dict from runner config.

        User-supplied ``env_vars`` plus the always-injected LOG_* / debug / rust-controller /
        sys-path keys. Shared by the fresh-build and inherited (rerun) RunSpec paths so debug's
        ssh-env injection and the log settings apply identically. Returns a fresh dict (never
        mutates ``self._env_vars``).
        """
        cfg = get_init_config()
        env: Dict[str, str] = dict(self._env_vars or {})
        if env.get("LOG_LEVEL") is None:
            env["LOG_LEVEL"] = str(self._log_level) if self._log_level else str(logger.getEffectiveLevel())
        env["LOG_FORMAT"] = self._log_format
        if self._user_log_level is not None:
            env["USER_LOG_LEVEL"] = str(self._user_log_level)
        if self._reset_root_logger:
            env["FLYTE_RESET_ROOT_LOGGER"] = "1"
        if self._debug:
            env["_F_E_VS"] = "1"

        use_rust_controller_env_var = os.getenv("_F_USE_RUST_CONTROLLER")
        if use_rust_controller_env_var:
            env["_F_USE_RUST_CONTROLLER"] = use_rust_controller_env_var

        # These paths will be appended to sys.path at runtime.
        if cfg.sync_local_sys_paths:
            root_dir_abs = pathlib.Path(cfg.root_dir).resolve()
            env[FLYTE_SYS_PATH] = ":".join(
                f"./{pathlib.Path(p).relative_to(root_dir_abs)}"
                for p in sys.path
                if pathlib.Path(p).is_relative_to(root_dir_abs)
            )

        # TODO: Remove once the actions service is the default and this env var is no longer needed.
        if os.getenv("_U_USE_ACTIONS") == "1":
            env["_U_USE_ACTIONS"] = "1"
        return env

    def _resolve_run_target(self, project: str | None, domain: str | None, org: str | None):
        """Resolve the create-run target: a RunIdentifier when a name is set, else a ProjectIdentifier."""
        from flyteidl2.common import identifier_pb2

        if self._name:
            return (
                identifier_pb2.RunIdentifier(project=project, domain=domain, org=org, name=self._name or None),
                None,
            )
        return None, identifier_pb2.ProjectIdentifier(name=project, domain=domain, organization=org)

    def _apply_overrides(
        self, base: Any, *, task: Any = None, recover_ref: str | None = None, related_to: Any = None
    ) -> Any:
        """Build the ``RunSpec`` for ``create_run``.

        ``base is None`` -> a fresh spec from runner config (the run / recover path).
        ``base`` set     -> deep-copy a prior run's ``RunSpec`` and merge runner overrides by key
        (the rerun path: env merge + explicitly-set field overrides). Pure proto assembly, no I/O.
        This is the single place runner config maps onto a ``RunSpec``. ``recover_ref`` is the already-
        resolved reference run to recover from (see ``_resolve_recover_ref``), or None. ``related_to``
        is the already-resolved provenance pointer (see ``_resolve_related_to``), or None.
        """
        from flyteidl2.core import literals_pb2, security_pb2
        from flyteidl2.task import run_pb2
        from google.protobuf import wrappers_pb2

        env = self._build_env_dict()
        if base is not None:
            # Inherit the prior run's env as the floor; runner overrides win.
            merged = {kv.key: kv.value for kv in base.envs.values}
            merged.update(env)
            env = merged

        kv_pairs: List[literals_pb2.KeyValuePair] = []
        for k, v in env.items():
            if not isinstance(v, str):
                raise ValueError(f"Environment variable {k} must be a string, got {type(v)}")
            kv_pairs.append(literals_pb2.KeyValuePair(key=k, value=v))
        env_kv = run_pb2.Envs(values=kv_pairs)

        notification_rule_name = None
        notification_rules = None
        if self._notifications:
            from flyte._internal.runtime.notifications_serde import resolve_notification_settings

            notification_rule_name, notification_rules = resolve_notification_settings(self._notifications)

        if base is None:
            raw_data_storage = (
                run_pb2.RawDataStorage(raw_data_prefix=self._raw_data_path) if self._raw_data_path else None
            )
            security_context = (
                security_pb2.SecurityContext(run_as=security_pb2.Identity(k8s_service_account=self._service_account))
                if self._service_account
                else None
            )
            run_spec = run_pb2.RunSpec(
                overwrite_cache=self._overwrite_cache,
                interruptible=wrappers_pb2.BoolValue(value=self._interruptible)
                if self._interruptible is not None
                else None,
                annotations=run_pb2.Annotations(values=self._annotations),
                labels=run_pb2.Labels(values=self._labels),
                envs=env_kv,
                cluster=self._queue or (task.queue if task is not None else ""),
                max_action_concurrency=self._max_action_concurrency or 0,
                raw_data_storage=raw_data_storage,
                run_base_dir=self._run_base_dir or "",
                security_context=security_context,
                cache_config=run_pb2.CacheConfig(
                    overwrite_cache=self._overwrite_cache,
                    cache_lookup_scope=_to_cache_lookup_scope(self._cache_lookup_scope)
                    if self._cache_lookup_scope
                    else None,
                ),
                notification_rule_name=notification_rule_name,
                notification_rules=notification_rules,
            )
        else:
            # Deep-copy the fetched spec (it is shared/cached on the RunDetails); never mutate in place.
            run_spec = run_pb2.RunSpec()
            run_spec.CopyFrom(base)
            run_spec.envs.CopyFrom(env_kv)
            if self._interruptible is not None:
                run_spec.interruptible.CopyFrom(wrappers_pb2.BoolValue(value=self._interruptible))
            if self._overwrite_cache:
                run_spec.overwrite_cache = True
                run_spec.cache_config.overwrite_cache = True
            if self._labels:
                for k, v in self._labels.items():
                    run_spec.labels.values[k] = v
            if self._annotations:
                for k, v in self._annotations.items():
                    run_spec.annotations.values[k] = v
            if self._cache_lookup_scope:
                run_spec.cache_config.cache_lookup_scope = _to_cache_lookup_scope(self._cache_lookup_scope)
            if self._max_action_concurrency:
                run_spec.max_action_concurrency = self._max_action_concurrency
            if self._queue:
                # TODO: cluster is being renamed to queue
                run_spec.cluster = self._queue
            if self._service_account:
                run_spec.security_context.CopyFrom(
                    security_pb2.SecurityContext(
                        run_as=security_pb2.Identity(k8s_service_account=self._service_account)
                    )
                )
            if notification_rule_name:
                run_spec.notification_rule_name = notification_rule_name
            if notification_rules:
                run_spec.notification_rules.CopyFrom(notification_rules)

        # recover: on the wire a recovery run is RunSpec.relation with RELATION_TYPE_RECOVER pointing
        # at the reference run (resolved in the current org/project/domain); RunSpec.recover carries
        # only optional extras (force_rerun_actions). Gated on the flyteidl2 build having the new
        # fields until the backend ships.
        if "relation" in run_pb2.RunSpec.DESCRIPTOR.fields_by_name:
            # Never inherit a rerun base's stale (grandparent) pointers.
            run_spec.ClearField("relation")
            run_spec.ClearField("recover")
        if recover_ref:
            if "recover" not in run_pb2.RunSpec.DESCRIPTOR.fields_by_name:
                raise NotImplementedError(
                    "recover is not yet supported by this backend "
                    "(RunSpec.recover is unavailable in this flyteidl2 build)."
                )
            from flyteidl2.common import identifier_pb2
            from flyteidl2.common import run_pb2 as common_run_pb2

            cfg = get_init_config()
            run_spec.relation.CopyFrom(
                common_run_pb2.Relation(
                    relation_type=common_run_pb2.RELATION_TYPE_RECOVER,
                    related_to=identifier_pb2.RunIdentifier(
                        org=cfg.org or "",
                        project=self._project or cfg.project or "",
                        domain=self._domain or cfg.domain or "",
                        name=recover_ref,
                    ),
                )
            )
            if self._recover_force_rerun_actions:
                # Escape hatch: these actions re-execute even though they succeeded in the
                # source run. A listed parent re-enqueues its children (list them too to force
                # the whole subtree); unknown names are ignored server-side.
                run_spec.recover.CopyFrom(run_pb2.Recover(force_rerun_actions=list(self._recover_force_rerun_actions)))

        # related_to: implicit provenance (rerun source, or the invoking in-cluster run).
        run_spec.ClearField("related_to")  # never inherit a rerun base's stale (grandparent) pointer
        if related_to is not None:
            run_spec.related_to.CopyFrom(related_to)

        return run_spec

    async def _submit_remote(
        self, *, task_spec: Any, task_id: Any, proto_inputs: Any, run_spec: Any, run_id: Any, project_id: Any
    ) -> Run:
        """Upload inputs and create the run. The single network call site for remote submission.

        Consumes an already-built ``run_spec`` (see ``_apply_overrides``), raw proto ``inputs``
        (``flyteidl2.task.Inputs``), and a task by reference (``task_id``) or by value
        (``task_spec``); shared by ``_run_remote`` and ``rerun``.
        """
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from flyteidl2.dataproxy import dataproxy_service_pb2
        from flyteidl2.workflow import run_service_pb2

        import flyte.errors
        from flyte.remote import Run

        try:
            upload_req = dataproxy_service_pb2.UploadInputsRequest(inputs=proto_inputs)
            # Pass the explicit run_base_dir so the offloaded inputs are written under the
            # same base the CreateRun below resolves (RunSpec.run_base_dir, set in _apply_overrides).
            # When unset the server falls back to settings/cluster default in both paths.
            if self._run_base_dir:
                upload_req.base_dir = self._run_base_dir
            # Reference an already-registered task by id; otherwise upload the full spec.
            if task_id is not None:
                upload_req.task_id.CopyFrom(task_id)
            else:
                upload_req.task_spec.CopyFrom(task_spec)
            if run_id is not None:
                upload_req.run_id.CopyFrom(run_id)
            else:
                upload_req.project_id.CopyFrom(project_id)

            upload_resp = await get_client().dataproxy_service.upload_inputs(upload_req)

            create_req = run_service_pb2.CreateRunRequest(
                run_id=run_id,
                project_id=project_id,
                offloaded_input_data=upload_resp.offloaded_input_data,
                run_spec=run_spec,
            )
            # Reference an already-registered task by id; otherwise send the full spec.
            if task_id is not None:
                create_req.task_id.CopyFrom(task_id)
            else:
                create_req.task_spec.CopyFrom(task_spec)

            resp = await get_client().run_service.create_run(create_req)
            return Run(pb2=resp.run, _preserve_original_types=self._preserve_original_types)
        except ConnectError as e:
            if e.code == Code.UNAVAILABLE:
                raise flyte.errors.RuntimeSystemError(
                    "SystemUnavailableError",
                    "Flyte system is currently unavailable. check your configuration, or the service status.",
                ) from e
            elif e.code == Code.INVALID_ARGUMENT:
                raise flyte.errors.RuntimeUserError("InvalidArgumentError", e.message)
            elif e.code == Code.ALREADY_EXISTS:
                # TODO maybe this should be a pass and return existing run?
                raise flyte.errors.RuntimeUserError(
                    "RunAlreadyExistsError",
                    f"A run with the name '{self._name}' already exists. Please choose a different name.",
                )
            else:
                raise flyte.errors.RuntimeSystemError(
                    "RunCreationError",
                    f"Failed to create run: {e.message}",
                ) from e

    @requires_initialization
    async def _run_remote(self, obj: TaskTemplate[P, R, F] | LazyEntity, *args: P.args, **kwargs: P.kwargs) -> Run:
        from flyteidl2.common import identifier_pb2
        from flyteidl2.workflow import run_definition_pb2

        import flyte.errors
        from flyte.remote import Run
        from flyte.remote._task import LazyEntity, TaskDetails

        from ._internal.runtime.convert import convert_from_native_to_inputs

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        task: TaskTemplate[P, R, F] | TaskDetails
        task_id = None
        if isinstance(obj, (LazyEntity, TaskDetails)):
            if isinstance(obj, LazyEntity):
                task = await obj.fetch.aio()
            else:
                task = obj
            task_spec = task.pb2.spec
            # A fetched task is normally run by reference (task_id only). But if it was modified via
            # `.override(...)`, the local spec no longer matches the registered task, so we must send
            # the full spec instead. Setting task_id to None routes every downstream branch to the
            # spec path.
            task_id = None if task.overridden else task.pb2.task_id
            inputs = await convert_from_native_to_inputs(
                task.interface, *args, custom_context=self._custom_context, **kwargs
            )
            version = task.pb2.task_id.version
            code_bundle = None
        elif isinstance(obj, TaskTemplate):
            task = cast(TaskTemplate[P, R, F], obj)
            task_spec, code_bundle, version = await self._build_task_spec_from_template(obj)
            inputs = await convert_from_native_to_inputs(
                obj.native_interface, *args, custom_context=self._custom_context, **kwargs
            )
        else:
            raise ValueError(f"Not supported Task Type: {type(task)}")

        if not self._dry_run:
            if get_client() is None:
                # This can only happen, if the user forces flyte.run(mode="remote") without initializing the client
                raise flyte.errors.InitializationError(
                    "ClientNotInitializedError",
                    "user",
                    "flyte.run requires client to be initialized. "
                    "Call flyte.init() with a valid endpoint/api-key before using this function"
                    "or Call flyte.init_from_config() with a valid path to the config file",
                )
            run_id, project_id = self._resolve_run_target(project, domain, cfg.org)
            # Fill in task id inside the task template if it's not provided.
            # Maybe this should be done here, or the backend.
            # Only needed for locally-defined tasks; a fetched task sent by reference (task_id set)
            # is skipped here. An overridden fetched task (task_id None) already carries a
            # fully-populated id, so the `== ""` guards below leave it untouched.
            if task_id is None:
                if task_spec.task_template.id.project == "":
                    task_spec.task_template.id.project = project or ""
                if task_spec.task_template.id.domain == "":
                    task_spec.task_template.id.domain = domain or ""
                if task_spec.task_template.id.org == "":
                    task_spec.task_template.id.org = cfg.org or ""
                if task_spec.task_template.id.version == "":
                    task_spec.task_template.id.version = version

            run_spec = self._apply_overrides(
                None,
                task=task,
                recover_ref=self._resolve_recover_ref(None),
                related_to=self._resolve_related_to(),
            )
            return await self._submit_remote(
                task_spec=task_spec,
                task_id=task_id,
                proto_inputs=inputs.proto_inputs,
                run_spec=run_spec,
                run_id=run_id,
                project_id=project_id,
            )

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
        from flyte._code_bundle import build_code_bundle, build_code_bundle_from_relative_paths, build_pkl_bundle
        from flyte._deploy import build_images
        from flyte.models import RawDataPath
        from flyte.storage import ABFS, GCS, S3

        from ._internal import create_controller
        from ._internal.runtime.taskrunner import run_task

        cfg = get_init_config()

        if obj.parent_env is None:
            raise ValueError("Task is not attached to an environment. Please attach the task to an environment.")

        # Resolve any CodeBundleLayer layers before building images.
        # Must cover the parent env AND all depends_on envs (recursively)
        # so that _build_images can compute the content hash for every image.
        env = cast(Environment, obj.parent_env())
        from flyte._deploy import plan_deploy
        from flyte._image import Image, resolve_code_bundle_layer

        for _env in plan_deploy(env)[0].envs.values():
            if isinstance(_env.image, Image):
                _env.image = resolve_code_bundle_layer(_env.image, self._copy_files, pathlib.Path(cfg.root_dir))

        image_cache = await build_images.aio(cast(Environment, obj.parent_env()), seed_cache=_ambient_image_cache())

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
            elif self._copy_files == "custom":
                if not self._bundle_relative_paths or not self._bundle_from_dir:
                    raise ValueError("copy_style='custom' requires _bundle_relative_paths and _bundle_from_dir")
                code_bundle = await build_code_bundle_from_relative_paths(
                    self._bundle_relative_paths,
                    from_dir=self._bundle_from_dir,
                    dryrun=self._dry_run,
                    copy_bundle_to=self._copy_bundle_to,
                )
            elif self._copy_files != "none":
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

        # controller = create_controller("remote", endpoint="localhost:8090", insecure=True)
        controller = create_controller("rust", endpoint="localhost:8090", insecure=True)
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
        checkpoint_paths = CheckpointPaths(prev_checkpoint_path=prev_checkpoint, checkpoint_path=checkpoint_path)

        async def _run_task() -> Tuple[Any, Optional[Exception]]:
            ctx = internal_ctx()
            tctx_kwargs: Dict[str, Any] = {
                "action": action,
                "checkpoint_paths": checkpoint_paths,
                "code_bundle": code_bundle,
                "output_path": output_path,
                "version": version or "na",  # does na not work for rust?
                "raw_data_path": raw_data_path_obj,
                "compiled_image_cache": image_cache,
                "run_base_dir": run_base_dir,
                "report": flyte.report.Report(name=action.name),
                "custom_context": self._custom_context,
            }
            if self._run_start_time is not None:
                tctx_kwargs["run_start_time"] = self._run_start_time
            tctx = TaskContext(**tctx_kwargs)
            async with ctx.replace_task_context(tctx):
                return await run_task(tctx=tctx, controller=controller, task=obj, inputs=inputs)

        outputs, err = await contextual_run(_run_task)
        if err:
            raise err
        return outputs

    async def _send_local_notifications(
        self,
        *,
        phase: ActionPhase,
        task_name: str,
        run_name: str,
        error: str = "",
    ) -> None:
        """Send notifications locally. Never raises — failures are logged."""
        from flyte.notify._notifiers import NamedRule as _NamedRule
        from flyte.notify._sender import send_notifications

        notifications = self._notifications
        if isinstance(notifications, _NamedRule):
            logger.info("Skipping named rule %r in local mode", notifications.name)
            return

        await send_notifications(
            notifications,  # type: ignore[arg-type]
            phase=phase,
            task_name=task_name,
            run_name=run_name,
            error=error,
            project=self._project or "",
            domain=self._domain or "",
        )

    async def _run_local(self, obj: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Run:
        from flyteidl2.common import identifier_pb2
        from flyteidl2.task import common_pb2

        from flyte._internal.controllers import create_controller
        from flyte._internal.controllers._local_controller import LocalController
        from flyte.remote import ActionOutputs, Run
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

        from flyte.storage import join as storage_join

        ctx = internal_ctx()
        rd_base = raw_data_path.path
        tctx = TaskContext(
            action=action,
            checkpoint_paths=CheckpointPaths(
                prev_checkpoint_path=storage_join(rd_base, "prev_checkpoint"),
                checkpoint_path=storage_join(rd_base, "checkpoint"),
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
            disable_run_cache=self._disable_run_cache,
            run_start_time=self._run_start_time or datetime.now(timezone.utc),
        )

        if self._tracker is not None:
            ctx = Context(ctx.data.replace(tracker=self._tracker))

        from flyte._initialize import is_persistence_enabled
        from flyte._persistence._recorder import RunRecorder

        persist = is_persistence_enabled()
        run_name = action.run_name or action.name

        if persist:
            RunRecorder.initialize_persistence()

        recorder = RunRecorder(tracker=self._tracker, persist=persist, run_name=run_name)
        controller.set_recorder(recorder)

        recorder.record_root_start(task_name=obj.name)

        try:
            with ctx.replace_task_context(tctx):
                # make the local version always runs on a different thread, returns a wrapped future.
                if obj._call_as_synchronous:
                    fut = controller.submit_sync(obj, *args, **kwargs)
                    awaitable = asyncio.wrap_future(fut)
                    outputs = await awaitable
                else:
                    outputs = await controller.submit(obj, *args, **kwargs)
        except Exception as e:
            recorder.record_root_failure(error=str(e))
            if self._notifications:
                await self._send_local_notifications(
                    phase=ActionPhase.FAILED, task_name=obj.name, run_name=run_name, error=str(e)
                )
            raise
        else:
            recorder.record_root_complete()
            if self._notifications:
                await self._send_local_notifications(phase=ActionPhase.SUCCEEDED, task_name=obj.name, run_name=run_name)

        class _LocalRun(Run):
            def __init__(self, outputs: Tuple[Any, ...] | Any):
                from flyteidl2.workflow import run_definition_pb2

                self._outputs = ActionOutputs(
                    common_pb2.Outputs(), outputs if isinstance(outputs, tuple) else (outputs,)
                )
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

            @syncify
            async def wait(  # type: ignore[override]
                self,
                quiet: bool = False,
                wait_for: Literal["terminal", "running"] = "terminal",
            ) -> None:
                pass

            @syncify
            async def outputs(self) -> ActionOutputs:  # type: ignore[override]
                return self._outputs

        return _LocalRun(outputs)

    @syncify  # type: ignore[arg-type]
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
        from flyte.remote._task import LazyEntity, TaskDetails

        if isinstance(task, (LazyEntity, TaskDetails)) and self._mode != "remote":
            raise ValueError("Remote task can only be run in remote mode.")

        if not isinstance(task, TaskTemplate) and not isinstance(task, (LazyEntity, TaskDetails)):
            raise TypeError(f"On Flyte tasks can be run, not generic functions or methods '{type(task)}'.")

        # recover is an actions-service / RunSpec concern — remote-only. Fail fast rather than silently
        # ignoring it in local/hybrid mode.
        if self._recover and self._mode != "remote":
            raise ValueError("recover is only supported in remote mode")

        # Set the run mode in the context variable so that offloaded types (files, directories, dataframes)
        # can check the mode for controlling auto-uploading behavior (only enabled in remote mode).
        _run_mode_var.set(self._mode)

        try:
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
        finally:
            _run_mode_var.set(None)

    @syncify  # type: ignore[arg-type]
    async def rerun(
        self,
        run_name: str,
        action_name: str = "a0",
        task_template: TaskTemplate[P, R, F] | None = None,
        inputs: Dict[str, Any] | None = None,
    ) -> Run:
        """Re-run a prior run, returning a new `Run`.

        - `rerun("r1")` re-runs with the prior run's exact inputs, fetching its task spec from the
          platform (no local code needed).
        - `rerun("r1", inputs={"x": 2})` changes input parameters (converted against the fetched
          task interface).
        - `rerun("r1", task_template=fixed)` substitutes new code, validated against the original
          inputs (or `inputs` if given).

        The prior run's `RunSpec` is inherited and merged with this context's overrides
        (`with_runcontext(env_vars=..., interruptible=..., recover=...)` etc.), so debug/recover
        compose with rerun. Currently remote-only.

        :param run_name: Name of the prior run to re-run.
        :param action_name: Action within the prior run to source the task + inputs from (default `a0`).
        :param task_template: Optional task to substitute for the prior run's code.
        :param inputs: Optional native kwargs to change input parameters; omit to reuse prior inputs.
        :return: the new Run.
        """
        if self._mode != "remote":
            raise NotImplementedError(f"rerun is only supported in remote mode, got mode={self._mode!r}")

        from flyteidl2.dataproxy import dataproxy_service_pb2

        from flyte.remote._action import ActionDetails
        from flyte.remote._run import RunDetails

        from ._internal.runtime.convert import convert_from_native_to_inputs

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        run_details = await RunDetails.get.aio(name=run_name)
        base_run_spec = run_details.pb2.run_spec
        if action_name == "a0":
            action_details = run_details.action_details
        else:
            action_details = await ActionDetails.get.aio(run_name=run_name, name=action_name)

        # Task source: substitute a freshly-built local spec, or reuse the prior action's spec.
        if task_template is not None:
            task_spec, _code_bundle, version = await self._build_task_spec_from_template(task_template)
        else:
            if not action_details.pb2.HasField("task"):
                raise ValueError(f"Action {run_name}/{action_name} has no task spec to rerun.")
            task_spec = action_details.pb2.task
            version = task_spec.task_template.id.version

        # Inputs: reuse the prior raw proto inputs, or convert new native kwargs against the interface.
        if inputs:
            if task_template is not None:
                iface = task_template.native_interface
            else:
                from flyte.types._interface import guess_interface

                iface = guess_interface(task_spec.task_template.interface)
            converted = await convert_from_native_to_inputs(iface, custom_context=self._custom_context, **inputs)
            proto_inputs = converted.proto_inputs
        else:
            resp = await get_client().dataproxy_service.get_action_data(
                request=dataproxy_service_pb2.GetActionDataRequest(action_id=action_details.pb2.id)
            )
            proto_inputs = resp.inputs

        run_id, project_id = self._resolve_run_target(project, domain, cfg.org)

        # A freshly-built substitute spec may carry empty ids; fill them like _run_remote does.
        if task_template is not None:
            tt_id = task_spec.task_template.id
            if tt_id.project == "":
                tt_id.project = project or ""
            if tt_id.domain == "":
                tt_id.domain = domain or ""
            if tt_id.org == "":
                tt_id.org = cfg.org or ""
            if tt_id.version == "":
                tt_id.version = version

        # Provenance: the run being rerun. Explicit source wins entirely over any ambient task
        # context (a rerun triggered from inside a task points at the rerun source, not the
        # invoking run). The fetched id can be empty in degenerate cases; fall back to run_name.
        from flyteidl2.common import identifier_pb2

        src = action_details.pb2.id.run
        related_to = self._resolve_related_to(
            identifier_pb2.RunIdentifier(org=src.org, project=src.project, domain=src.domain, name=src.name or run_name)
        )
        run_spec = self._apply_overrides(
            base_run_spec, recover_ref=self._resolve_recover_ref(run_name), related_to=related_to
        )
        return await self._submit_remote(
            task_spec=task_spec,
            task_id=None,
            proto_inputs=proto_inputs,
            run_spec=run_spec,
            run_id=run_id,
            project_id=project_id,
        )


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
    # TODO: will move onto RunSpec; for now accept as a run-context override (mainly for local simulation / tests).
    run_start_time: Optional[datetime] = None,
    overwrite_cache: bool = False,
    project: str | None = None,
    domain: str | None = None,
    env_vars: Dict[str, str] | None = None,
    labels: Dict[str, str] | None = None,
    annotations: Dict[str, str] | None = None,
    interruptible: bool | None = None,
    log_level: int | None = None,
    log_format: LogFormat = "console",
    user_log_level: int | None = None,
    reset_root_logger: bool = False,
    disable_run_cache: bool = False,
    queue: Optional[str] = None,
    max_action_concurrency: int | None = None,
    notifications: Notification | Tuple[Notification, ...] | None = None,
    custom_context: Dict[str, str] | None = None,
    cache_lookup_scope: CacheLookupScope = "global",
    preserve_original_types: bool = False,
    debug: bool = False,
    recover: bool | str | None = False,
    recover_force_rerun_actions: Sequence[str] | None = None,
    _tracker: Any = None,
) -> _Runner:
    """
    Launch a new run with the given parameters as the context.

    Example:
    ```python
    import flyte
    import flyte.notify as notify
    from flyte.models import ActionPhase

    env = flyte.TaskEnvironment("example")

    @env.task
    async def example_task(x: int, y: str) -> str:
        return f"{x} {y}"

    if __name__ == "__main__":
        flyte.with_runcontext(
            name="example_run_id",
            notifications=notify.Slack(
                on_phase=ActionPhase.FAILED,
                webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                message="Task failed: {run.error}",
            ),
        ).run(example_task, 1, y="hello")
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
    :param run_start_time: Optional UTC datetime at which the run was triggered. If not provided, defaults to
     ``datetime.now(timezone.utc)`` at TaskContext construction. Useful for local simulation/tests that need a
     deterministic timestamp. Accessible inside a task via ``flyte.ctx().run_start_time``.
    :param overwrite_cache: Optional If true, the cache will be overwritten for the run
    :param project: Optional The project to use for the run
    :param domain: Optional The domain to use for the run
    :param env_vars: Optional Environment variables to set for the run
    :param labels: Optional user-defined labels to attach to the run as KEY=VALUE pairs, used for
        filtering and organizing runs (e.g. ``flyte get run --with-label team=ml``)
    :param annotations: Optional Annotations to set for the run
    :param interruptible: Optional If true, the run can be scheduled on interruptible instances and false implies
        that all tasks in the run should only be scheduled on non-interruptible instances. If not specified the
        original setting on all tasks is retained.
    :param log_level: Optional Log level to set for the run. If not provided, it will be set to the default log level
        set using `flyte.init()`
    :param log_format: Optional Log format to set for the run. If not provided, it will be set to the default log format
    :param reset_root_logger: If true, the root logger will be preserved and not modified by Flyte.
    :param disable_run_cache: Optional If true, the run cache will be disabled. This is useful for testing purposes.
    :param queue: Optional The queue to use for the run. This is used to specify the cluster to use for the run.
    :param max_action_concurrency: Optional Maximum number of actions that can run concurrently within this run.
        Only applies to remote runs. If not provided, the platform default (configurable via the
        ``run.max_action_concurrency`` setting at org/domain/project scope) applies. Must be 0
        (platform default) or at least 2 — a value of 1 would deadlock the run, since the parent
        action holds a concurrency slot while waiting for its child actions.
    :param notifications: Optional Notification(s) to send when the run reaches specific execution phases.
        Accepts a single notification or a tuple of notifications. Supports Email, Slack, Teams, and Webhook types.
        See `flyte.notify` for available notification types and template variables.
    :param custom_context: Optional global input context to pass to the task. This will be available via
        get_custom_context() within the task and will automatically propagate to sub-tasks.
        Acts as base/default values that can be overridden by context managers in the code.
    :param cache_lookup_scope: Optional Scope to use for the run. This is used to specify the scope to use for cache
        lookups. If not specified, it will be set to the default scope (global unless overridden at the system level).
    :param preserve_original_types: Optional If true, the type engine will preserve original types (e.g., pd.DataFrame)
        when guessing python types from literal types. If false (default), it will return the generic
        flyte.io.DataFrame. This option is automatically set to True if interactive_mode is True unless overridden
        explicitly by this parameter.
    :param debug: Optional If true, the task will be run as a VSCode debug task, starting a code-server in the
        container so users can connect via the UI to interactively debug/run the task.
    :param recover: Recover (reuse a prior run's succeeded actions, re-running only what failed or
        changed). ``True`` recovers from the run being rerun — only valid with ``.rerun(...)``; a
        run-name string recovers from that named run and is the only form valid on ``.run(...)``.
        Remote-only. Requires a backend with recovery support.
    :param recover_force_rerun_actions: Optional names of actions that must re-execute in the
        recovery run even if they succeeded in the source run (escape hatch). A listed parent
        action re-enqueues its children — list them too to force the whole subtree; a listed
        condition re-pauses for a new signal. Unknown names are ignored. Only valid with
        ``recover``.
    :param _tracker: This is an internal only parameter used by the CLI to render the TUI.

    :return: runner

    """
    if mode == "hybrid" and not name and not run_base_dir:
        raise ValueError("Run name and run base dir are required for hybrid mode")
    if copy_style == "custom":
        raise ValueError("copy_style='custom' is not yet supported through with_runcontext.")
    if copy_style == "none" and not version:
        raise ValueError("Version is required when copy_style is 'none'")
    if max_action_concurrency is not None and (max_action_concurrency < 0 or max_action_concurrency == 1):
        raise ValueError(
            f"max_action_concurrency must be 0 (platform default) or at least 2, got {max_action_concurrency}. "
            "A value of 1 would deadlock the run: the parent action holds a concurrency slot while "
            "waiting for its child actions to run."
        )

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
        run_start_time=run_start_time,
        overwrite_cache=overwrite_cache,
        env_vars=env_vars,
        labels=labels,
        annotations=annotations,
        interruptible=interruptible,
        project=project,
        domain=domain,
        log_level=log_level,
        log_format=log_format,
        user_log_level=user_log_level,
        reset_root_logger=reset_root_logger,
        disable_run_cache=disable_run_cache,
        queue=queue,
        max_action_concurrency=max_action_concurrency,
        notifications=notifications,
        custom_context=custom_context,
        cache_lookup_scope=cache_lookup_scope,
        preserve_original_types=preserve_original_types,
        debug=debug,
        recover=recover,
        recover_force_rerun_actions=recover_force_rerun_actions,
        _tracker=_tracker,
    )


@syncify
async def run(task: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Run:
    """
    Run a task with the given parameters
    :param task: task to run
    :param args: args to pass to the task
    :param kwargs: kwargs to pass to the task
    :return: Run | Result of the task
    """
    # using syncer causes problems
    return await _Runner().run.aio(task, *args, **kwargs)  # type: ignore


@syncify
async def rerun(
    run_name: str,
    action_name: str = "a0",
    task_template: TaskTemplate[P, R, F] | None = None,
    **inputs: Any,
) -> Run:
    """Re-run a prior run, returning a new `Run`.

    `rerun("r1")` reuses the prior run's exact inputs (fetching its code from the platform);
    pass keyword inputs to change parameters (`rerun("r1", x=2)`), or `task_template=` to substitute
    code. Use `with_runcontext(...).rerun(...)` to apply run-context overrides (env_vars, recover, …).

    :param run_name: Name of the prior run to re-run.
    :param action_name: Action within the prior run to source the task + inputs from (default `a0`).
    :param task_template: Optional task to substitute for the prior run's code.
    :param inputs: Optional native keyword inputs to change parameters; omit to reuse prior inputs.
    :return: the new Run.
    """
    return await _Runner().rerun.aio(run_name, action_name, task_template, inputs=inputs or None)
