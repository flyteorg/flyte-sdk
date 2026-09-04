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
from flyte._logging import LogFormat, logger, user_logger
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
from ._sentry import track_operation

if TYPE_CHECKING:
    from flyteidl2.core import artifact_id_pb2

    from flyte.notify import NamedRule, Notification
    from flyte.remote import Run
    from flyte.remote._task import LazyEntity
    from flyte.remote._trigger import Trigger as RemoteTrigger
    from flyte.remote._trigger import TriggerDetails

    from ._code_bundle import CopyFiles
    from ._internal.imagebuild.image_builder import ImageCache

    # A "source" records where an unwrapped value came from: the artifact's typed identity for
    # a plain artifact argument, or (element_index, identity) pairs for artifacts inside a list.
    _ArtifactSource = Union[artifact_id_pb2.ArtifactVersionId, List[Tuple[int, artifact_id_pb2.ArtifactVersionId]]]

Mode = Literal["local", "remote", "hybrid"]
CacheLookupScope = Literal["global", "project-domain"]


# ContextVar for run mode - thread-safe and coroutine-safe alternative to a global variable.
# This allows offloaded types (files, directories, dataframes) to be aware of the run mode
# for controlling auto-uploading behavior (only enabled in remote mode).
_run_mode_var: contextvars.ContextVar[Mode | None] = contextvars.ContextVar("run_mode", default=None)


async def _unwrap_artifact_value(value: Any) -> Tuple[Any, _ArtifactSource | None]:
    """
    Unwrap a single `flyte.remote.Artifact` (or a list containing artifacts) into the
    python value stored in its literal, which is what tasks actually consume.
    Non-artifact values are returned unchanged. Also returns the artifact source (tracker
    string, or per-element trackers for lists) so callers can record provenance.

    This is the local/hybrid path, which runs the task in-process and therefore needs real
    python values. Remote submission does not go through here: it binds the artifact's stored
    literal directly (`convert.bind_artifact_literals`).
    """
    # Imported lazily so ``import flyte`` does not eagerly pull in the remote package.
    from flyte.remote import Artifact

    if isinstance(value, Artifact):
        return await value.to_python(), value.artifact_version_id
    if isinstance(value, list) and len(value) > 0:
        ids = [(i, item.artifact_version_id) for i, item in enumerate(value) if isinstance(item, Artifact)]
        if ids:
            unwrapped = [await item.to_python() if isinstance(item, Artifact) else item for item in value]
            return unwrapped, ids
    return value, None


async def _unwrap_artifacts(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any], Dict[Union[int, str], _ArtifactSource]]:
    """
    Unwrap any `Artifact` instances passed as positional or keyword arguments into their
    underlying python values. Returns the converted `(args, kwargs)` pair plus the artifact
    sources keyed by positional index or keyword name.
    """
    sources: Dict[Union[int, str], _ArtifactSource] = {}
    new_args = []
    for i, v in enumerate(args):
        unwrapped, source = await _unwrap_artifact_value(v)
        new_args.append(unwrapped)
        if source is not None:
            sources[i] = source
    new_kwargs = {}
    for k, v in kwargs.items():
        unwrapped, source = await _unwrap_artifact_value(v)
        new_kwargs[k] = unwrapped
        if source is not None:
            sources[k] = source
    return tuple(new_args), new_kwargs, sources


def _wrap_inline_run(outputs: Tuple[Any, ...] | Any, url: str) -> Run:
    """Wrap natively-computed task outputs in a `Run` so every execution mode returns one.

    Local and hybrid modes execute the task in-process and end up holding the task's
    native outputs rather than a platform run handle. This wrapper presents those
    outputs through the same `Run` interface remote mode returns: `wait()` is an
    immediate no-op (the work already happened) and `outputs()` serves the captured
    values.
    """
    from flyteidl2.common import identifier_pb2
    from flyteidl2.task import common_pb2
    from flyteidl2.workflow import run_definition_pb2

    from flyte.remote import ActionOutputs, Run

    class _InlineRun(Run):
        def __init__(self, outputs: Tuple[Any, ...] | Any):
            self._outputs = ActionOutputs(common_pb2.Outputs(), outputs if isinstance(outputs, tuple) else (outputs,))
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
            return url

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

    return _InlineRun(outputs)


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
    and shipped the resolved URIs here (`TaskContext.compiled_image_cache`). A nested
    `flyte.run(...)` submitted from task code seeds image resolution with it so
    already-built environments are never re-resolved in-cluster — where the predicted URI
    can differ from where the builder actually pushed (e.g. the remote builder's system
    registry), and where no builder may be available at all. Same-run child calls already
    reuse this cache via the controller; this extends that behavior to nested runs.
    Returns None on the driver (no task context), leaving behavior unchanged there.
    """
    tctx = internal_ctx().data.task_context
    return tctx.compiled_image_cache if tctx else None


def _uri_inputs_hash(inputs_uri: str) -> str:
    """Deterministic stand-in for `OffloadedInputData.inputs_hash` when the inputs blob
    cannot be read client-side (rerun of a source run whose outputs were cleaned up).

    The server computes the real hash as FNV-64a over the marshaled inputs and requires the
    field to be non-empty (it feeds cache-key computation). Hashing the source inputs URI in
    the same format is conservative: identical source location -> identical inputs -> same
    key, while an accidental match with a content-derived hash is a ~2^-64 event — the same
    collision odds the content hash itself carries. Worst case is a lost cache hit, never a
    wrong one beyond those odds.
    """
    import base64

    h = 0xCBF29CE484222325
    for b in inputs_uri.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return base64.urlsafe_b64encode(h.to_bytes(8, "big")).decode().rstrip("=")


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
        tracked: bool = False,
        tracked_strict: bool = False,
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
        self._interactive_mode = interactive_mode if interactive_mode is not None else ipython_check()
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
        # Report tracked run state to the control plane (TrackedRunService). Local-only; also
        # enabled via the `local.tracked` config key / flyte.init(local_tracked=...).
        self._tracked = tracked
        # Strict reporting (debugging): any reporting failure fails the run loudly instead of
        # being swallowed. Also enabled via the `local.tracked_strict` config key.
        self._tracked_strict = tracked_strict

    def _resolve_spawn_parent(self) -> Any | None:
        """Resolve the implicit *spawn* provenance parent (`Relation.related_to`, `SPAWN`).

        When a fresh run is created from inside a running remote task container
        (`TaskContext.is_in_cluster()`), the invoking run is the parent that spawned it. The
        pointer is stamped only when the invoking run's scope equals the new run's target scope
        exactly and all four id fields are non-empty (the server requires min_len=1 on each, and
        `Relation.related_to` is same-org/project/domain as the new run by contract). Returns
        None otherwise — provenance must never fail run creation. Pure resolution, no I/O.
        """
        from flyteidl2.common import identifier_pb2

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
            logger.debug(f"Skipping spawn relation: source scope {(org, project, domain)} != target {target}")
            return None
        if not (org and project and domain and name):
            logger.debug("Skipping spawn relation: incomplete source run identifier")
            return None
        return identifier_pb2.RunIdentifier(org=org, project=project, domain=domain, name=name)

    async def _build_task_spec_from_template(self, obj: TaskTemplate[P, R, F]) -> Tuple[Any, Any, str]:
        """Build `(task_spec, code_bundle, version)` from a local `TaskTemplate`.

        Used by `_run_remote` (local-task branch) for copy_files / dry_run / interactive_mode /
        include-files fidelity. Heavy
        imports stay function-local to keep `import flyte` cheap. The built `image_cache` is
        folded into the returned `task_spec` via the serialization context, so it is not returned.
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

        User-supplied `env_vars` plus the always-injected LOG_* / debug / rust-controller /
        sys-path keys. Shared by the fresh-build and inherited (rerun) RunSpec paths so debug's
        ssh-env injection and the log settings apply identically. Returns a fresh dict (never
        mutates `self._env_vars`).
        """
        cfg = get_init_config()
        env: Dict[str, str] = dict(self._env_vars or {})
        if env.get("LOG_LEVEL") is None:
            env["LOG_LEVEL"] = str(self._log_level) if self._log_level else str(logger.getEffectiveLevel())
        env["LOG_FORMAT"] = self._log_format
        if env.get("USER_LOG_LEVEL") is None:
            env["USER_LOG_LEVEL"] = str(self._user_log_level or user_logger.getEffectiveLevel())
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
        self,
        base: Any,
        *,
        task: Any = None,
        relation: Tuple[Any, str] | None = None,
        force_rerun_actions: Sequence[str] | None = None,
    ) -> Any:
        """Build the `RunSpec` for `create_run`.

        `base is None` -> a fresh spec from runner config (the run / recover path).
        `base` set     -> deep-copy a prior run's `RunSpec` and merge runner overrides by key
        (the rerun path: env merge + explicitly-set field overrides). Pure proto assembly, no I/O.
        This is the single place runner config maps onto a `RunSpec`. `relation` is the provenance
        link to record on `RunSpec.relation`: `(parent RunIdentifier, "rerun" | "recover" | "spawn")`,
        or None. The identifier must be fully qualified (org/project/domain/name) — the server rejects
        partial ones. `force_rerun_actions` (recover only) lands on `RunSpec.recover`.
        """
        from flyteidl2.core import literals_pb2, security_pb2
        from flyteidl2.task import run_pb2
        from google.protobuf import wrappers_pb2

        # google.protobuf ships no type stubs for the dynamically generated wrappers_pb2 module.
        _bool_value_cls = cast(Any, wrappers_pb2).BoolValue

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
                interruptible=_bool_value_cls(value=self._interruptible) if self._interruptible is not None else None,
                annotations=run_pb2.Annotations(values=self._annotations),
                labels=run_pb2.Labels(values=self._labels),
                envs=env_kv,
                queue=self._queue or (task.queue if task is not None else ""),
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
            # Provenance is per-run, never inherited: a rerun of a rerun must point at its immediate
            # parent (set below from `relation`), not the grandparent captured in the prior spec.
            for provenance_field in ("relation", "related_to", "recover"):
                # DESCRIPTOR internals are opaque to checkers; guard for fields absent from the current pin.
                if provenance_field in cast(Any, run_pb2.RunSpec).DESCRIPTOR.fields_by_name:
                    run_spec.ClearField(provenance_field)
            run_spec.envs.CopyFrom(env_kv)
            if self._interruptible is not None:
                run_spec.interruptible.CopyFrom(_bool_value_cls(value=self._interruptible))
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
                run_spec.queue = self._queue
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

        # relation: gated until the flyteidl2 pin includes RunSpec.relation. Recover semantics depend
        # on the field, so recover fails loudly without it; rerun/spawn provenance is best-effort.
        if relation:
            ref, kind = relation
            # google.protobuf ships no stubs for descriptor internals; DESCRIPTOR is opaque to checkers.
            if "relation" not in cast(Any, run_pb2.RunSpec).DESCRIPTOR.fields_by_name:
                if kind == "recover":
                    raise NotImplementedError(
                        "recover is not yet supported by this backend "
                        "(RunSpec.relation is unavailable in this flyteidl2 build)."
                    )
            else:
                from flyteidl2.common import run_pb2 as common_run_pb2

                # Relation / RELATION_TYPE_* / RunSpec.relation are absent from current flyteidl2 stubs
                # (runtime-gated by the DESCRIPTOR check above).
                _relation_pb = cast(Any, common_run_pb2)
                relation_type = {
                    "rerun": _relation_pb.RELATION_TYPE_RERUN,
                    "recover": _relation_pb.RELATION_TYPE_RECOVER,
                    # SPAWN ships in a later flyteidl2 than RERUN/RECOVER; drop the pointer on
                    # older builds rather than fail run creation.
                    "spawn": getattr(common_run_pb2, "RELATION_TYPE_SPAWN", None),
                }.get(kind)
                if relation_type is not None:
                    cast(Any, run_spec).relation.CopyFrom(
                        _relation_pb.Relation(related_to=ref, relation_type=relation_type)
                    )
                if kind == "recover" and force_rerun_actions:
                    # Escape hatch: these actions re-execute even though they succeeded in the
                    # source run. A listed parent re-enqueues its children (list them too to force
                    # the whole subtree); unknown names are ignored server-side.
                    cast(Any, run_spec).recover.CopyFrom(
                        cast(Any, run_pb2).Recover(force_rerun_actions=list(force_rerun_actions))
                    )

        return run_spec

    async def _submit_remote(
        self,
        *,
        task_spec: Any,
        task_id: Any,
        proto_inputs: Any,
        run_spec: Any,
        run_id: Any,
        project_id: Any,
        offloaded_input_data: Any = None,
        trigger_name: Any = None,
    ) -> Run:
        """Upload inputs and create the run. The single network call site for remote submission.

        Consumes an already-built `run_spec` (see `_apply_overrides`), raw proto `inputs`
        (`flyteidl2.task.Inputs`), and a task by reference (`task_id`) or by value
        (`task_spec`); shared by `_run_remote`, `_run_trigger` and `rerun`. `offloaded_input_data`
        (`flyteidl2.common.OffloadedInputData`) references already-offloaded inputs (e.g. the
        source run's inputs.pb on a rerun whose inputs can't be re-downloaded) and skips the
        upload; exactly one of `proto_inputs` / `offloaded_input_data` is used. `trigger_name`
        (`flyteidl2.common.TriggerName`) fires the run *as* that trigger: it replaces `task_id`
        on the create-run request (the server resolves the trigger's pinned task itself and
        records the trigger as the run's origin); `task_id` is still used to upload the inputs.
        """
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from flyteidl2.dataproxy import dataproxy_service_pb2
        from flyteidl2.workflow import run_service_pb2

        import flyte.errors
        from flyte._internal.runtime.convert import generate_content_inputs_hash
        from flyte.remote import Run

        try:
            if offloaded_input_data is None:
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
                offloaded_input_data = upload_resp.offloaded_input_data

                # The hash the server derives from the marshaled inputs folds in the offloaded
                # blob URI and ignores `Literal.hash`, so content-based caching silently degrades
                # to URI-based caching at the run entrypoint: identical content uploaded to a
                # fresh URI misses. Sub-actions don't have this problem — the controller hashes
                # the same inputs through `generate_inputs_repr_for_literal`, which substitutes
                # the content hash. Recompute over that representation so the root action agrees.
                # Returns None (leaving the server's value alone) unless a hashed input survives
                # cache-ignore filtering, so cache keys for everyone else are untouched.
                #
                # `task_spec` is populated on every path into here, including the by-reference
                # one where `task_id` is also set (`task_spec = task.pb2.spec` on a fetched
                # task), so the ignore list is always the registered task's own.
                md = task_spec.task_template.metadata if task_spec is not None else None
                content_hash = generate_content_inputs_hash(
                    proto_inputs, list(md.cache_ignore_input_vars) if md else []
                )
                if content_hash is not None:
                    offloaded_input_data.inputs_hash = content_hash

            create_req = run_service_pb2.CreateRunRequest(
                run_id=run_id,
                project_id=project_id,
                offloaded_input_data=offloaded_input_data,
                run_spec=run_spec,
            )
            # `task` is a oneof: fire as a trigger, else reference an already-registered task by
            # id, else send the full spec.
            if trigger_name is not None:
                create_req.trigger_name.CopyFrom(trigger_name)
            elif task_id is not None:
                create_req.task_id.CopyFrom(task_id)
            else:
                create_req.task_spec.CopyFrom(task_spec)

            with track_operation("create_run"):
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

        from ._internal.runtime.convert import convert_from_native_to_inputs_binding_artifacts

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        # A `flyte.remote.Artifact` argument binds to the literal the artifact service already
        # stored for it, rather than being materialized to python and re-serialized. That literal
        # already carries `Literal.artifact_id`, so provenance on the run's inputs is the service's
        # assertion rather than one this process re-stamps. The declared input type is checked
        # against the artifact's stored type first -- see `bind_artifact_literals`.
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
            inputs = await convert_from_native_to_inputs_binding_artifacts(
                task.interface, args, kwargs, custom_context=self._custom_context
            )
            version = task.pb2.task_id.version
            code_bundle = None
        elif isinstance(obj, TaskTemplate):
            task = cast(TaskTemplate[P, R, F], obj)
            task_spec, code_bundle, version = await self._build_task_spec_from_template(obj)
            inputs = await convert_from_native_to_inputs_binding_artifacts(
                obj.native_interface, args, kwargs, custom_context=self._custom_context
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

            # Provenance for a fresh run: when launched from inside a running remote task,
            # record a spawn link to the invoking run. (Recovery is a rerun() concern.)
            relation: Tuple[Any, str] | None
            spawn_parent = self._resolve_spawn_parent()
            relation = (spawn_parent, "spawn") if spawn_parent is not None else None
            run_spec = self._apply_overrides(None, task=task, relation=relation)
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

    @requires_initialization
    async def _run_trigger(self, trigger: RemoteTrigger | TriggerDetails, *args: Any, **kwargs: Any) -> Run:
        """Fire a deployed trigger on demand, returning the new `Run`.

        The trigger is a saved launch configuration: its registered inputs and `RunSpec` (env
        vars, queue, notifications, ...) are the floor. Keyword inputs override individual
        trigger inputs; `with_runcontext(...)` overrides layer on top of the trigger's run spec.
        The run is created *as* the trigger, so the platform records it as trigger-fired, exactly
        like a scheduled fire. Inputs are resolved client-side (the server only restores a
        trigger's inputs when it fires with nothing else set), so an inline-registered trigger
        launches with its inputs too, not just an offloaded one.
        """
        from flyteidl2.core import literals_pb2
        from flyteidl2.task import common_pb2 as task_common_pb2

        import flyte.errors
        from flyte.remote._task import Task
        from flyte.remote._trigger import Trigger as RemoteTrigger
        from flyte.remote._trigger import TriggerDetails

        from ._internal.runtime.convert import KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, convert_from_native_to_inputs

        if args:
            raise ValueError(
                "Trigger inputs can only be overridden by keyword (flyte.run(trigger, x=1)): every input left "
                "out keeps the value the trigger was deployed with, so positional arguments are ambiguous."
            )
        if self._dry_run:
            raise ValueError("dry_run is not supported when running a trigger.")
        if get_client() is None:
            raise flyte.errors.InitializationError(
                "ClientNotInitializedError",
                "user",
                "flyte.run requires client to be initialized. "
                "Call flyte.init() with a valid endpoint/api-key before using this function"
                "or Call flyte.init_from_config() with a valid path to the config file",
            )

        details: TriggerDetails
        if isinstance(trigger, RemoteTrigger):
            # A listed trigger carries no spec; fetch the details once.
            details = trigger.details or await TriggerDetails.get.aio(name=trigger.name, task_name=trigger.task_name)
        else:
            details = trigger
        trigger_name = details.pb2.id.name
        spec = details.pb2.spec

        # The trigger is pinned to one task version; fetch it for the interface (keyword conversion)
        # and the cache metadata the upload path consults.
        task = await Task.get(
            name=trigger_name.task_name,
            project=trigger_name.project,
            domain=trigger_name.domain,
            version=spec.task_version,
        ).fetch.aio()

        # Inputs. Three cases: inline literals (backend without input offloading), offloaded
        # literals (read back only when something has to be merged into them), or none.
        proto_inputs: task_common_pb2.Inputs | None = None
        offloaded_input_data = None
        base_inputs: task_common_pb2.Inputs | None = None
        match spec.WhichOneof("input_wrapper"):
            case "inputs":
                base_inputs = spec.inputs
            case "offloaded_input_data":
                if kwargs:
                    from ._internal.runtime.io import load_inputs

                    base_inputs = (await load_inputs(spec.offloaded_input_data.uri)).proto_inputs
                else:
                    offloaded_input_data = spec.offloaded_input_data
            case _:
                base_inputs = task_common_pb2.Inputs()

        if kwargs:
            from flyte.models import NativeInterface

            iface = task.interface
            unknown = sorted(set(kwargs) - set(iface.inputs))
            if unknown:
                known = ", ".join(iface.inputs) or "<none>"
                raise ValueError(
                    f"Unknown input(s) {unknown} for trigger {trigger_name.name!r} on task "
                    f"{trigger_name.task_name!r}. Known inputs: {known}."
                )
            # Only the overridden inputs are converted; the rest keep the trigger's literals.
            reduced_iface = NativeInterface(
                inputs={k: v for k, v in iface.inputs.items() if k in kwargs},
                outputs={},
                _remote_defaults=iface._remote_defaults,
            )
            converted = await convert_from_native_to_inputs(
                reduced_iface, custom_context=self._custom_context, **kwargs
            )
            assert base_inputs is not None
            overrides = {lit.name: lit.value for lit in converted.proto_inputs.literals}
            merged = [
                task_common_pb2.NamedLiteral(name=lit.name, value=overrides.pop(lit.name, lit.value))
                for lit in base_inputs.literals
            ]
            # An input the trigger never bound (left to the task default) still has to land.
            merged += [task_common_pb2.NamedLiteral(name=name, value=v) for name, v in overrides.items()]
            # Context: the trigger's registered context (custom_context, and the kickoff-time input
            # marker on scheduled triggers) is the floor; this call's custom_context wins per key.
            context = {kv.key: kv.value for kv in base_inputs.context}
            context.update({kv.key: kv.value for kv in converted.proto_inputs.context})
            # The runtime fills the kickoff-time input from run_start_time whenever the marker is
            # present. An explicit value for that input must win, so drop the marker.
            if context.get(KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY) in kwargs:
                del context[KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY]
            proto_inputs = task_common_pb2.Inputs(
                literals=merged,
                context=[literals_pb2.KeyValuePair(key=k, value=v) for k, v in context.items()],
            )
        elif base_inputs is not None:
            proto_inputs = task_common_pb2.Inputs()
            proto_inputs.CopyFrom(base_inputs)
            if self._custom_context:
                context = {kv.key: kv.value for kv in proto_inputs.context}
                context.update(self._custom_context)
                del proto_inputs.context[:]
                proto_inputs.context.extend(literals_pb2.KeyValuePair(key=k, value=v) for k, v in context.items())

        # Run spec: the trigger's registered spec is the floor (that is what a scheduled fire
        # uses); runner overrides merge on top, provenance is per-run.
        spawn_parent = self._resolve_spawn_parent()
        relation = (spawn_parent, "spawn") if spawn_parent is not None else None
        run_spec = self._apply_overrides(spec.run_spec, relation=relation)

        run_id, project_id = self._resolve_run_target(trigger_name.project, trigger_name.domain, trigger_name.org)
        return await self._submit_remote(
            task_spec=task.pb2.spec,
            task_id=task.pb2.task_id,
            proto_inputs=proto_inputs,
            offloaded_input_data=offloaded_input_data,
            run_spec=run_spec,
            run_id=run_id,
            project_id=project_id,
            trigger_name=trigger_name,
        )

    @requires_storage
    @requires_initialization
    async def _run_hybrid(self, obj: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Run:
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
        return _wrap_inline_run(outputs, url=output_path)

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
        from flyte.notify._notifiers import Notification as _Notification
        from flyte.notify._sender import send_notifications

        notifications = self._notifications
        if isinstance(notifications, _NamedRule):
            logger.info("Skipping named rule %r in local mode", notifications.name)
            return

        await send_notifications(
            cast(Union[_Notification, Tuple[_Notification, ...]], notifications),
            phase=phase,
            task_name=task_name,
            run_name=run_name,
            error=error,
            project=self._project or "",
            domain=self._domain or "",
        )

    def _resolve_tracked_report_scope(self) -> Tuple[str | None, str, str] | None:
        """Resolve (org, project, domain) for tracked-run reporting, or None when reporting
        should be skipped (with a single warning). Raises with a clear message when
        reporting is requested but project/domain are not configured, or — in strict
        mode — when no client is initialized."""
        import flyte.errors
        from flyte._initialize import is_local_tracked_enabled, is_local_tracked_strict

        if not (self._tracked or is_local_tracked_enabled()):
            # An explicit strict request without reporting is a caller error; a config-only
            # `local.tracked_strict` with reporting disabled is simply inert.
            if self._tracked_strict:
                raise ValueError(
                    "Strict tracked-run reporting (tracked_strict) requires reporting to be enabled: "
                    "pass tracked=True / --tracked or set local.tracked in your config."
                )
            return None

        init_config = _get_init_config()
        if init_config is None or init_config.client is None:
            if self._tracked_strict or is_local_tracked_strict():
                raise flyte.errors.InitializationError(
                    "ClientNotInitializedError",
                    "user",
                    "Strict tracked-run reporting requires an initialized client. Call flyte.init() "
                    "with a valid endpoint/api-key or flyte.init_from_config().",
                )
            logger.warning(
                "Tracked-run reporting was requested but no Flyte client is initialized; "
                "running without reporting. Call flyte.init() with a valid endpoint/api-key "
                "or flyte.init_from_config() to enable reporting."
            )
            return None

        project = self._project or init_config.project
        domain = self._domain or init_config.domain
        if not project or not domain:
            raise flyte.errors.InitializationError(
                "ProjectDomainNotConfigured",
                "user",
                "Tracked-run reporting requires a project and domain. Set them in the 'task' section "
                "of your config file, pass them to flyte.init(project=..., domain=...), or use "
                "flyte run --project/--domain.",
            )
        return init_config.org, project, domain

    async def _run_local(self, obj: TaskTemplate[P, R, F], *args: P.args, **kwargs: P.kwargs) -> Run:

        from flyte._internal.controllers import create_controller
        from flyte._internal.controllers._local_controller import LocalController
        from flyte.report import Report

        controller = cast(LocalController, create_controller("local"))

        report_scope = self._resolve_tracked_report_scope()
        if report_scope is not None:
            from flyte._persistence._remote_reporter import generate_tracked_run_name, validate_tracked_run_name

            org, project, domain = report_scope
            if self._name is not None:
                validate_tracked_run_name(self._name)
                run_name = self._name
            else:
                run_name = generate_tracked_run_name()
            action = ActionID(name=run_name, project=project, domain=domain, org=org)
        elif self._name is None:
            action = ActionID.create_random()
        else:
            action = ActionID(name=self._name)

        if self._metadata_path is None:
            metadata_path = pathlib.Path("/") / "tmp" / "flyte" / "metadata" / action.name
        else:
            metadata_path = pathlib.Path(self._metadata_path) / action.name
        output_path = metadata_path / "a0"
        if self._raw_data_path is None:
            path = pathlib.Path("/") / "tmp" / "flyte" / "raw_data" / action.name
            raw_data_path = RawDataPath(path=str(path))
        else:
            raw_data_path = RawDataPath(path=self._raw_data_path)

        from flyte.storage import join as storage_join

        ctx = internal_ctx()
        rd_base = raw_data_path.path
        run_start_time = self._run_start_time or datetime.now(timezone.utc)
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
            run_start_time=run_start_time,
        )

        if self._tracker is not None:
            ctx = Context(ctx.data.replace(tracker=self._tracker))

        from flyte._initialize import is_persistence_enabled
        from flyte._persistence._recorder import RunRecorder

        persist = is_persistence_enabled()
        run_name = action.run_name or action.name

        if persist:
            RunRecorder.initialize_persistence()

        reporter = None
        run_url = str(metadata_path)
        if report_scope is not None:
            from flyte._initialize import is_local_tracked_strict
            from flyte._persistence._remote_reporter import start_tracked_run_reporting

            org, project, domain = report_scope
            init_config = get_init_config()
            reporter = await start_tracked_run_reporting(
                client=get_client(),
                task=obj,
                run_name=run_name,
                org=org,
                project=project,
                domain=domain,
                run_spec=self._apply_overrides(None, task=obj),
                labels=self._labels,
                run_start_time=run_start_time,
                args=args,
                kwargs=kwargs,
                root_dir=init_config.root_dir,
                strict=self._tracked_strict or is_local_tracked_strict(),
            )
            if reporter is not None:
                run_url = get_client().console.tracked_run_url(project=project, domain=domain, run_name=run_name)
                logger.info(f"Reporting tracked run to the control plane: {run_url}")

        recorder = RunRecorder(tracker=self._tracker, persist=persist, run_name=run_name, reporter=reporter)
        controller.set_recorder(recorder)

        recorder.record_root_start(task_name=obj.name)

        new_args, new_kwargs, _ = await _unwrap_artifacts(args, kwargs)

        # When reporting is active, catch SIGTERM for the duration of the run so an
        # external termination reports ABORTED like Ctrl+C does. SIGINT is left to the
        # interpreter's KeyboardInterrupt / asyncio cancellation flow.
        interrupt_signal: List[str] = []
        sigterm_installed = False
        prev_sigterm: Any = None
        if reporter is not None:
            import signal
            import threading

            def _on_sigterm(signum: int, frame: Any) -> None:
                interrupt_signal.append("SIGTERM")
                raise KeyboardInterrupt

            if threading.current_thread() is threading.main_thread():
                try:
                    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
                    sigterm_installed = True
                except (ValueError, OSError):  # non-main interpreter contexts
                    sigterm_installed = False

        try:
            with ctx.replace_task_context(tctx):
                # make the local version always runs on a different thread, returns a wrapped future.
                if obj._call_as_synchronous:
                    fut = controller.submit_sync(obj, *new_args, **new_kwargs)
                    awaitable = asyncio.wrap_future(fut)
                    outputs = await awaitable
                else:
                    outputs = await controller.submit(obj, *new_args, **new_kwargs)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Interrupted (Ctrl+C / SIGTERM / cancellation): report every in-flight
            # action — the root included — as ABORTED with a short, bounded flush,
            # then re-raise so conventional signal semantics are preserved.
            if reporter is not None:
                signal_name = interrupt_signal[0] if interrupt_signal else "SIGINT"
                reporter.abort_all(reason=f"aborted by user ({signal_name})")
                try:
                    # Blocking is fine here — the process is exiting. A reporting
                    # failure (even strict) must never replace the interrupt.
                    reporter.close(timeout=5.0)
                except Exception as flush_err:
                    logger.warning(f"Tracked-run abort reporting incomplete: {flush_err}")
            raise
        except Exception as e:
            recorder.record_root_failure(error=str(e))
            if reporter is not None:
                # Bounded flush so the terminal state lands before the process exits.
                # Even in strict mode, a reporting failure must never mask the task's
                # own error — log it instead of raising over `e`.
                try:
                    await reporter.aclose()
                except Exception as flush_err:
                    logger.warning(f"Tracked-run reporting failed during shutdown: {flush_err}")
            if self._notifications:
                await self._send_local_notifications(
                    phase=ActionPhase.FAILED, task_name=obj.name, run_name=run_name, error=str(e)
                )
            raise
        else:
            try:
                recorder.record_root_complete()
            finally:
                # Bounded flush barrier; in strict mode this re-raises the first
                # captured reporting failure so the run exits loudly.
                if reporter is not None:
                    await reporter.aclose()
            if self._notifications:
                await self._send_local_notifications(phase=ActionPhase.SUCCEEDED, task_name=obj.name, run_name=run_name)
        finally:
            if sigterm_installed:
                import signal

                try:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass

        return _wrap_inline_run(outputs, url=run_url)

    @syncify  # type: ignore[arg-type]
    async def run(
        self,
        task: TaskTemplate[P, R, F] | LazyEntity | RemoteTrigger | TriggerDetails,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Run:
        """
        Run an async `@env.task` or `TaskTemplate` instance. The existing async context will be used.

        A deployed trigger (`flyte.remote.Trigger.get(...)`) can be run too: the run is fired *as*
        the trigger, with the trigger's registered inputs, env vars, queue and notifications.
        Keyword arguments override individual trigger inputs. Remote mode only.

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

        Args:
            task: TaskTemplate instance `@env.task` or `TaskTemplate`, a fetched remote task, or a
                deployed trigger (`flyte.remote.Trigger` / `TriggerDetails`)
            args: Arguments to pass to the Task (not allowed for a trigger)
            kwargs: Keyword arguments to pass to the Task (for a trigger: overrides of its inputs)

        Returns:
            A Run handle in every mode. Remote mode returns the platform run; local and
            hybrid modes return an in-process wrapper whose `outputs()` serves the task's
            native results and whose `wait()` is an immediate no-op.
        """
        from flyte.remote._task import LazyEntity, TaskDetails
        from flyte.remote._trigger import Trigger as RemoteTrigger
        from flyte.remote._trigger import TriggerDetails

        if isinstance(task, (LazyEntity, TaskDetails, RemoteTrigger, TriggerDetails)) and self._mode != "remote":
            raise ValueError("Remote tasks and triggers can only be run in remote mode.")

        if not isinstance(task, (TaskTemplate, LazyEntity, TaskDetails, RemoteTrigger, TriggerDetails)):
            raise TypeError(f"On Flyte tasks can be run, not generic functions or methods '{type(task)}'.")

        # report mirrors a locally-orchestrated run onto the control plane as a tracked run —
        # local-only. Fail fast rather than silently ignoring it in remote/hybrid mode
        # (remote runs are already reported).
        if self._tracked and self._mode != "local":
            raise ValueError("report is only supported in local mode (use --tracked)")

        # Set the run mode in the context variable so that offloaded types (files, directories, dataframes)
        # can check the mode for controlling auto-uploading behavior (only enabled in remote mode).
        _run_mode_var.set(self._mode)

        try:
            if self._mode == "remote":
                if isinstance(task, (RemoteTrigger, TriggerDetails)):
                    return await self._run_trigger(task, *args, **kwargs)
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
        recover: bool = False,
        force_rerun_actions: Sequence[str] | None = None,
        allow_missing_source_outputs: bool = False,
        **inputs: Any,
    ) -> Run:
        """Re-run a prior run, returning a new `Run`.

        - `rerun("r1")` creates a whole new run with the prior run's exact inputs, fetching its
          task spec from the platform (no local code needed). Everything re-executes, subject to
          global caching.
        - `rerun("r1", recover=True)` creates a whole new run with the same inputs, but reuses the
          prior run's succeeded actions and re-executes only what failed or never ran.
        - `rerun("r1", x=2)` changes input parameters (converted against the fetched task
          interface); every input left out keeps the source run's value. Task inputs share the
          keyword namespace with the arguments above, so a task input named `run_name`,
          `action_name`, `recover`, `force_rerun_actions` or `allow_missing_source_outputs` is not
          reachable this way.
        - `rerun("r1", recover=True, x=2)` combines the two: the new run starts from the changed
          inputs while still reusing the source run's succeeded actions. Recovered actions keep
          the outputs they produced under the *original* inputs — name them in
          `force_rerun_actions` to re-execute them against the new inputs.

        The prior run's code is always replayed as-is: this never substitutes local code. Replaying
        a run with new code is fork, reserved for flyteplugins-union.

        The prior run's `RunSpec` is inherited and merged with this context's overrides
        (`with_runcontext(env_vars=..., interruptible=...)` etc.). Provenance is recorded on
        `RunSpec.relation` — RERUN pointing at `run_name`, or RECOVER when `recover=True` (when the
        flyteidl2 build supports it). Currently remote-only.

        Args:
            run_name: Name of the prior run to re-run.
            action_name: Action within the prior run to source the task + inputs from. Defaults to
                `a0`, the root action — i.e. the whole run. Naming a child action instead roots the
                new run at that action's task, run with the exact inputs it received. Cannot be
                combined with `recover`.
            recover: Reuse the prior run's succeeded actions, re-running only what failed or never
                ran, instead of re-executing everything. Requires a backend (and flyteidl2 build)
                with RunSpec.relation recovery support; raises NotImplementedError at submit
                otherwise.
            force_rerun_actions: With `recover`, names of actions that must re-execute even though
                they succeeded in the source run (escape hatch). A listed parent action re-enqueues
                its children — list them too to force the whole subtree; a listed condition re-pauses
                for a new signal. Unknown names are ignored.
            allow_missing_source_outputs: Proceed when the source run's outputs were cleaned up
                from storage, using its inputs URI directly. The client cannot verify the inputs
                still exist — if they were deleted too, the new run fails at runtime. Irrelevant
                when the new inputs cover every input of the task, since the source inputs are
                then not read at all.
            inputs: Optional native keyword inputs to change parameters. Any input not passed
                keeps the source run's value, so passing none reuses the source run's inputs
                wholesale.

        Returns:
            the new Run.
        """
        if self._mode != "remote":
            raise NotImplementedError(f"rerun is only supported in remote mode, got mode={self._mode!r}")
        if force_rerun_actions and not recover:
            raise ValueError("force_rerun_actions requires recover=True")
        # Recovery still replays the source run's code as-is: substituting code is `flyte fork`,
        # reserved for flyteplugins-union.
        # Recovery matches succeeded actions from the source run by deterministic name; a run
        # rooted at a sub-action has a different action tree, so the reuse set would not line up.
        if recover and action_name != "a0":
            raise ValueError(
                f"recover=True cannot be combined with action_name={action_name!r}: recovery "
                f"matches succeeded actions from the source run by name, and a run rooted at a "
                f"single action has a different action tree. Re-run the action on its own "
                f"(recover=False), or recover the whole run."
            )
        if recover and inputs:
            # Recovery reuses succeeded actions by name, and those actions ran under the source
            # run's inputs. Changing the root inputs is allowed, but the reused outputs are stale
            # with respect to them unless the actions are forced to re-execute.
            logger.warning(
                f"Recovering {run_name} with changed inputs {sorted(inputs)}: the new run "
                f"starts from the changed inputs, but every action recovered from {run_name} "
                f"keeps the output it produced under the original inputs. Pass "
                f"force_rerun_actions=[...] for the actions that must re-execute against the "
                f"new inputs."
            )

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

        # Task source: always the prior action's spec — rerun never substitutes local code.
        if not action_details.pb2.HasField("task"):
            raise ValueError(f"Action {run_name}/{action_name} has no task spec to rerun.")
        task_spec = action_details.pb2.task

        # Inputs: reuse the prior run's raw proto inputs, with any new native inputs overlaid on
        # top of them. New inputs that cover the whole interface stand on their own, so the source
        # inputs are not fetched at all (the escape hatch when they are gone from storage).
        proto_inputs = None
        offloaded_input_data = None
        reduced_iface = None
        if inputs:
            from flyte.models import NativeInterface
            from flyte.types._interface import guess_interface

            iface = guess_interface(task_spec.task_template.interface)
            unknown = sorted(set(inputs) - set(iface.inputs))
            if unknown:
                known = ", ".join(iface.inputs) or "<none>"
                raise ValueError(f"Unknown input(s) {unknown} for {run_name}/{action_name}. Known inputs: {known}.")
            # Only the changed inputs are converted; the rest keep the source run's literals. Built
            # in interface order so the resulting Inputs keep the task's declared ordering.
            reduced_iface = NativeInterface(
                inputs={k: v for k, v in iface.inputs.items() if k in inputs},
                outputs={},
                _remote_defaults=iface._remote_defaults,
            )
            changes_every_input = len(reduced_iface.inputs) == len(iface.inputs)
        else:
            changes_every_input = False

        if not changes_every_input:
            # Rerun/recover only need the source run's INPUTS. GetActionData resolves inputs AND
            # outputs server-side concurrently and 404s wholesale when either blob has been
            # cleaned up (retention) — and which half the error names is a race. The client has
            # no RPC to check the inputs blob alone, so a missing-data 404 is a hard error by
            # default; `allow_missing_source_outputs` opts into proceeding with the inputs URI
            # (fails at runtime if the inputs turn out to be gone too).
            from connectrpc.code import Code
            from connectrpc.errors import ConnectError
            from flyteidl2.common import run_pb2 as common_run_pb2
            from flyteidl2.workflow import run_service_pb2

            import flyte.errors

            try:
                resp = await get_client().dataproxy_service.get_action_data(
                    request=dataproxy_service_pb2.GetActionDataRequest(action_id=action_details.pb2.id)
                )
                proto_inputs = resp.inputs
            except ConnectError as e:
                if e.code != Code.NOT_FOUND:
                    raise
                if "inputs" in str(e.message):
                    # The inputs blob itself is gone — nothing to feed the new run; fail
                    # fast with a clear story instead of the server's raw 404.
                    raise flyte.errors.RuntimeUserError(
                        "SourceRunInputsUnavailableError",
                        f"Source run {run_name}'s inputs are no longer in storage (deleted by "
                        f"retention/cleanup), so it cannot be rerun or recovered with its "
                        f"original inputs. Pass every input explicitly instead: "
                        f"flyte.with_runcontext(...).rerun('{run_name}', x=..., y=...), or "
                        f"launch fresh local code with `flyte run ...` "
                        f"(inputs come from the CLI parameters).",
                    ) from e
                if not allow_missing_source_outputs:
                    raise flyte.errors.RuntimeUserError(
                        "SourceRunOutputsUnavailableError",
                        f"Source run {run_name}'s outputs are no longer in storage. Rerun/recover "
                        f"only needs its inputs, but whether those still exist cannot be verified "
                        f"from the client. If you know the inputs are intact, retry with "
                        f"--allow-missing-outputs "
                        f"(rerun(..., allow_missing_source_outputs=True)); if they were "
                        f"deleted too, the new run would fail at runtime — pass every input "
                        f"explicitly instead (rerun('{run_name}', x=..., y=...) or "
                        f"`flyte run ...`).",
                    ) from e
                uris = await get_client().run_service.get_action_data_u_r_is(
                    run_service_pb2.GetActionDataURIsRequest(action_id=action_details.pb2.id)
                )
                if not uris.inputs_uri:
                    raise
                logger.warning(
                    f"Source run {run_name} outputs are no longer in storage; proceeding with its "
                    f"inputs at {uris.inputs_uri} (--allow-missing-outputs). If the inputs were "
                    f"deleted too the new run will fail at runtime, and recovered actions "
                    f"referencing deleted outputs will fail if consumed "
                    f"(use --force-rerun-action to re-execute them)."
                )
                offloaded_input_data = common_run_pb2.OffloadedInputData(
                    uri=uris.inputs_uri,
                    inputs_hash=_uri_inputs_hash(uris.inputs_uri),
                )

        if reduced_iface is not None:
            from flyteidl2.task import common_pb2 as task_common_pb2

            converted = await convert_from_native_to_inputs(
                reduced_iface, custom_context=self._custom_context, **inputs
            )
            if changes_every_input:
                proto_inputs = converted.proto_inputs
            elif proto_inputs is None:
                # Only the source inputs' URI is in hand (--allow-missing-outputs), so the
                # unchanged inputs cannot be read to merge the changed ones into.
                import flyte.errors

                raise flyte.errors.RuntimeUserError(
                    "SourceRunInputsUnavailableError",
                    f"Source run {run_name}'s inputs could not be read, so the changed inputs "
                    f"{sorted(inputs)} cannot be merged with the ones being kept. Pass every "
                    f"input of {run_name}/{action_name} explicitly instead.",
                )
            else:
                overrides = {lit.name: lit.value for lit in converted.proto_inputs.literals}
                merged = [
                    task_common_pb2.NamedLiteral(name=lit.name, value=overrides.pop(lit.name, lit.value))
                    for lit in proto_inputs.literals
                ]
                # An input the source run never carried (e.g. added since) still has to land.
                merged += [task_common_pb2.NamedLiteral(name=name, value=v) for name, v in overrides.items()]
                proto_inputs = task_common_pb2.Inputs(
                    literals=merged,
                    context=converted.proto_inputs.context or proto_inputs.context,
                )

        run_id, project_id = self._resolve_run_target(project, domain, cfg.org)

        # Every rerun records provenance to the run being rerun; recover upgrades it to RECOVER.
        # Relation identifiers must be fully qualified; the parent is scoped to the same
        # org/project/domain as the new run.
        from flyteidl2.common import identifier_pb2

        relation = (
            identifier_pb2.RunIdentifier(org=cfg.org, project=project, domain=domain, name=run_name),
            "recover" if recover else "rerun",
        )
        run_spec = self._apply_overrides(base_run_spec, relation=relation, force_rerun_actions=force_rerun_actions)
        return await self._submit_remote(
            task_spec=task_spec,
            task_id=None,
            proto_inputs=proto_inputs,
            run_spec=run_spec,
            run_id=run_id,
            project_id=project_id,
            offloaded_input_data=offloaded_input_data,
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
    tracked: bool = False,
    tracked_strict: bool = False,
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

    Args:
        mode: Optional The mode to use for the run, if not provided, it will be computed from flyte.init
        version: Optional The version to use for the run, if not provided, it will be computed from the code bundle
        name: Optional The name to use for the run
        service_account: Optional The service account to use for the run context
        copy_style: Optional The copy style to use for the run context
        dry_run: Optional If true, the run will not be executed, but the bundle will be created
        copy_bundle_to: When dry_run is True, the bundle will be copied to this location if specified
        interactive_mode: Optional, can be forced to True or False.
            If not provided, it will be set based on the current environment. For example Jupyter notebooks are
            considered
            interactive mode, while scripts are not. This is used to determine how the code bundle is created.
        raw_data_path: Use this path to store the raw data for the run for local and remote, and can be used to
            store raw data in specific locations.
        run_base_dir: Optional The base directory to use for the run. This is used to store the metadata for the run,
            that is passed between tasks.
        run_start_time: Optional UTC datetime at which the run was triggered. If not provided, defaults to
            `datetime.now(timezone.utc)` at TaskContext construction. Useful for local simulation/tests that need a
            deterministic timestamp. Accessible inside a task via `flyte.ctx().run_start_time`.
        overwrite_cache: Optional If true, the cache will be overwritten for the run
        project: Optional The project to use for the run
        domain: Optional The domain to use for the run
        env_vars: Optional Environment variables to set for the run
        labels: Optional user-defined labels to attach to the run as KEY=VALUE pairs, used for
            filtering and organizing runs (e.g. `flyte get run --with-label team=ml`)
        annotations: Optional Annotations to set for the run
        interruptible: Optional If true, the run can be scheduled on interruptible instances and false implies
            that all tasks in the run should only be scheduled on non-interruptible instances. If not specified the
            original setting on all tasks is retained.
        log_level: Optional Log level to set for the run. If not provided, it will be set to the default log level
            set using `flyte.init()`
        log_format: Optional Log format to set for the run. If not provided, it will be set to the default log format
        reset_root_logger: If True, replace the root logger's handlers with Flyte's own, so lines
            from third-party libraries that propagate to the root logger are formatted the same way
            as Flyte's (JSON when `log_format` is `json`, otherwise Rich or plain console). Defaults
            to False, which leaves those handlers in place and instead wraps each one so its output
            carries the run and action context. Can also be turned on with the environment variable
            `FLYTE_RESET_ROOT_LOGGER=1`.
        disable_run_cache: Optional If true, the run cache will be disabled. This is useful for testing purposes.
        queue: Optional The queue to use for the run. This is used to specify the cluster to use for the run.
        max_action_concurrency: Optional Maximum number of actions that can run concurrently within this run.
            Only applies to remote runs. If not provided, the platform default (configurable via the
            `run.max_action_concurrency` setting at org/domain/project scope) applies. Must be 0
            (platform default) or at least 2 — a value of 1 would deadlock the run, since the parent
            action holds a concurrency slot while waiting for its child actions.
        notifications: Optional Notification(s) to send when the run reaches specific execution phases.
            Accepts a single notification or a tuple of notifications. Supports Email, Slack, Teams, and Webhook types.
            See `flyte.notify` for available notification types and template variables.
        custom_context: Optional global input context to pass to the task. This will be available via
            get_custom_context() within the task and will automatically propagate to sub-tasks.
            Acts as base/default values that can be overridden by context managers in the code.
        cache_lookup_scope: Optional Scope to use for the run. This is used to specify the scope to use for cache
            lookups. If not specified, it will be set to the default scope (global unless overridden at the system
            level).
        preserve_original_types: Optional If true, the type engine will preserve original types (e.g., pd.DataFrame)
            when guessing python types from literal types. If false (default), it will return the generic
            flyte.io.DataFrame. This option is automatically set to True if interactive_mode is True unless overridden
            explicitly by this parameter.
        debug: Optional If true, the task will be run as a VSCode debug task, starting a code-server in the
            container so users can connect via the UI to interactively debug/run the task.
        tracked: Local-only. If true, report tracked run state (actions, attempts, outputs, reports)
            to the Flyte control plane via TrackedRunService so the run shows up in the console. Requires
            an initialized client and a configured project/domain. Can also be enabled globally with the
            `local.tracked` config key. Reporting is best-effort and never fails the local run.
        tracked_strict: Local-only, for debugging reporting itself. When true (with `tracked`),
            the first reporting failure — registration, an artifact upload, a rejected or undeliverable
            ReportActions update, or a flush timeout — fails the run loudly instead of being logged and
            swallowed. Can also be enabled globally with the `local.tracked_strict` config key.
        _tracker: This is an internal only parameter used by the CLI to render the TUI.

    Returns:
        runner

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
        tracked=tracked,
        tracked_strict=tracked_strict,
        _tracker=_tracker,
    )


@syncify
async def run(
    task: TaskTemplate[P, R, F] | LazyEntity | RemoteTrigger | TriggerDetails, *args: P.args, **kwargs: P.kwargs
) -> Run:
    """
    Run a task with the given parameters, or fire a deployed trigger on demand.

    ```python
    trigger = flyte.remote.Trigger.get(name="full-report", task_name="reports.report")
    run = flyte.run(trigger)              # the trigger's inputs, env vars, queue, notifications
    run = flyte.run(trigger, days=7)      # override one input, keep the rest
    ```

    Args:
        task: task to run, or a deployed trigger (`flyte.remote.Trigger.get(...)`)
        args: args to pass to the task (not allowed for a trigger)
        kwargs: kwargs to pass to the task (for a trigger: overrides of its registered inputs)

    Returns:
        Run | Result of the task
    """
    # using syncer causes problems
    return await _Runner().run.aio(task, *args, **kwargs)  # type: ignore


@syncify
async def rerun(
    run_name: str,
    action_name: str = "a0",
    recover: bool = False,
    force_rerun_actions: Sequence[str] | None = None,
    allow_missing_source_outputs: bool = False,
    **inputs: Any,
) -> Run:
    """Re-run a prior run, returning a new `Run`.

    `rerun("r1")` creates a whole new run with the prior run's exact inputs (fetching its code from
    the platform); `rerun("r1", recover=True)` does the same but reuses the prior run's succeeded
    actions, re-executing only what failed or never ran. Pass keyword inputs to change
    parameters (`rerun("r1", x=2)`); inputs left out keep the prior run's values. New inputs
    combine with recovery (`rerun("r1", recover=True, x=2)`), in which case recovered actions keep
    the outputs they produced under the original inputs unless listed in `force_rerun_actions`.
    Use `with_runcontext(...).rerun(...)` to apply run-context overrides (env_vars, labels, …).
    The prior run's code is always replayed as-is.

    Args:
        run_name: Name of the prior run to re-run.
        action_name: Action within the prior run to source the task + inputs from. Defaults to
            `a0`, the root action — i.e. the whole run. Naming a child action instead roots the new
            run at that action's task, run with the exact inputs it received. Cannot be combined
            with `recover`.
        recover: Reuse the prior run's succeeded actions, re-running only what failed or never ran.
            Remote-only; requires a backend (and flyteidl2 build) with RunSpec.relation recovery
            support.
        force_rerun_actions: With `recover`, names of actions that must re-execute even though they
            succeeded in the source run (escape hatch). A listed parent action re-enqueues its
            children — list them too to force the whole subtree. Unknown names are ignored.
        allow_missing_source_outputs: Proceed when the source run's outputs were cleaned up from
            storage, using its inputs URI directly. The client cannot verify the inputs still
            exist — if they were deleted too, the new run fails at runtime. Irrelevant when the
            new inputs cover every input of the task, since the source inputs are then not read
            at all.
        inputs: Optional native keyword inputs to change parameters. Any input not passed keeps
            the source run's value, so passing none reuses the source run's inputs wholesale.

    Returns:
        the new Run.
    """
    return await _Runner().rerun.aio(
        run_name,
        action_name,
        recover=recover,
        force_rerun_actions=force_rerun_actions,
        allow_missing_source_outputs=allow_missing_source_outputs,
        **inputs,
    )
