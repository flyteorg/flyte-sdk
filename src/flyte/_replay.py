from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, cast

from flyte._environment import Environment
from flyte._initialize import get_client, get_init_config
from flyte._run import Mode
from flyte._task import F, P, R, TaskTemplate
from flyte.models import (
    ActionID,
    RawDataPath,
    SerializationContext,
    TaskContext,
)
from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyte.remote import Run


class _Replayer:
    """Internal class that handles replay logic."""

    def __init__(
        self,
        mode: Mode | None = None,
    ):
        self._mode = mode

    @syncify
    async def replay(
        self,
        run_name: str,
        action_name: str = "a0",
        task_template: TaskTemplate | None = None,
    ) -> Run:
        """Execute the replay: fetch original run's inputs and RunSpec, then launch a new run."""
        from flyte._initialize import ensure_client
        from flyte.remote import ActionDetails, RunDetails

        ensure_client()

        # Determine mode
        mode = self._mode
        if mode is None:
            client = get_client()
            if client is not None:
                mode = "remote"
            else:
                mode = "local"

        if mode == "local" and task_template is None:
            raise ValueError(
                "Local replay requires a task_template to be provided. "
                "Without a TaskTemplate, there is no Python function to execute locally."
            )

        # Step 1: Fetch RunDetails to get the RunSpec and root action details
        run_details = await RunDetails.get.aio(name=run_name)
        original_run_spec = run_details.pb2.run_spec

        # Step 2: Get the action details for the requested action
        if action_name == "a0":
            # Root action is already available in run_details
            action_details = run_details.action_details
        else:
            action_details = await ActionDetails.get.aio(
                run_name=run_name,
                name=action_name,
            )

        # Step 3: Fetch raw proto inputs via get_action_data
        from flyteidl2.workflow import run_service_pb2

        resp = await get_client().run_service.get_action_data(
            request=run_service_pb2.GetActionDataRequest(
                action_id=action_details.pb2.id,
            )
        )
        raw_inputs = resp.inputs

        # Step 4: Determine task_spec
        if task_template is None:
            # Reuse the resolved task spec from the original action
            task_spec = action_details.pb2.resolved_task_spec
        else:
            # Build a fresh task_spec from the provided template
            task_spec = await self._build_task_spec(task_template)

        # Step 5: Dispatch by mode
        if mode == "remote":
            return await self._replay_remote(task_spec, raw_inputs, original_run_spec)
        elif mode == "local":
            return await self._replay_local(task_spec, raw_inputs, action_details, task_template)
        elif mode == "hybrid":
            return await self._replay_hybrid(task_spec, raw_inputs, action_details, original_run_spec)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    async def _build_task_spec(self, task: TaskTemplate):
        """Build a task_spec from a TaskTemplate, including code bundling and image building."""
        import flyte.report
        from flyte._code_bundle import build_code_bundle
        from flyte._deploy import build_images
        from flyte._image import Image, resolve_code_bundle_layer
        from flyte._initialize import get_init_config
        from flyte._internal.runtime.task_serde import translate_task_to_wire

        cfg = get_init_config()

        if task.parent_env is None:
            raise ValueError("Task is not attached to an environment. Please attach the task to an environment.")

        parent_env = cast(Environment, task.parent_env())

        from flyte._deploy import plan_deploy

        for _env in plan_deploy(parent_env)[0].envs.values():
            if isinstance(_env.image, Image):
                _env.image = resolve_code_bundle_layer(_env.image, "loaded_modules", pathlib.Path(cfg.root_dir))

        image_cache = await build_images.aio(parent_env)

        code_bundle = await build_code_bundle(
            from_dir=cfg.root_dir,
            dryrun=False,
            copy_bundle_to=None,
            copy_style="loaded_modules",
        )

        version = code_bundle.computed_version if code_bundle and code_bundle.computed_version else None
        if not version:
            raise ValueError("Version is required when running a task")

        project = cfg.project
        domain = cfg.domain
        org = cfg.org

        s_ctx = SerializationContext(
            code_bundle=code_bundle,
            version=version,
            image_cache=image_cache,
            root_dir=cfg.root_dir,
        )
        action = ActionID(name="{{.actionName}}", run_name="{{.runName}}", project=project, domain=domain, org=org)
        tctx = TaskContext(
            action=action,
            code_bundle=code_bundle,
            output_path="",
            version=version,
            raw_data_path=RawDataPath(path=""),
            compiled_image_cache=image_cache,
            run_base_dir="",
            report=flyte.report.Report(name=action.name),
        )
        return translate_task_to_wire(task, s_ctx, default_inputs=None, task_context=tctx)

    async def _replay_remote(self, task_spec, raw_inputs, original_run_spec) -> Run:
        """Replay by creating a new remote run with the original RunSpec and inputs."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from flyteidl2.common import identifier_pb2
        from flyteidl2.dataproxy import dataproxy_service_pb2
        from flyteidl2.workflow import run_service_pb2

        import flyte.errors
        from flyte.remote import Run

        cfg = get_init_config()
        project_id = identifier_pb2.ProjectIdentifier(
            name=cfg.project,
            domain=cfg.domain,
            organization=cfg.org,
        )

        # Upload inputs via dataproxy
        upload_req = dataproxy_service_pb2.UploadInputsRequest(
            inputs=raw_inputs,
            task_spec=task_spec,
        )
        upload_req.project_id.CopyFrom(project_id)
        upload_resp = await get_client().dataproxy_service.upload_inputs(upload_req)

        # Create run with original RunSpec
        try:
            resp = await get_client().run_service.create_run(
                run_service_pb2.CreateRunRequest(
                    project_id=project_id,
                    task_spec=task_spec,
                    offloaded_input_data=upload_resp.offloaded_input_data,
                    run_spec=original_run_spec,
                ),
            )
            return Run(pb2=resp.run)
        except ConnectError as e:
            if e.code == Code.UNAVAILABLE:
                raise flyte.errors.RuntimeSystemError(
                    "SystemUnavailableError",
                    "Flyte system is currently unavailable. Check your configuration, or the service status.",
                ) from e
            elif e.code == Code.INVALID_ARGUMENT:
                raise flyte.errors.RuntimeUserError("InvalidArgumentError", e.message)
            elif e.code == Code.ALREADY_EXISTS:
                raise flyte.errors.RuntimeUserError(
                    "RunAlreadyExistsError",
                    "A run with this name already exists. Please choose a different name.",
                )
            else:
                raise flyte.errors.RuntimeSystemError(
                    "RunCreationError",
                    f"Failed to create run: {e.message}",
                ) from e

    async def _replay_local(self, task_spec, raw_inputs, action_details, task_template) -> Run:
        """Replay locally by converting inputs to native and executing the task."""
        import flyte.types as types
        from flyte._internal.runtime import convert
        from flyte._run import run_task_locally

        task = task_template
        assert task is not None  # validated in replay()

        # Convert proto inputs to native Python types
        native_iface = None
        if action_details.pb2.HasField("task"):
            iface = action_details.pb2.task.task_template.interface
            native_iface = types.guess_interface(iface)

        if native_iface is None:
            native_iface = task.native_interface

        native_inputs = await convert.convert_from_inputs_to_native(native_iface, convert.Inputs(raw_inputs))

        return await run_task_locally(task, run_label="replay-local", **native_inputs)

    async def _replay_hybrid(self, task_spec, raw_inputs, action_details, original_run_spec) -> Any:
        """Replay in hybrid mode: run parent locally, children remotely."""
        # Hybrid replay is not yet implemented
        raise ValueError(
            "Hybrid replay requires a run_base_dir. Use with_replaycontext with additional configuration, "
            "or use remote mode for replaying runs."
        )


def with_replaycontext(
    mode: Mode | None = None,
) -> _Replayer:
    """
    Create a replay context with the given mode.

    Supports the same modes as with_runcontext: local, remote, and hybrid.

    Example::

        flyte.with_replaycontext(mode="remote").replay("my-run-name", action_name="a0")

    :param mode: The execution mode - "local", "remote", or "hybrid". If not provided,
        defaults to "remote" if a client is configured, else "local".
    :return: A _Replayer with a .replay() method.
    """
    return _Replayer(mode=mode)


@syncify
async def replay(
    run_name: str,
    action_name: str = "a0",
    task_template: TaskTemplate[P, R, F] | None = None,
) -> Run:
    """
    Replay an existing run by re-executing it with the same inputs and RunSpec.

    Retrieves the entire RunSpec and inputs from the original run/action, then launches
    a new run. If task_template is not provided, the original remote task template is used.
    If task_template is provided, the new code is bundled and used with the original inputs.

    Example::

        # Replay with original task template
        flyte.replay("my-run-name")

        # Replay a specific action
        flyte.replay("my-run-name", action_name="a1")

        # Replay with new code
        flyte.replay("my-run-name", task_template=my_updated_task)

    :param run_name: The name of the run to replay.
    :param action_name: The name of the action to replay inputs from. Defaults to "a0" (root action).
    :param task_template: Optional new TaskTemplate to use. If not provided, the original
        remote task template is used.
    :return: A Run object representing the new run.
    """
    return await _Replayer().replay.aio(
        run_name=run_name,
        action_name=action_name,
        task_template=task_template,
    )
