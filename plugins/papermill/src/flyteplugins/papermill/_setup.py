"""Notebook kernel setup — called from the injected setup cell.

This module is imported inside the notebook kernel subprocess. Keep imports
at function scope so the module itself has no side-effects on import.
"""

from __future__ import annotations


def initialize_context() -> None:
    """Initialize the Flyte runtime context inside a notebook kernel.

    Reads the serialized task context from the ``_FLYTE_NB_CTX`` environment
    variable (set by the parent task runner before launching papermill) and
    reconstructs a ``TaskContext`` + ContextVar so that Flyte APIs work
    normally inside the notebook.

    Does nothing if the environment variable is not set.
    """
    import json
    import os

    raw = os.environ.get("_FLYTE_NB_CTX")
    if not raw:
        return

    try:
        data = json.loads(raw)

        import flyte.report as _report
        from flyte._context import Context, ContextData, root_context_var
        from flyte._internal.imagebuild.image_builder import ImageCache
        from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext

        if data.get("mode") == "local":
            # Local mode: no controller or remote connection needed.
            # Set _init_config directly to avoid @syncify running in a
            # background thread, which can cause visibility issues with
            # module-level globals in the kernel's main thread.
            import flyte._initialize as _init_mod

            with _init_mod._init_lock:
                if _init_mod._init_config is None:
                    from pathlib import Path

                    from flyte._initialize import _InitConfig

                    _init_mod._init_config = _InitConfig(root_dir=Path.cwd())
        else:
            # Remote mode: use the same init + controller pattern as runtime.py.
            from flyte._initialize import init_in_cluster
            from flyte._internal.controllers import create_controller

            controller_kwargs = init_in_cluster()
            create_controller(ct="remote", **controller_kwargs)

        action = ActionID(
            name=data["action_name"],
            run_name=data["run_name"],
            project=data["project"],
            domain=data["domain"],
            org=data["org"],
        )

        cb = None
        if "code_bundle" in data:
            cbd = data["code_bundle"]
            cb = CodeBundle(
                tgz=cbd.get("tgz"),
                pkl=cbd.get("pkl"),
                destination=cbd.get("destination", "."),
                computed_version=cbd.get("computed_version", ""),
            )

        ic = None
        if "image_cache" in data:
            ic = ImageCache.from_transport(data["image_cache"])

        tctx = TaskContext(
            action=action,
            version=data["version"],
            output_path=data["output_path"],
            run_base_dir=data["run_base_dir"],
            raw_data_path=RawDataPath(path=data["raw_data_path"]),
            mode=data["mode"],
            interactive_mode=data.get("interactive_mode", False),
            report=_report.Report(name=action.name),
            code_bundle=cb,
            compiled_image_cache=ic,
        )
        root_context_var.set(Context(data=ContextData(task_context=tctx)))
        print("[flyte-notebook] Context initialized successfully")
    except Exception as err:
        import traceback

        print(f"[flyte-notebook] WARNING: Failed to initialize context: {err}")
        traceback.print_exc()
