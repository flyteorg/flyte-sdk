from __future__ import annotations

import pathlib

import rich_click as click

from . import _common as common
from ._common import CLIConfig
from ._deploy import DeployArguments


@click.command("deploy-examples")
@click.option(
    "--type",
    "example_type",
    type=str,
    default="basics",
    show_default=True,
    help="The example folder to deploy (folder name under examples/).",
)
@click.pass_context
def deploy_examples(ctx: click.Context, example_type: str, **kwargs):
    """Deploy bundled example workflows from the examples directory.

    By default deploys the **basics** examples. Use ``--type`` to choose
    a different folder (e.g. ``--type genai``).

    All standard deploy options (``--project``, ``--domain``, ``--version``, etc.)
    are supported.

    \b
    Examples:
        flyte deploy-examples
        flyte deploy-examples --type genai
        flyte deploy-examples --type ml --project my-project
    """
    import importlib.util
    import sys

    import flyte
    from flyte._environment import list_loaded_environments
    from flyte._status import status

    deploy_params = {k: v for k, v in ctx.params.items() if k != "example_type"}
    deploy_args = DeployArguments.from_dict(deploy_params)

    obj: CLIConfig = ctx.obj
    common.initialize_config(
        ctx=ctx,
        project=deploy_args.project,
        domain=deploy_args.domain,
        root_dir=deploy_args.root_dir,
        sync_local_sys_paths=not deploy_args.no_sync_local_sys_paths,
        images=tuple(deploy_args.image) or None,
    )

    # Locate the examples directory relative to the package root
    examples_root = pathlib.Path(__file__).resolve().parents[3] / "examples"
    examples_dir = examples_root / example_type

    if not examples_dir.is_dir():
        available = sorted(d.name for d in examples_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
        raise click.ClickException(f"Example type '{example_type}' not found. Available types: {', '.join(available)}")

    status.step(f"Loading examples from {examples_dir}")

    # Add the examples directory to sys.path so sibling imports within examples work
    examples_dir_str = str(examples_dir)
    if examples_dir_str not in sys.path:
        sys.path.insert(0, examples_dir_str)

    # Load example modules file-by-file using spec_from_file_location to avoid
    # package resolution issues. Examples are standalone scripts, not proper packages.
    python_files = sorted(f for f in examples_dir.glob("*.py") if f.name != "__init__.py")
    loaded_modules = []
    failed_paths = []
    for file_path in python_files:
        module_name = file_path.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                failed_paths.append((file_path, "Could not create module spec"))
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded_modules.append(module)
        except Exception as e:
            failed_paths.append((file_path, str(e)))

    if failed_paths:
        status.warn(f"Loaded {len(loaded_modules)} modules, but failed to load {len(failed_paths)} paths")
        common.print_output(
            common.format("Modules", [[("Path", str(p)), ("Err", e)] for p, e in failed_paths], obj.output_format),
            obj.output_format,
        )
        if not deploy_args.ignore_load_errors:
            raise click.ClickException(
                f"Failed to load {len(failed_paths)} files. Use --ignore-load-errors to ignore these errors."
            )
    else:
        status.info(f"Loaded {len(loaded_modules)} modules")

    all_envs_raw = list_loaded_environments()
    # Deduplicate environments by name — examples may define envs with the same name
    seen_names: dict[str, object] = {}
    all_envs = []
    for env in all_envs_raw:
        if env.name not in seen_names:
            seen_names[env.name] = env
            all_envs.append(env)

    if not all_envs:
        status.info("No environments found to deploy")
        return

    common.print_output(
        common.format("Loaded Environments", [[("name", e.name)] for e in all_envs], obj.output_format),
        obj.output_format,
    )

    with common.cli_status(obj.output_format, "Deploying..."):
        deployments = flyte.deploy(
            *all_envs,
            dryrun=deploy_args.dry_run,
            copy_style=deploy_args.copy_style,
            version=deploy_args.version,
        )

    common.print_output(
        common.format("Environments", [env for d in deployments for env in d.env_repr()], obj.output_format),
        obj.output_format,
    )
    common.print_output(
        common.format("Tasks", [task for d in deployments for task in d.table_repr()], obj.output_format),
        obj.output_format,
    )


# Attach all the standard deploy options to the command
for opt in DeployArguments.options():
    deploy_examples.params.append(opt)
