from typing import get_args

import rich_click as click

import flyte.remote as remote
from flyte.remote import SecretTypes

from . import _common as common
from ._option import DependentOption, MutuallyExclusiveOption


@click.group(name="update")
def update():
    """
    Update various flyte entities.
    """


@update.command("project", cls=click.RichCommand)
@click.argument("id", type=str)
@click.option("--name", type=str, default=None, help="Update the project display name.")
@click.option("--description", type=str, default=None, help="Update the project description.")
@click.option(
    "--label",
    "-l",
    multiple=True,
    callback=common.key_value_callback,
    help="Set labels as key=value pairs. Can be specified multiple times. Replaces all existing labels.",
)
@click.option("--archive/--unarchive", default=None, help="Archive or unarchive the project.")
@click.pass_obj
def project(
    cfg: common.CLIConfig,
    id: str,
    name: str | None,
    description: str | None,
    label: dict[str, str] | None,
    archive: bool | None,
):
    """
    Update a project's name, description, labels, or archive state.

    \b
    Example usage:

    ```bash
    flyte update project my_project --archive
    flyte update project my_project --unarchive
    flyte update project my_project --description "New description"
    flyte update project my_project --name "New Display Name"
    flyte update project my_project --label team=ml --label env=prod
    ```
    """
    if name is None and description is None and label is None and archive is None:
        raise click.UsageError(
            "At least one of --name, --description, --label, or --archive/--unarchive must be provided."
        )

    cfg.init()
    console = common.get_console()

    state = None
    if archive is not None:
        state = "archived" if archive else "active"

    with console.status(f"Updating project {id}..."):
        remote.Project.update(id=id, name=name, description=description, labels=label, state=state)

    parts = []
    if name is not None:
        parts.append(f"name set to '{name}'")
    if description is not None:
        parts.append("description updated")
    if label is not None:
        parts.append("labels updated")
    if archive is True:
        parts.append("[red]archived[/red]")
    elif archive is False:
        parts.append("[green]unarchived[/green]")
    console.print(f"Project [bold]{id}[/bold]: {', '.join(parts)}.")


@update.command("trigger", cls=common.CommandBase)
@click.argument("name", type=str)
@click.argument("task_name", type=str)
@click.option("--activate/--deactivate", required=True, help="Activate or deactivate the trigger.")
@click.pass_obj
def trigger(cfg: common.CLIConfig, name: str, task_name: str, activate: bool, project: str | None, domain: str | None):
    """
    Update a trigger.

    \b
    Example usage:

    ```bash
    flyte update trigger <trigger_name> <task_name> --activate | --deactivate
    [--project <project_name> --domain <domain_name>]
    ```
    """
    cfg.init(project, domain)
    console = common.get_console()
    to_state = "active" if activate else "deactivate"
    with console.status(f"Updating trigger {name} for task {task_name} to {to_state}..."):
        remote.Trigger.update(name, task_name, activate)
    console.print(f"Trigger updated and is set to [fuchsia]{to_state}[/fuchsia]")


@update.command("app", cls=common.CommandBase)
@click.argument("name", type=str)
@click.option("--activate/--deactivate", "is_activate", default=None, help="Activate or deactivate app.")
@click.option("--wait", is_flag=True, default=False, help="Wait for the app to reach the desired state.")
@click.pass_obj
def app(
    cfg: common.CLIConfig, name: str, is_activate: bool | None, wait: bool, project: str | None, domain: str | None
):
    """
    Update an app by starting or stopping it.

    \b
    Example usage:

    ```bash
    flyte update app <app_name> --activate | --deactivate [--wait] [--project <project_name>] [--domain <domain_name>]
    ```
    """

    if is_activate is None:
        raise click.UsageError("Missing option '--activate' / '--deactivate'.")

    cfg.init(project, domain)
    console = common.get_console()

    app_obj = remote.App.get(name=name)
    if is_activate:
        if app_obj.is_active():
            console.print(f"App [yellow]{name}[/yellow] is already active.")
            return
        state = "activate"
        color = "green"
        with console.status(f"Activating app {name}..."):
            app_obj.activate(wait=wait)
    else:
        state = "deactivate"
        color = "red"
        if app_obj.is_deactivated():
            console.print(f"App [red]{name}[/red] is already deactivated.")
            return
        with console.status(f"Deactivating app {name}..."):
            app_obj.deactivate(wait=wait)

    if wait:
        console.print(f"App [{color}]{name}[/{color}] {state}d successfully")
    else:
        console.print(f"App [{color}]{name}[/{color}] {state} initiated")


@update.command(cls=common.CommandBase)
@click.argument("name", type=str, required=True)
@click.option(
    "--value",
    help="Secret value",
    prompt="Enter secret value",
    hide_input=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["from_file", "from_docker_config", "registry"],
)
@click.option(
    "--from-file",
    type=click.Path(exists=True),
    help="Path to the file with the binary secret.",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["value", "from_docker_config", "registry"],
)
@click.option(
    "--type", type=click.Choice(get_args(SecretTypes)), default="regular", help="Type of the secret.", show_default=True
)
@click.option(
    "--from-docker-config",
    is_flag=True,
    help="Create image pull secret from Docker config file (only for --type image_pull).",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["value", "from_file", "registry", "username", "password"],
)
@click.option(
    "--docker-config-path",
    type=click.Path(exists=True),
    cls=DependentOption,
    help="Path to Docker config file (defaults to ~/.docker/config.json or $DOCKER_CONFIG).",
    requires=["from_docker_config"],
)
@click.option(
    "--registries",
    help="Comma-separated list of registries to include (only with --from-docker-config).",
)
@click.option(
    "--registry",
    help="Registry hostname (e.g., ghcr.io, docker.io) for explicit credentials (only for --type image_pull).",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["value", "from_file", "from_docker_config"],
)
@click.option(
    "--username",
    help="Username for the registry (only with --registry).",
)
@click.option(
    "--password",
    help="Password for the registry (only with --registry). If not provided, will prompt.",
    hide_input=True,
)
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str,
    value: str | bytes | None = None,
    from_file: str | None = None,
    type: SecretTypes = "regular",
    from_docker_config: bool = False,
    docker_config_path: str | None = None,
    registries: str | None = None,
    registry: str | None = None,
    username: str | None = None,
    password: str | None = None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Update (replace) an existing secret by deleting and recreating it.

    \b
    Example usage:

    ```bash
    flyte update secret my_secret --value my_new_value
    flyte update secret my_secret --from-file /path/to/secret_file
    flyte update secret my_secret --type image_pull --registry ghcr.io --username myuser
    ```
    """
    from flyte.remote import Secret

    # todo: remove this hack when secrets creation more easily distinguishes between org and project/domain level
    #   (and domain level) secrets
    project = "" if project is None else project
    domain = "" if domain is None else domain
    cfg.init(project, domain)

    console = common.get_console()

    # Handle image pull secret creation
    if type == "image_pull":
        if project != "" or domain != "":
            raise click.ClickException("Project and domain must not be set when creating an image pull secret.")

        if from_docker_config:
            from flyte._utils.docker_credentials import create_dockerconfigjson_from_config

            registry_list = [r.strip() for r in registries.split(",")] if registries else None
            try:
                value = create_dockerconfigjson_from_config(
                    registries=registry_list,
                    docker_config_path=docker_config_path,
                )
            except Exception as e:
                raise click.ClickException(f"Failed to create dockerconfigjson from Docker config: {e}") from e

        elif registry:
            from flyte._utils.docker_credentials import create_dockerconfigjson_from_credentials

            if not username:
                username = click.prompt("Username")
            if not password:
                password = click.prompt("Password", hide_input=True)

            value = create_dockerconfigjson_from_credentials(registry, username, password)

        else:
            from flyte._utils.docker_credentials import create_dockerconfigjson_from_credentials

            registry = click.prompt("Registry (e.g., ghcr.io, docker.io)")
            username = click.prompt("Username")
            password = click.prompt("Password", hide_input=True)

            value = create_dockerconfigjson_from_credentials(registry, username, password)

    elif from_file:
        with open(from_file, "rb") as f:
            value = f.read()

    # Encode string values to bytes
    if isinstance(value, str):
        value = value.encode("utf-8")

    with console.status(f"Deleting secret {name}..."):
        Secret.delete(name=name)

    with console.status(f"Creating secret {name}..."):
        Secret.create(name=name, value=value, type=type)

    console.print(f"[bold green]Secret {name} updated successfully![/bold green]")
    console.print("[yellow]Note: Secret replication may take a few moments to propagate across the cluster.[/yellow]")
