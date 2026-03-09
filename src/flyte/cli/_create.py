from pathlib import Path
from typing import Any, Dict

import rich_click as click

import flyte
import flyte.cli._common as common


@click.group(name="create")
def create():
    """
    Create resources in a Flyte deployment.
    """


@create.command("project", cls=click.RichCommand)
@click.option("--id", type=str, required=True, help="Unique identifier for the project (immutable).")
@click.option("--name", type=str, required=True, help="Display name for the project.")
@click.option("--description", type=str, default="", help="Description for the project.")
@click.option(
    "--label",
    "-l",
    multiple=True,
    callback=common.key_value_callback,
    help="Labels as key=value pairs. Can be specified multiple times.",
)
@click.pass_obj
def project(cfg: common.CLIConfig, id: str, name: str, description: str, label: dict[str, str] | None):
    """
    Create a new project.

    \b
    Example usage:

    ```bash
    flyte create project --id my_project_id --name "My Project"
    flyte create project --id my_project_id --name "My Project" --description "My project" -l team=ml -l env=prod
    ```
    """
    from flyte.remote import Project

    cfg.init()
    console = common.get_console()
    with console.status(f"Creating project {id}..."):
        Project.create(id=id, name=name, description=description, labels=label)
    console.print(f"[bold green]Project {id} created successfully![/bold green]")


@create.command(cls=common.CommandBase)
@common.secret_options
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str,
    value: str | bytes | None = None,
    from_file: str | None = None,
    type: str = "regular",
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
    Create a new secret. The name of the secret is required. For example:

    ```bash
    $ flyte create secret my_secret --value my_value
    ```

    If you don't provide a `--value` flag, you will be prompted to enter the
    secret value in the terminal.

    ```bash
    $ flyte create secret my_secret
    Enter secret value:
    ```

    If `--from-file` is specified, the value will be read from the file instead of being provided directly:

    ```bash
    $ flyte create secret my_secret --from-file /path/to/secret_file
    ```

    The `--type` option can be used to create specific types of secrets.
    Either `regular` or `image_pull` can be specified.
    Secrets intended to access container images should be specified as `image_pull`.
    Other secrets should be specified as `regular`.
    If no type is specified, `regular` is assumed.

    For image pull secrets, you have several options:

    1. Interactive mode (prompts for registry, username, password):
    ```bash
    $ flyte create secret my_secret --type image_pull
    ```

    2. With explicit credentials:
    ```bash
    $ flyte create secret my_secret --type image_pull --registry ghcr.io --username myuser
    ```

    3. Lastly, you can create a secret from your existing Docker installation (i.e., you've run `docker login` in
    the past) and you just want to pull from those credentials. Since you may have logged in to multiple registries,
    you can specify which registries to include. If no registries are specified, all registries are added.
    ```bash
    $ flyte create secret my_secret --type image_pull --from-docker-config --registries ghcr.io,docker.io
    ```
    """
    from flyte.remote import Secret

    # todo: remove this hack when secrets creation more easily distinguishes between org and project/domain level
    #   (and domain level) secrets
    project = "" if project is None else project
    domain = "" if domain is None else domain
    cfg.init(project, domain)

    value = common.resolve_secret_value(
        value, from_file, type, from_docker_config, docker_config_path, registries, registry, username, password,
        project, domain,
    )
    Secret.create(name=name, value=value, type=type)


@create.command(cls=common.CommandBase)
@click.option("--endpoint", type=str, help="Endpoint of the Flyte backend.")
@click.option("--insecure", is_flag=True, help="Use an insecure connection to the Flyte backend.")
@click.option(
    "--org",
    type=str,
    required=False,
    help="Organization to use. This will override the organization in the configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, writable=True),
    default=Path.cwd() / ".flyte" / "config.yaml",
    help="Path to the output directory where the configuration will be saved. Defaults to current directory.",
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force overwrite of the configuration file if it already exists.",
    show_default=True,
)
@click.option(
    "--image-builder",
    "--builder",
    type=click.Choice(["local", "remote"]),
    default="local",
    help="Image builder to use for building images. Defaults to 'local'.",
    show_default=True,
)
@click.option(
    "--auth-type",
    type=click.Choice(common.ALL_AUTH_OPTIONS, case_sensitive=False),
    default=None,
    help="Authentication type to use for the Flyte backend. Defaults to 'pkce'.",
    show_default=True,
    required=False,
)
@click.option(
    "--local-persistence",
    is_flag=True,
    default=False,
    help="Enable SQLite persistence for local run metadata, allowing past runs to be browsed via 'flyte start tui'.",
    show_default=True,
)
def config(
    output: str,
    endpoint: str | None = None,
    insecure: bool = False,
    org: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    force: bool = False,
    image_builder: str | None = None,
    auth_type: str | None = None,
    local_persistence: bool = False,
):
    """
    Creates a configuration file for Flyte CLI.
    If the `--output` option is not specified, it will create a file named `config.yaml` in the current directory.
    If the file already exists, it will raise an error unless the `--force` option is used.
    """
    import yaml

    from flyte._utils import org_from_endpoint, sanitize_endpoint

    output_path = Path(output)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    if output_path.exists() and not force:
        force = click.confirm(f"Overwrite [{output_path}]?", default=False)
        if not force:
            click.echo(f"Will not overwrite the existing config file at {output_path}")
            return

    admin: Dict[str, Any] = {}
    if endpoint:
        endpoint = sanitize_endpoint(endpoint)
        admin["endpoint"] = endpoint
    if insecure:
        admin["insecure"] = insecure
    if auth_type:
        admin["authType"] = common.sanitize_auth_type(auth_type)

    if not org and endpoint:
        org = org_from_endpoint(endpoint)

    task: Dict[str, str] = {}
    if org:
        task["org"] = org
    if project:
        task["project"] = project
    if domain:
        task["domain"] = domain

    image: Dict[str, str] = {}
    if image_builder:
        image["builder"] = image_builder

    local: Dict[str, Any] = {}
    if local_persistence:
        local["persistence"] = True

    if not admin and not task and not local:
        raise click.BadParameter("At least one of --endpoint, --org, or --local-persistence must be provided.")

    with open(output_path, "w") as f:
        d: Dict[str, Any] = {}
        if admin:
            d["admin"] = admin
        if task:
            d["task"] = task
        if image:
            d["image"] = image
        if local:
            d["local"] = local
        yaml.dump(d, f)

    click.echo(f"Config file written to {output_path}")


@create.command(cls=common.CommandBase)
@click.argument("task_name", type=str, required=True)
@click.argument("name", type=str, required=True)
@click.option(
    "--schedule",
    type=str,
    required=True,
    help="Cron schedule for the trigger. Defaults to every minute.",
    show_default=True,
)
@click.option(
    "--description",
    type=str,
    default="",
    help="Description of the trigger.",
    show_default=True,
)
@click.option(
    "--auto-activate",
    is_flag=True,
    default=True,
    help="Whether the trigger should not be automatically activated. Defaults to True.",
    show_default=True,
)
@click.option(
    "--trigger-time-var",
    type=str,
    default="trigger_time",
    help="Variable name for the trigger time in the task inputs. Defaults to 'trigger_time'.",
    show_default=True,
)
@click.pass_obj
def trigger(
    cfg: common.CLIConfig,
    task_name: str,
    name: str,
    schedule: str,
    trigger_time_var: str = "trigger_time",
    auto_activate: bool = True,
    description: str = "",
    project: str | None = None,
    domain: str | None = None,
):
    """
    Create a new trigger for a task. The task name and trigger name are required.

    Example:

    ```bash
    $ flyte create trigger my_task my_trigger --schedule "0 0 * * *"
    ```

    This will create a trigger that runs every day at midnight.
    """
    from flyte.remote import Trigger

    cfg.init(project, domain)
    console = common.get_console()

    trigger = flyte.Trigger(
        name=name,
        automation=flyte.Cron(schedule),
        description=description,
        auto_activate=auto_activate,
        inputs={trigger_time_var: flyte.TriggerTime},  # Use the trigger time variable in inputs
        env_vars=None,
        interruptible=None,
    )
    with console.status("Creating trigger..."):
        v = Trigger.create(trigger, task_name=task_name)
    console.print(f"[bold green]Trigger {v.name} created successfully![/bold green]")
