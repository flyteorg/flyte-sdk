from pathlib import Path
from typing import Any, Dict, get_args

import rich_click as click

import flyte
import flyte.cli._common as common
from flyte.cli._option import MutuallyExclusiveOption
from flyte.remote import SecretTypes


@click.group(name="create")
def create():
    """
    Create resources in a Flyte deployment.
    """


@create.command(cls=common.CommandBase)
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
    help="Path to Docker config file (defaults to ~/.docker/config.json or $DOCKER_CONFIG).",
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

    # Handle image pull secret creation
    if type == "image_pull":
        if project != "" or domain != "":
            raise click.ClickException("Project and domain must not be set when creating an image pull secret.")

        if from_docker_config:
            # Mode 3: From Docker config
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
            # Mode 2: Explicit credentials
            from flyte._utils.docker_credentials import create_dockerconfigjson_from_credentials

            if not username:
                username = click.prompt("Username")
            if not password:
                password = click.prompt("Password", hide_input=True)

            value = create_dockerconfigjson_from_credentials(registry, username, password)

        else:
            # Mode 1: Interactive prompts
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

    if not admin and not task:
        raise click.BadParameter("At least one of --endpoint or --org must be provided.")

    with open(output_path, "w") as f:
        d: Dict[str, Any] = {}
        if admin:
            d["admin"] = admin
        if task:
            d["task"] = task
        if image:
            d["image"] = image
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
