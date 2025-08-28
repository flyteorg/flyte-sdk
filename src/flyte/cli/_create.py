from pathlib import Path
from typing import Any, Dict, get_args

import rich_click as click

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
    mutually_exclusive=["from_file"],
)
@click.option(
    "--from-file",
    type=click.Path(exists=True),
    help="Path to the file with the binary secret.",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["value"],
)
@click.option(
    "--type", type=click.Choice(get_args(SecretTypes)), default="regular", help="Type of the secret.", show_default=True
)
@click.pass_obj
def secret(
    cfg: common.CLIConfig,
    name: str,
    value: str | bytes | None = None,
    from_file: str | None = None,
    type: SecretTypes = "regular",
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

    ```bash
    $ flyte create secret my_secret --type image_pull
    ```
    """
    from flyte.remote import Secret

    cfg.init(project, domain)
    if from_file:
        with open(from_file, "rb") as f:
            value = f.read()
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
    default=Path.cwd() / "config.yaml",
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
