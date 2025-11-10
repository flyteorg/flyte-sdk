import rich_click as click
from typing_extensions import get_args

from flyte._logging import LogFormat, initialize_logger, logger

from . import _common as common
from ._abort import abort
from ._build import build
from ._common import CLIConfig
from ._create import create
from ._delete import delete
from ._deploy import deploy
from ._gen import gen
from ._get import get
from ._plugins import discover_and_register_plugins
from ._run import run
from ._serve import serve
from ._update import update
from ._user import whoami

help_config = click.RichHelpConfiguration(
    use_markdown=True,
    use_markdown_emoji=True,
    command_groups={
        "flyte": [
            {
                "name": "Run and stop tasks",
                "commands": ["run", "abort"],
            },
            {
                "name": "Management of various objects.",
                "commands": ["create", "get", "delete", "update"],
            },
            {
                "name": "Build and deploy environments, tasks and images.",
                "commands": ["build", "deploy"],
            },
            {
                "name": "Documentation generation",
                "commands": ["gen"],
            },
            {
                "name": "User information",
                "commands": ["whoami"],
            },
        ]
    },
)


def _verbosity_to_loglevel(verbosity: int) -> int | None:
    """
    Converts a verbosity level from the CLI to a logging level.

    :param verbosity: verbosity level from the CLI
    :return: logging level
    """
    import logging

    match verbosity:
        case 0:
            return None
        case 1:
            return logging.WARNING
        case 2:
            return logging.INFO
        case _:
            return logging.DEBUG


@click.group(cls=click.RichGroup)
@click.option(
    "--endpoint",
    type=str,
    required=False,
    help="The endpoint to connect to. This will override any configuration file and simply use `pkce` to connect.",
)
@click.option(
    "--insecure",
    is_flag=True,
    required=False,
    help="Use an insecure connection to the endpoint. If not specified, the CLI will use TLS.",
    type=bool,
    default=None,
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
    "-v",
    "--verbose",
    required=False,
    help="Show verbose messages and exception traces. Repeating multiple times increases the verbosity (e.g., -vvv).",
    count=True,
    default=0,
    type=int,
)
@click.option(
    "--org",
    type=str,
    required=False,
    help="The organization to which the command applies.",
)
@click.option(
    "-c",
    "--config",
    "config_file",
    required=False,
    type=click.Path(exists=True),
    help="Path to the configuration file to use. If not specified, the default configuration file is used.",
)
@click.option(
    "--output-format",
    "-of",
    type=click.Choice(get_args(common.OutputFormat), case_sensitive=False),
    default="table",
    help="Output format for commands that support it. Defaults to 'table'.",
    show_default=True,
    required=False,
)
@click.option(
    "--log-format",
    type=click.Choice(get_args(LogFormat), case_sensitive=False),
    envvar="LOG_FORMAT",
    default="console",
    help="Formatting for logs, defaults to 'console' which is meant to be human readable."
    " 'json' is meant for machine parsing.",
    show_default=True,
    required=False,
)
@click.rich_config(help_config=help_config)
@click.pass_context
def main(
    ctx: click.Context,
    endpoint: str | None,
    insecure: bool,
    verbose: int,
    log_format: LogFormat,
    org: str | None,
    config_file: str | None,
    auth_type: str | None = None,
    output_format: common.OutputFormat = "table",
):
    """
    The Flyte CLI is the command line interface for working with the Flyte SDK and backend.

    It follows a simple verb/noun structure,
    where the top-level commands are verbs that describe the action to be taken,
    and the subcommands are nouns that describe the object of the action.

    The root command can be used to configure the CLI for persistent settings,
    such as the endpoint, organization, and verbosity level.

    Set endpoint and organization:

    ```bash
    $ flyte --endpoint <endpoint> --org <org> get project <project_name>
    ```

    Increase verbosity level (This is useful for debugging,
    this will show more logs and exception traces):

    ```bash
    $ flyte -vvv get logs <run-name>
    ```

    Override the default config file:

    ```bash
    $ flyte --config /path/to/config.yaml run ...
    ```

    * [Documentation](https://www.union.ai/docs/flyte/user-guide/)
    * [GitHub](https://github.com/flyteorg/flyte): Please leave a star if you like Flyte!
    * [Slack](https://slack.flyte.org): Join the community and ask questions.
    * [Issues](https://github.com/flyteorg/flyte/issues)

    """
    import flyte.config as config

    log_level = _verbosity_to_loglevel(verbose)
    if log_level is not None or log_format != "console":
        initialize_logger(log_level=log_level, log_format=log_format)

    cfg = config.auto(config_file=config_file)
    if cfg.source:
        logger.debug(f"Using config file discovered at location `{cfg.source.absolute()}`")

    ctx.obj = CLIConfig(
        log_level=log_level,
        log_format=log_format,
        endpoint=endpoint,
        insecure=insecure,
        org=org,
        config=cfg,
        ctx=ctx,
        auth_type=auth_type,
        output_format=output_format,
    )


main.add_command(run)
main.add_command(deploy)
main.add_command(get)  # type: ignore
main.add_command(create)  # type: ignore
main.add_command(abort)  # type: ignore
main.add_command(gen)  # type: ignore
main.add_command(delete)  # type: ignore
main.add_command(build)
main.add_command(whoami)  # type: ignore
main.add_command(update)  # type: ignore
main.add_command(serve)  # type: ignore

# Discover and register CLI plugins from installed packages
discover_and_register_plugins(main)
