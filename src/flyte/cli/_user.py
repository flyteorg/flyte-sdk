import rich_click as click

import flyte.remote as remote

from . import _common as common


@click.command()
@click.pass_obj
def whoami(
    cfg: common.CLIConfig,
):
    """Display the current user information."""
    cfg.init()
    console = common.get_console()
    user_info = remote.User.get()
    console.print(user_info.to_json())
