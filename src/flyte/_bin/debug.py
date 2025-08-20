import click


@click.group()
def _debug():
    """Debug commands for Flyte."""


@_debug.command("resume")
@click.option("--pid", "-m", type=int, required=True, help="PID of the vscode server.")
def resume(pid):
    """
    Resume a Flyte task for debugging purposes.

    Args:
        pid (int): PID of the vscode server.
    """
    import os
    import signal

    print("Terminating server and resuming task.")
    answer = (
        input(
            "This operation will kill the server. All unsaved data will be lost,"
            " and you will no longer be able to connect to it. Do you really want to terminate? (Y/N): "
        )
        .strip()
        .upper()
    )
    if answer == "Y":
        os.kill(pid, signal.SIGTERM)
        print("The server has been terminated and the task has been resumed.")
    else:
        print("Operation canceled.")


if __name__ == "__main__":
    _debug()
