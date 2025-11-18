"""
Runtime entrypoint for Databricks connector.
"""

import sys
from flyte._bin.runtime import main as _a0

import click


def main():
    args = sys.argv
    click_ctx = click.Context(click.Command("databricks"))
    if args[1] == "a0":
        parser = _a0.make_parser(click_ctx)
        args, _, _ = parser.parse_args(args[2:])
        with click_ctx:
            _a0.callback(**args, debug=False, interactive_mode=False, tgz=None, pkl=None, dest=None)
    else:
        raise ValueError(f"Unrecognized command: {args[1:]}")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    main()
