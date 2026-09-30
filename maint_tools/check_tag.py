"""Check the format of a release version string.

The release workflow runs this before doing anything else so that a typo like
``v2.9`` or ``2.9.0`` fails fast and cheaply instead of halfway through a
publish. Valid forms are exactly ``v<major>.<minor>.<patch>`` with an optional
pre-release suffix:

    v2.9.0        final release
    v2.9.0b1      beta release
    v2.9.0a0      alpha release

The accepted grammar is ``^v\\d+\\.\\d+\\.\\d+([ab]\\d+)?$``. This is intentionally
tight, matching the ``tag_regex`` that ``setuptools_scm`` uses in ``pyproject.toml``
(``^v(?P<version>\\d+\\.\\d+\\.\\d+.*)$``) so that every tag we cut also produces a
valid PEP 440 wheel version.
"""

import re
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
parser.add_argument("version", help="Version to validate, e.g. v2.9.0 or v2.9.0b1")
parser.add_argument(
    "--pattern",
    default=r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<pre>[ab]\d+)?",
    help="Regex (without leading 'v') describing a valid version (default: %(default)s)",
)

args = parser.parse_args()

version = args.version

if not version.startswith("v"):
    print(f"error: version {version!r} must start with 'v' (e.g. v2.9.0)", file=sys.stderr)
    sys.exit(1)

body = version[1:]

match = re.fullmatch(args.pattern, body)
if not match:
    print(
        f"error: version {version!r} does not match the release pattern 'v{args.pattern}' (e.g. v2.9.0 or v2.9.0b1)",
        file=sys.stderr,
    )
    sys.exit(1)

parts = match.groupdict()
print(
    f"ok: {version} (major={parts['major']}, minor={parts['minor']}, "
    f"patch={parts['patch']}, pre={parts.get('pre') or 'none'})"
)
