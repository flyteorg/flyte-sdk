"""Shared test setup: keep hermes-agent's config sandboxed and offline.

``hermes-agent`` reads/writes its config under ``HERMES_HOME`` (default
``~/.hermes``); point it at a throwaway directory so tests never touch — or
depend on — the developer's real Hermes setup.
"""

import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-home-test-"))
