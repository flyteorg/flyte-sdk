"""Fixtures + eager task registration for the functional (environment-validation) suite.

The pytest option/markers/ordering live in the installed plugin (``plugin.py``, a
``pytest11`` entry point). This conftest loads only when the suite itself is collected,
so it holds the heavier bits it wouldn't be safe to run globally:

* Import every scenario task module up front. Two reasons: their module-level
  ``TaskEnvironment()``s must register before the first ``flyte.init`` so a later import
  can't reset routing; and — critically for ``app`` — this pulls each task module into the
  fast-register code bundle, so the serving pod can re-import it (a lazily imported module
  is absent from the bundle). Importing ``app`` builds a FastAPI app at import time, so the
  ``[app]`` extra (fastapi/httpx) must be installed wherever the suite is collected.
* The per-test client-init fixtures (see ``flyte_ops``).

Task definitions live in ``tasks/`` (one module per scenario); connection and retry
helpers live in ``flyte_ops.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from . import flyte_ops
from .tasks import app, imgbuild, imgcache, reusable, simple, trigger  # noqa: F401


@pytest.fixture(scope="session")
def suite_config() -> dict:
    """Resolved connection + naming knobs for the run (see flyte_ops.env_config)."""
    return flyte_ops.env_config()


@pytest.fixture
def flyte_ctx(suite_config) -> dict:
    """(Re-)initialise the flyte client before each test.

    Re-init per test because importing a task module (module-level
    ``TaskEnvironment()``) can reset the client's project/org routing.
    """
    asyncio.run(flyte_ops.init_client(suite_config))
    return suite_config
