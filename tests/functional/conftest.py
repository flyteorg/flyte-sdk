"""Fixtures for the functional (environment-validation) suite.

These tests run the flyte v2 scenarios end-to-end against a real Flyte backend
to show a deployment is healthy: it can run tasks, return I/O, build and cache
images, serve apps, and register scheduled triggers. They are marked
``integration`` (excluded from the default unit run) and require a configured
backend — see this directory's README.

Task definitions live in ``tasks/`` (one module per scenario). Connection and
retry helpers live in ``flyte_ops.py``.
"""

from __future__ import annotations

import asyncio

import pytest

# This dir and tasks/ are packages (they have __init__.py), so import within the
# package rather than mutating sys.path — a global sys.path insert would expose the
# generically-named task modules (app, trigger, …) tree-wide and shadow real
# packages when the suite is collected alongside the rest of the tests.
from . import flyte_ops

# Import the lightweight task modules once, up front: their module-level
# TaskEnvironment()s register before the first flyte.init so a later import can't
# reset routing. These import only `flyte` at module load. The `app` module is
# imported lazily inside test_app instead — it builds a FastAPI app at import
# time, so importing it here would pull `fastapi` into plain collection (breaking
# the unit run, which collects every test before -k filters this suite out).
from .tasks import imgbuild, imgcache, reusable, simple, trigger  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--skip",
        action="append",
        default=[],
        metavar="SCENARIO",
        help="Skip scenarios a backend doesn't support, by marker or test name. "
        "Repeatable and comma-separated, e.g. --skip app --skip logs or --skip app,logs.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: runs against a real Flyte backend")
    config.addinivalue_line("markers", "app: app-serving scenario")
    config.addinivalue_line("markers", "logs: log-retrieval scenario")


# Run light/warming scenarios first, heaviest (app) last. The app deploy (two
# image builds + a serving cold-start) is much slower and flakes if it runs first
# on a cold backend before the image builder / operator warm up.
_ORDER = {
    "test_simple": 0,
    "test_image_builder": 1,
    "test_image_cache": 2,
    "test_io": 3,
    "test_logs": 4,
    "test_trigger": 5,
    "test_reusable": 6,
    "test_app": 7,
}


def pytest_collection_modifyitems(config, items):
    # --skip is registered by this suite's pytest_addoption, which pytest only honors
    # when the suite is collected directly. In a full-tree run (`pytest tests`) this is
    # a deep subdirectory conftest, so its addoption is ignored — the option is absent
    # and this isn't a functional-suite run, so do nothing (and don't reorder the wider
    # collection).
    if not hasattr(config.option, "skip"):
        return
    skip = {tok.strip() for val in config.getoption("--skip") for tok in val.split(",") if tok.strip()}
    for item in items:
        name = item.name.split("[")[0]
        # A token matches any of: a marker on the test, the full test name, or the
        # short name (without the test_ prefix) — so every scenario is skippable, e.g.
        # --skip app, --skip logs, --skip reusable, --skip image_cache.
        names = set(item.keywords) | {name, name.removeprefix("test_")}
        matched = skip & names
        if matched:
            item.add_marker(pytest.mark.skip(reason=f"--skip {','.join(sorted(matched))}"))
    items.sort(key=lambda it: _ORDER.get(it.name.split("[")[0], 3))


@pytest.fixture(scope="session")
def suite_config() -> dict:
    """Resolved connection + naming knobs for the run (see flyte_ops.env_config)."""
    return flyte_ops.env_config()


@pytest.fixture
def flyte_ctx(suite_config) -> dict:
    """(Re-)initialise the flyte client before each test.

    Re-init per test because importing a task module (module-level
    TaskEnvironment()) can reset the client's project/org routing.
    """
    asyncio.run(flyte_ops.init_client(suite_config))
    return suite_config
