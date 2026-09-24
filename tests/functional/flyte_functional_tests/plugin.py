"""pytest plugin for the flyte functional suite.

Registered as a ``pytest11`` entry point (see pyproject) so its option, markers, and
ordering are available whenever the installed package is present — including under
``pytest --pyargs flyte_functional_tests``. A bare in-package ``conftest`` can't
guarantee this: with ``--pyargs`` its ``pytest_addoption`` may run after option parsing,
so ``--skip`` would be rejected. Kept deliberately import-light (no scenario/task
imports) so loading the plugin never drags fastapi etc. into an unrelated pytest run.
"""

from __future__ import annotations

import pytest

# Run lightest scenarios first, the app deploy (two image builds + a serving cold-start)
# last — it is much slower and flakes if it runs first on a cold backend.
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

# This package's import root, so the hooks only touch the suite's own items (the plugin
# is global once installed; it must not reorder or skip an unrelated repo's tests).
_PKG = __name__.rsplit(".", 1)[0]


def _is_suite_item(item) -> bool:  # type: ignore[no-untyped-def]
    name = getattr(getattr(item, "module", None), "__name__", "") or ""
    return name == _PKG or name.startswith(_PKG + ".")


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


def pytest_collection_modifyitems(config, items):
    suite = [it for it in items if _is_suite_item(it)]
    if not suite:
        return  # no suite tests collected → leave an unrelated collection untouched
    skip = {tok.strip() for val in config.getoption("--skip") for tok in val.split(",") if tok.strip()}
    for item in suite:
        name = item.name.split("[")[0]
        # A token matches any of: a marker on the test, the full test name, or the short
        # name (without test_), so every scenario is skippable: --skip app, --skip logs, …
        names = set(item.keywords) | {name, name.removeprefix("test_")}
        matched = skip & names
        if matched:
            item.add_marker(pytest.mark.skip(reason=f"--skip {','.join(sorted(matched))}"))
    # Only reorder a standalone suite run; leave a mixed full-tree collection alone.
    if len(suite) == len(items):
        items.sort(key=lambda it: _ORDER.get(it.name.split("[")[0], 3))
