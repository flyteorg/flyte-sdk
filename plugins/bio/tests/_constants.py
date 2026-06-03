"""Shared constants for the bio plugin's end-to-end tests."""

from __future__ import annotations

import pathlib

# nf-core/test-datasets, branch ``modules``. Same path root nf-core's
# nf-test scripts reference via ``params.modules_testdata_base_path`` —
# so a path that works in an upstream nf-test file can be pasted in here
# directly.
NF_CORE_RAW_BASE = "https://raw.githubusercontent.com/nf-core/test-datasets/modules/data"

# Cache root, shared across every test script in this directory. Layout
# mirrors the upstream tree so files with the same basename in different
# sub-trees don't collide.
CACHE_DIR = pathlib.Path(__file__).parent / "_fixtures"
