"""Shared constants for the bio plugin's end-to-end tests."""

from __future__ import annotations

import pathlib

# Public fixture base used by the bio plugin tests.
FIXTURE_BASE_URL = "https://raw.githubusercontent.com/nf-core/test-datasets/modules/data"

# Cache root, shared across every test script in this directory.
CACHE_DIR = pathlib.Path(__file__).parent / "_fixtures"
