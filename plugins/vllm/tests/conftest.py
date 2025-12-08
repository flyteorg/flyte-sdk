"""Pytest configuration for vLLM plugin tests."""

import sys
from pathlib import Path

# Add the plugin src directory to the path for testing
plugin_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(plugin_src))

