"""Shared fixtures for codegen plugin tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Patch missing flyte.sandbox.ImageConfig before any plugin module is imported.
# ImageConfig comes from a separate PR that may not be merged locally yet.
# ---------------------------------------------------------------------------
import flyte.sandbox as _sandbox_mod
import pytest

if not hasattr(_sandbox_mod, "ImageConfig"):
    _sandbox_mod.ImageConfig = type("ImageConfig", (), {})

# Ensure the plugin source is importable
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Mock LLM response helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    """Build a minimal object that looks like a litellm response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.fixture()
def mock_llm_response():
    """Factory fixture: call with content string to get a mock LLM response."""
    return _make_llm_response


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_dataframe():
    import pandas as pd

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [88.5, 92.0, 75.3, 98.1, 81.0],
            "passed": [True, True, False, True, True],
        }
    )


@pytest.fixture()
def sample_code_gen_eval_result():
    from flyteplugins.codegen.core.types import CodeGenEvalResult, CodeSolution

    return CodeGenEvalResult(
        solution=CodeSolution(language="python", code="print('hello')"),
        success=True,
        output="All tests passed",
        exit_code=0,
        image="my-image:latest",
    )
