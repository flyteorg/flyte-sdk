"""Tests for pure utility functions in flyteplugins.codegen.generation.llm."""

import pytest

from flyteplugins.codegen.core.types import TestFunctionPatch
from flyteplugins.codegen.generation.llm import (
    _strip_parametrize_suffix,
    apply_test_patches,
    extract_error_messages_from_pytest,
    filter_stdlib,
    strip_code_fences,
)


# ---------------------------------------------------------------------------
# strip_code_fences
# ---------------------------------------------------------------------------


class TestStripCodeFences:
    def test_with_language_tag(self):
        code = "```python\nprint('hello')\n```"
        assert strip_code_fences(code) == "print('hello')"

    def test_without_language_tag(self):
        code = "```\nprint('hello')\n```"
        assert strip_code_fences(code) == "print('hello')"

    def test_no_fences(self):
        code = "print('hello')"
        assert strip_code_fences(code) == "print('hello')"

    def test_nested_fences_not_stripped(self):
        """Only outer fences should be stripped."""
        code = "```python\ncode = '```nested```'\n```"
        result = strip_code_fences(code)
        assert "```nested```" in result

    def test_whitespace_stripped(self):
        code = "  \n```python\nprint('hello')\n```  \n"
        assert strip_code_fences(code) == "print('hello')"

    def test_multiline_code(self):
        code = "```python\nimport os\nprint(os.getcwd())\n```"
        result = strip_code_fences(code)
        assert "import os" in result
        assert "print(os.getcwd())" in result


# ---------------------------------------------------------------------------
# filter_stdlib
# ---------------------------------------------------------------------------


class TestFilterStdlib:
    def test_filters_stdlib_modules(self):
        packages = ["os", "sys", "json", "numpy", "pandas"]
        result = filter_stdlib(packages)
        assert "os" not in result
        assert "sys" not in result
        assert "json" not in result
        assert "numpy" in result
        assert "pandas" in result

    def test_empty_list(self):
        assert filter_stdlib([]) == []

    def test_all_stdlib(self):
        assert filter_stdlib(["os", "sys", "json"]) == []

    def test_all_third_party(self):
        result = filter_stdlib(["numpy", "pandas", "scikit-learn"])
        assert len(result) == 3

    def test_dotted_module_names(self):
        """os.path should be filtered because its root is 'os'."""
        result = filter_stdlib(["os.path", "numpy.linalg"])
        assert "os.path" not in result
        assert "numpy.linalg" in result


# ---------------------------------------------------------------------------
# extract_error_messages_from_pytest
# ---------------------------------------------------------------------------


class TestExtractErrorMessagesFromPytest:
    def test_single_failure(self):
        output = """\
============================= FAILURES =============================
_____________________________ test_add _____________________________
    def test_add():
>       assert add(1, 2) == 4
E       AssertionError: assert 3 == 4
E       RecursionError: maximum recursion depth exceeded

============================= short test summary info =============================
"""
        result = extract_error_messages_from_pytest(output)
        assert "test_add" in result
        assert "RecursionError" in result["test_add"]

    def test_multiple_failures(self):
        output = """\
_____________________________ test_add _____________________________
E       AssertionError: assert 3 == 4
_____________________________ test_subtract _____________________________
E       TypeError: unsupported operand type
"""
        result = extract_error_messages_from_pytest(output)
        assert "test_add" in result
        assert "test_subtract" in result

    def test_parametrized_test_name(self):
        output = """\
_____________________________ test_calc[0-True] _____________________________
E       AssertionError: Failed assertion
"""
        result = extract_error_messages_from_pytest(output)
        # Parametrize suffix should be stripped
        assert "test_calc" in result

    def test_no_failures(self):
        output = "============================= 3 passed ============================="
        result = extract_error_messages_from_pytest(output)
        assert result == {}

    def test_error_line_must_contain_error_or_exception(self):
        """Lines starting with 'E   ' but without Error/Exception/Failed are skipped."""
        output = """\
_____________________________ test_foo _____________________________
E       where 3 = add(1, 2)
"""
        result = extract_error_messages_from_pytest(output)
        # "where 3 = add(1, 2)" doesn't contain Error/Exception/Failed
        assert result == {}


# ---------------------------------------------------------------------------
# _strip_parametrize_suffix
# ---------------------------------------------------------------------------


class TestStripParametrizeSuffix:
    def test_strips_suffix(self):
        assert _strip_parametrize_suffix("test_foo[0-True]") == "test_foo"

    def test_no_suffix(self):
        assert _strip_parametrize_suffix("test_foo") == "test_foo"

    def test_complex_suffix(self):
        assert _strip_parametrize_suffix("test_calc[530.00-33-3]") == "test_calc"


# ---------------------------------------------------------------------------
# apply_test_patches
# ---------------------------------------------------------------------------


class TestApplyTestPatches:
    def test_replace_function_body(self):
        original = """\
import pytest

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
"""
        patches = [
            TestFunctionPatch(
                test_name="test_add",
                fixed_code="def test_add():\n    assert add(1, 2) == 4",
            )
        ]
        result = apply_test_patches(original, patches)
        assert "assert add(1, 2) == 4" in result
        # Other function should be preserved
        assert "def test_subtract():" in result
        assert "assert subtract(5, 3) == 2" in result

    def test_preserves_imports(self):
        original = """\
import pytest
from solution import add

def test_add():
    assert add(1, 2) == 3
"""
        patches = [
            TestFunctionPatch(
                test_name="test_add",
                fixed_code="def test_add():\n    assert add(1, 2) == 4",
            )
        ]
        result = apply_test_patches(original, patches)
        assert "import pytest" in result
        assert "from solution import add" in result

    def test_missing_function_name_ignored(self):
        original = """\
def test_add():
    assert add(1, 2) == 3
"""
        patches = [
            TestFunctionPatch(
                test_name="test_nonexistent",
                fixed_code="def test_nonexistent():\n    pass",
            )
        ]
        result = apply_test_patches(original, patches)
        # Original should be unchanged since patch target doesn't exist
        assert "assert add(1, 2) == 3" in result
