"""Tests for flyteplugins.codegen.generation.prompts."""

from flyte.io import Dir, File

from flyteplugins.codegen.generation.prompts import (
    FILE_EXTENSIONS,
    PACKAGE_MANAGER_MAP,
    TEST_FRAMEWORKS,
    build_enhanced_prompt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_file_extensions(self):
        assert "python" in FILE_EXTENSIONS
        assert FILE_EXTENSIONS["python"] == ".py"

    def test_test_frameworks(self):
        py = TEST_FRAMEWORKS["python"]
        assert py["name"] == "pytest"
        assert "pytest" in py["packages"]
        assert py["command"] == "python -m pytest"

    def test_package_manager_map(self):
        assert "python" in PACKAGE_MANAGER_MAP


# ---------------------------------------------------------------------------
# build_enhanced_prompt
# ---------------------------------------------------------------------------


class TestBuildEnhancedPrompt:
    def test_basic_prompt(self):
        result = build_enhanced_prompt(
            prompt="Sort a list",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs=None,
            outputs=None,
        )
        assert "Language: python" in result
        assert "Sort a list" in result
        # Should always include the script constraint
        assert "solution.py" in result

    def test_with_schema(self):
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema="CREATE TABLE users (id INT)",
            constraints=None,
            data_context=None,
            inputs=None,
            outputs=None,
        )
        assert "Schema:" in result
        assert "CREATE TABLE users" in result

    def test_with_constraints(self):
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema=None,
            constraints=["Must handle nulls", "Must be O(n)"],
            data_context=None,
            inputs=None,
            outputs=None,
        )
        assert "Must handle nulls" in result
        assert "Must be O(n)" in result

    def test_with_data_context(self):
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema=None,
            constraints=None,
            data_context="col1,col2\n1,2\n3,4",
            inputs=None,
            outputs=None,
        )
        assert "Data context:" in result
        assert "col1,col2" in result

    def test_with_inputs_includes_cli_args(self):
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs={"threshold": float, "mode": str},
            outputs=None,
        )
        assert "--threshold" in result
        assert "--mode" in result
        assert "command line arguments" in result

    def test_with_file_input(self):
        result = build_enhanced_prompt(
            prompt="Process file",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs={"csv_data": File},
            outputs=None,
        )
        assert "File and Dir arguments are string paths" in result

    def test_with_dir_input(self):
        result = build_enhanced_prompt(
            prompt="Process directory",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs={"input_dir": Dir},
            outputs=None,
        )
        assert "--input_dir (str): path to dir" in result
        assert "File and Dir arguments are string paths" in result

    def test_with_dir_output(self):
        result = build_enhanced_prompt(
            prompt="Write directory output",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs=None,
            outputs={"output_dir": Dir},
        )
        assert "/var/outputs/output_dir" in result

    def test_with_outputs_includes_output_requirements(self):
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs=None,
            outputs={"result": str, "count": int},
        )
        assert "OUTPUT REQUIREMENTS" in result
        assert "/var/outputs/result" in result
        assert "/var/outputs/count" in result

    def test_data_context_without_inputs(self):
        """When data_context provided but no inputs, should suggest appropriate args."""
        result = build_enhanced_prompt(
            prompt="Process data",
            language="python",
            schema=None,
            constraints=None,
            data_context="x,y\n1,2",
            inputs=None,
            outputs=None,
        )
        assert "appropriate command line arguments to process the data" in result

    def test_no_inputs_no_samples(self):
        """When neither inputs nor samples, should still mention args."""
        result = build_enhanced_prompt(
            prompt="Hello world",
            language="python",
            schema=None,
            constraints=None,
            data_context=None,
            inputs=None,
            outputs=None,
        )
        assert "command line arguments if needed" in result
