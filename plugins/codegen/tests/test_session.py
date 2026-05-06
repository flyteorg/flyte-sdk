"""Tests for _CodeGenSession internals in flyteplugins.codegen.auto_coder_agent."""

from flyteplugins.codegen.core.types import CodePlan, CodeSolution

# ---------------------------------------------------------------------------
# _CodeGenSession helpers
# ---------------------------------------------------------------------------


def _make_session(**overrides):
    """Create a _CodeGenSession with sensible defaults."""
    from flyteplugins.codegen.auto_coder_agent import AutoCoderAgent, _CodeGenSession

    agent = AutoCoderAgent(model="test-model")

    defaults = {
        "agent": agent,
        "language": "python",
        "prompt": "test prompt",
        "schema": None,
        "constraints": None,
        "inputs": None,
        "outputs": None,
        "base_packages": ["pytest"],
        "extracted_data_context": None,
        "sample_files": None,
        "schemas_as_code": {},
        "base_messages": [{"role": "user", "content": "test"}],
        "plan": CodePlan(description="test plan", approach="test approach"),
        "skip_tests": False,
        "initial_input_tokens": 0,
        "initial_output_tokens": 0,
    }
    defaults.update(overrides)
    return _CodeGenSession(**defaults)


class TestComputeImageName:
    def test_deterministic(self):
        session = _make_session()
        name1 = session._compute_image_name(["pytest"], [])
        name2 = session._compute_image_name(["pytest"], [])
        assert name1 == name2

    def test_different_packages_different_name(self):
        session = _make_session()
        name1 = session._compute_image_name(["pytest"], [])
        name2 = session._compute_image_name(["pytest", "numpy"], [])
        assert name1 != name2

    def test_different_system_packages_different_name(self):
        session = _make_session()
        name1 = session._compute_image_name(["pytest"], [])
        name2 = session._compute_image_name(["pytest"], ["gcc"])
        assert name1 != name2

    def test_name_format(self):
        session = _make_session()
        name = session._compute_image_name(["pytest"], [])
        assert name.startswith("auto-coder-agent-python-")

    def test_order_independent(self):
        """Packages are sorted before hashing, so order shouldn't matter."""
        session = _make_session()
        name1 = session._compute_image_name(["numpy", "pandas"], [])
        name2 = session._compute_image_name(["pandas", "numpy"], [])
        assert name1 == name2


class TestTrackTokens:
    def test_accumulates(self):
        session = _make_session(initial_input_tokens=10, initial_output_tokens=20)
        assert session.total_input_tokens == 10
        assert session.total_output_tokens == 20

        session._track_tokens(5, 10)
        assert session.total_input_tokens == 15
        assert session.total_output_tokens == 30

        session._track_tokens(3, 7)
        assert session.total_input_tokens == 18
        assert session.total_output_tokens == 37


class TestMakeResult:
    def test_populates_all_fields(self):
        session = _make_session(initial_input_tokens=50, initial_output_tokens=100)
        session.solution = CodeSolution(language="python", code="print('hi')")
        session.tests = "def test_hi(): pass"
        session.detected_packages = ["numpy"]
        session.detected_system_packages = ["gcc"]
        session.current_image = "my-img:latest"

        result = session._make_result(
            success=True,
            test_output="all passed",
            exit_code=0,
            attempt=3,
        )

        assert result.success is True
        assert result.output == "all passed"
        assert result.exit_code == 0
        assert result.attempts == 3
        assert result.solution.code == "print('hi')"
        assert result.tests == "def test_hi(): pass"
        assert result.detected_packages == ["numpy"]
        assert result.detected_system_packages == ["gcc"]
        assert result.image == "my-img:latest"
        assert result.total_input_tokens == 50
        assert result.total_output_tokens == 100
        assert result.plan is not None

    def test_with_error(self):
        session = _make_session()
        session.solution = CodeSolution()
        result = session._make_result(
            success=False,
            test_output="FAILED",
            exit_code=1,
            attempt=1,
            error="something broke",
        )
        assert result.success is False
        assert result.error == "something broke"

    def test_no_solution_uses_default(self):
        session = _make_session()
        session.solution = None
        result = session._make_result(
            success=False,
            test_output="",
            exit_code=-1,
            attempt=1,
        )
        assert result.solution is not None
        assert result.solution.code == ""
