"""Tests for surfacing subtask errors when parent task fails due to unhandled map exceptions.

``BaseRuntimeError`` implements numeric dunder methods (``__radd__``, etc.) that
re-raise the original error.  This means that when ``flyte.map`` returns
exception objects as values (``return_exceptions=True``, the default), user code
that blindly performs arithmetic — e.g. ``sum(results)`` — surfaces the *real*
subtask error instead of a confusing ``TypeError``.
"""

import pytest

import flyte
import flyte.errors
from flyte.errors import BaseRuntimeError


class TestBaseRuntimeErrorArithmetic:
    """Test that BaseRuntimeError re-raises when used in arithmetic operations."""

    def test_radd_reraises(self):
        err = flyte.errors.RuntimeUserError(code="ValueError", message="subtask failed")
        with pytest.raises(flyte.errors.RuntimeUserError, match="subtask failed"):
            0 + err

    def test_add_reraises(self):
        err = flyte.errors.RuntimeUserError(code="ValueError", message="subtask failed")
        with pytest.raises(flyte.errors.RuntimeUserError, match="subtask failed"):
            err + 0

    def test_sum_reraises(self):
        err = flyte.errors.RetriesExhaustedError(code="ValueError", message="task failed for input 1")
        with pytest.raises(flyte.errors.RetriesExhaustedError, match="task failed for input 1"):
            sum([1, 2, err, 4])

    def test_mul_reraises(self):
        err = flyte.errors.RuntimeUserError(code="ValueError", message="subtask failed")
        with pytest.raises(flyte.errors.RuntimeUserError, match="subtask failed"):
            err * 2

    def test_truediv_reraises(self):
        err = flyte.errors.RuntimeUserError(code="ValueError", message="subtask failed")
        with pytest.raises(flyte.errors.RuntimeUserError, match="subtask failed"):
            err / 2

    def test_sub_reraises(self):
        err = flyte.errors.RuntimeUserError(code="ValueError", message="subtask failed")
        with pytest.raises(flyte.errors.RuntimeUserError, match="subtask failed"):
            err - 1


class TestMapSubtaskErrorSurfacing:
    """Test that subtask errors from flyte.map are surfaced correctly."""

    def test_map_with_failing_subtask_surfaces_subtask_error(self):
        """With return_exceptions=True (default), the dunder methods surface the subtask error."""
        env = flyte.TaskEnvironment(name="subtask-error-test")

        @env.task
        def failing_task(x: int) -> int:
            raise ValueError(f"task failed for input {x}")

        @env.task
        def parent_task(inputs: list[int] = [1, 2, 3]) -> float:
            results = list(flyte.map(failing_task, inputs))
            return sum(results) / len(results)

        flyte.init()
        with pytest.raises(BaseRuntimeError) as excinfo:
            flyte.with_runcontext(mode="local").run(parent_task)
        err = excinfo.value
        # The error should be the subtask error, not a TypeError from sum()
        assert "task failed for input" in str(err), (
            f"Expected subtask error to be surfaced, got: code={err.code}, msg={err!s}"
        )
        assert err.code != "TypeError", f"Expected subtask error code, not TypeError. Got: {err.code}"

    def test_map_without_subtask_errors_preserves_original_error(self):
        env = flyte.TaskEnvironment(name="no-subtask-error-test")

        @env.task
        def parent_task() -> float:
            raise RuntimeError("direct parent failure")

        flyte.init()
        with pytest.raises(BaseRuntimeError) as excinfo:
            flyte.with_runcontext(mode="local").run(parent_task)
        assert "direct parent failure" in str(excinfo.value)
