"""Registry of benchmark task types.

Everything in the demo — the three agent examples and the benchmark — is
parameterized by a task key, so the same System 1 / System 2 / Flyte machinery
runs customer-support triage, code review and contract review:

    from tasks import TASKS, get_task

    task = get_task("code_review")
    questions = task.structure_questions(task.cases[0])

Add a task type by dropping a module next to this one that exposes a
``TASK = MyTask()`` and listing it below.
"""

from tasks._base import EvalCase, TaskSpec
from tasks.code_review import TASK as CODE_REVIEW_TASK
from tasks.contract import TASK as CONTRACT_TASK
from tasks.support import TASK as SUPPORT_TASK

TASKS: dict[str, TaskSpec] = {
    SUPPORT_TASK.key: SUPPORT_TASK,
    CODE_REVIEW_TASK.key: CODE_REVIEW_TASK,
    CONTRACT_TASK.key: CONTRACT_TASK,
}

TASK_KEYS = list(TASKS)

DEFAULT_TASK = SUPPORT_TASK.key


def get_task(key: str) -> TaskSpec:
    try:
        return TASKS[key]
    except KeyError:
        raise KeyError(f"unknown task {key!r}; available: {TASK_KEYS}") from None


__all__ = ["DEFAULT_TASK", "TASKS", "TASK_KEYS", "EvalCase", "TaskSpec", "get_task"]
