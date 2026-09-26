"""The artifact source a task-side Artifact.create sends carries the run's scope.

The artifact service requires org/project/domain on the source's RunIdentifier;
an empty scope failed every publish from a task with
``spec.source.task_action.action.run.org: must be at least 1 characters``.
"""

from flyteidl2.artifact import artifact_pb2

import flyte
from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.remote._artifact import _current_task_source


def _task_context(action: ActionID) -> TaskContext:
    return TaskContext(
        action=action,
        version="v1",
        raw_data_path=RawDataPath(path="s3://bucket/raw"),
        output_path="s3://bucket/out",
        run_base_dir="s3://bucket/base",
        report=flyte.report.Report(name=action.name),
    )


def test_outside_a_task_there_is_no_source():
    assert _current_task_source() is None


def test_the_running_actions_scope_is_sent():
    action = ActionID(name="a1", run_name="r1", org="demo", project="haytham", domain="development")
    with internal_ctx().replace_task_context(_task_context(action)):
        src = _current_task_source()
    run = src.task_action.action.run
    assert (run.org, run.project, run.domain, run.name) == ("demo", "haytham", "development", "r1")
    assert src.task_action.action.name == "a1"


def test_the_artifacts_scope_fills_what_the_action_lacks():
    action = ActionID(name="a1", run_name="r1")
    scope = artifact_pb2.ArtifactName(org="demo", project="p", domain="d", name="art")
    with internal_ctx().replace_task_context(_task_context(action)):
        src = _current_task_source(scope)
    run = src.task_action.action.run
    assert (run.org, run.project, run.domain) == ("demo", "p", "d")


def test_the_real_task_action_is_used_inside_a_trace():
    task = ActionID(name="task", run_name="r1", org="o", project="p", domain="d")
    traced = ActionID(name="trace-step", run_name="r1")
    tctx = _task_context(task).replace(action=traced)
    assert tctx.task_action == task
    with internal_ctx().replace_task_context(tctx):
        src = _current_task_source()
    assert src.task_action.action.name == "task"
    assert src.task_action.action.run.org == "o"


def test_a_task_publishing_into_another_project_sends_no_source():
    """The service only accepts a source in the artifact's own scope."""
    action = ActionID(name="a1", run_name="r1", org="demo", project="haytham", domain="development")
    elsewhere = artifact_pb2.ArtifactName(org="demo", project="shared", domain="development", name="art")
    same = artifact_pb2.ArtifactName(org="demo", project="haytham", domain="development", name="art")
    with internal_ctx().replace_task_context(_task_context(action)):
        assert _current_task_source(elsewhere) is None
        assert _current_task_source(same) is not None
