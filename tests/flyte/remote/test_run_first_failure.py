"""
Tests for `Run.first_failure()` and `ActionDetails.error_message`.

These are the observation half of an automated repair loop: after a run fails, an agent (or a
human script) asks which step broke and why, patches the code, and reruns/forks the run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.workflow import run_definition_pb2

from flyte.remote._action import ActionDetails
from flyte.remote._run import Run

RUN_NAME = "run-abc"


def _action_pb2(name: str, parent: str | None = None, task_name: str | None = None) -> run_definition_pb2.Action:
    action = run_definition_pb2.Action(
        id=identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=RUN_NAME),
            name=name,
        ),
        status=run_definition_pb2.ActionStatus(phase=phase_pb2.ACTION_PHASE_FAILED),
    )
    if parent:
        action.metadata.parent = parent
    if task_name:
        action.metadata.task.id.name = task_name
    return action


def _details_pb2(name: str, message: str) -> run_definition_pb2.ActionDetails:
    return run_definition_pb2.ActionDetails(
        id=identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=RUN_NAME),
            name=name,
        ),
        error_info=run_definition_pb2.ErrorInfo(message=message),
    )


def _run(phase=phase_pb2.ACTION_PHASE_FAILED) -> Run:
    return Run(
        run_definition_pb2.Run(
            action=run_definition_pb2.Action(
                id=identifier_pb2.ActionIdentifier(
                    run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=RUN_NAME),
                    name="a0",
                ),
                status=run_definition_pb2.ActionStatus(phase=phase),
            )
        )
    )


def _mock_client(failed_actions, details_by_name):
    client = MagicMock()

    resp = MagicMock()
    resp.actions = failed_actions
    resp.token = ""
    client.run_service.list_actions = AsyncMock(return_value=resp)

    async def get_action_details(request):
        details_resp = MagicMock()
        details_resp.details = details_by_name[request.action_id.name]
        return details_resp

    client.run_service.get_action_details = AsyncMock(side_effect=get_action_details)
    return client


def _first_failure(run, client):
    cfg = MagicMock()
    cfg.org, cfg.project, cfg.domain = "o", "p", "d"
    with (
        patch("flyte.remote._action.ensure_client"),
        patch("flyte.remote._action.get_client", return_value=client),
        patch("flyte.remote._action.get_init_config", return_value=cfg),
    ):
        return run.first_failure()


class TestRunFirstFailure:
    def test_prefers_failed_sub_action_over_root(self):
        # The root action's error just repeats the step's — the step is the useful answer.
        client = _mock_client(
            failed_actions=[
                _action_pb2("a0"),
                _action_pb2("clean-1", parent="a0", task_name="clean_records"),
            ],
            details_by_name={
                "a0": _details_pb2("a0", "child failed"),
                "clean-1": _details_pb2("clean-1", "KeyError: 'price'"),
            },
        )
        failure = _first_failure(_run(), client)
        assert failure is not None
        assert failure.error_message == "KeyError: 'price'"

    def test_falls_back_to_root_when_only_failure(self):
        client = _mock_client(
            failed_actions=[_action_pb2("a0")],
            details_by_name={"a0": _details_pb2("a0", "OOMKilled")},
        )
        failure = _first_failure(_run(), client)
        assert failure is not None
        assert failure.error_message == "OOMKilled"

    def test_none_when_no_action_failed(self):
        client = _mock_client(failed_actions=[], details_by_name={})
        assert _first_failure(_run(phase=phase_pb2.ACTION_PHASE_SUCCEEDED), client) is None


class TestActionDetailsErrorMessage:
    def test_message_of_failed_action(self):
        details = ActionDetails(_details_pb2("a0", "boom"))
        assert details.error_message == "boom"

    def test_empty_when_no_error_info(self):
        details = ActionDetails(
            run_definition_pb2.ActionDetails(
                id=identifier_pb2.ActionIdentifier(
                    run=identifier_pb2.RunIdentifier(org="o", project="p", domain="d", name=RUN_NAME),
                    name="a0",
                ),
            )
        )
        assert details.error_message == ""
