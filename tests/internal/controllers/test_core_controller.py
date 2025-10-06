from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List, Optional, Tuple

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.core import execution_pb2
from flyteidl2.task import task_definition_pb2
from flyteidl2.workflow import (
    queue_service_pb2,
    run_definition_pb2,
    state_service_pb2,
)

from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import Controller
from flyte._internal.controllers.remote._service_protocol import (
    ClientSet,
    QueueService,
    StateService,
)
from flyte._logging import logger
from flyte._utils import run_coros


class DummyService(QueueService, StateService, ClientSet):
    """
    Dummy service that implements the QueueService and StateService interfaces. This is used for testing
    purposes only.
    This service stores the queue and state in memory.
    The StateService waits for a few seconds and then calls a callback, which is a coroutine, allowing
    mutation of the state (phase) of the task. This is then returned in the watch API. When the watch API
    is invoked the first time, it returns all known tasks as runUpdate.
    """

    def __init__(
        self,
        phases: Dict[
            str,
            List[
                Tuple[
                    run_definition_pb2.Phase,
                    Optional[execution_pb2.ExecutionError],
                    Optional[str],
                ]
            ],
        ],
        queue: Dict[str, queue_service_pb2.EnqueueActionRequest] | None = None,
    ):
        """
        Initialize the DummyService with a dictionary of phases for each run_id.name.
        :param phases: A dictionary mapping run_id.name to a list of phases.
        """
        self._queue: Dict[str, queue_service_pb2.EnqueueActionRequest] = queue or {}
        self._phases: Dict[
            str,
            List[
                Tuple[
                    run_definition_pb2.Phase,
                    Optional[execution_pb2.ExecutionError],
                    Optional[str],
                ]
            ],
        ] = phases
        self._lock = asyncio.Lock()

    @classmethod
    async def create(
        cls,
        phases: Dict[
            str,
            List[
                Tuple[
                    run_definition_pb2.Phase,
                    Optional[execution_pb2.ExecutionError],
                    Optional[str],
                ]
            ],
        ],
        queue: Dict[str, queue_service_pb2.EnqueueActionRequest] | None = None,
    ) -> DummyService:
        return cls(phases, queue)

    @property
    def queue_service(self) -> QueueService:
        return self

    @property
    def state_service(self) -> StateService:
        return self

    async def Watch(
        self, req: state_service_pb2.WatchRequest, **kwargs
    ) -> AsyncIterator[state_service_pb2.WatchResponse]:
        """Simulate watching for state updates."""
        sentinel = False
        while True:
            async with self._lock:
                queue = self._queue.copy()

            for run_name, queue_req in queue.items():
                if self._phases.get(run_name):
                    # Update the phase sequentially
                    phase, error, outputs_uri = self._phases[run_name].pop(0)
                    yield state_service_pb2.WatchResponse(
                        action_update=state_service_pb2.ActionUpdate(
                            action_id=queue_req.action_id,
                            phase=phase,
                            # task_id=queue[run_name].task_id,  # TODO: do we removed task_id from proto?
                            error=error,
                            output_uri=outputs_uri,
                        )
                    )
            # We mimic sentinel
            if not sentinel:
                logger.info("DummyService: Sending sentinel update...")
                yield state_service_pb2.WatchResponse(
                    control_message=state_service_pb2.ControlMessage(
                        sentinel=True,
                    )
                )
            sentinel = True
            await asyncio.sleep(0.1)

    async def EnqueueAction(
        self,
        req: queue_service_pb2.EnqueueActionRequest,
        **kwargs,
    ) -> queue_service_pb2.EnqueueActionResponse:
        """Enqueue a task."""
        print(f"Dummy service enqueuing task: {req.action_id.name}")
        if req.action_id.name in self._queue:
            pytest.fail("Task already in queue")
        self._queue[req.action_id.name] = req
        print("Enqueueing task:", req.action_id.name)
        return queue_service_pb2.EnqueueActionResponse()

    async def AbortQueuedAction(
        self,
        req: queue_service_pb2.AbortQueuedActionRequest,
        **kwargs,
    ) -> queue_service_pb2.AbortQueuedActionResponse:
        """Abort a queued task."""
        print(f"Dummy service aborting task: {req.action_id.name}")
        if req.action_id.name in self._queue:
            del self._queue[req.action_id.name]
        return queue_service_pb2.AbortQueuedActionResponse()


@pytest.mark.asyncio
async def test_basic_end_to_end_one_task():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    service = DummyService.create(
        phases={
            "subrun-1": [
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
                (
                    run_definition_pb2.Phase.PHASE_SUCCEEDED,
                    None,
                    "s3://bucket/run-id/sub-action/1",
                ),
            ],
        }
    )
    input_node = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="output_uri",
    )
    c = Controller(client_coro=service, workers=2, max_system_retries=2)

    async def _run():
        final_node = await c.submit_action(input_node)
        assert final_node.started
        assert final_node.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED, (
            f"Expected phase to be PHASE_SUCCEEDED, found {run_definition_pb2.Phase.Name(final_node.phase)},"
            f" for {final_node.action_id.name}"
        )
        assert final_node.realized_outputs_uri == "s3://bucket/run-id/sub-action/1"
        await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
        await c.stop()

    await run_coros(_run(), c.watch_for_errors())


@pytest.mark.asyncio
async def test_basic_three_tasks():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action1/1",
            ),
        ],
        "subrun-2": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action2/1",
            ),
        ],
        "subrun-3": [
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/subrun-3/4",
            ),
        ],
    }
    service = DummyService.create(phases=phases)

    nodes = []
    for k, v in phases.items():
        nodes.append(
            Action(
                action_id=identifier_pb2.ActionIdentifier(
                    name=k,
                    run=run_id,
                ),
                parent_action_name=parent_action_name,
                task=task_definition_pb2.TaskSpec(),
                inputs_uri="input_uri",
                run_output_base="my_run_base",
            )
        )
    c = Controller(client_coro=service, workers=2, max_system_retries=2)
    futs = [c.submit_action(n) for n in nodes]
    final_nodes = await asyncio.gather(*futs)
    for final_node in final_nodes:
        assert final_node.started
        assert final_node.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED
        assert final_node.realized_outputs_uri

    await c._finalize_parent_action(run_id, parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_recover():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action1/1",
            ),
        ],
        "subrun-2": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action2/5",
            ),
        ],
        "subrun-3": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action3/1",
            ),
        ],
    }
    service = DummyService.create(
        phases=phases,
        queue={
            "subrun-1": queue_service_pb2.EnqueueActionRequest(
                action_id=identifier_pb2.ActionIdentifier(
                    name="subrun-1",
                    run=run_id,
                ),
            ),
            "subrun-2": queue_service_pb2.EnqueueActionRequest(
                action_id=identifier_pb2.ActionIdentifier(
                    name="subrun-2",
                    run=run_id,
                ),
            ),
            "subrun-3": queue_service_pb2.EnqueueActionRequest(
                action_id=identifier_pb2.ActionIdentifier(
                    name="subrun-3",
                    run=run_id,
                ),
            ),
        },
    )

    nodes = []
    for k, v in phases.items():
        nodes.append(
            Action(
                action_id=identifier_pb2.ActionIdentifier(
                    name=k,
                    run=run_id,
                ),
                parent_action_name=parent_action_name,
                task=task_definition_pb2.TaskSpec(),
                inputs_uri="input_uri",
                run_output_base="run-base",
            )
        )
    c = Controller(client_coro=service, workers=2, max_system_retries=2)
    futs = [c.submit_action(n) for n in nodes]
    final_nodes = await asyncio.gather(*futs)
    for final_node in final_nodes:
        assert final_node.started
        assert final_node.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED
        assert final_node.realized_outputs_uri

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_multiple_submits_sequential():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action1/1",
            ),
        ],
        "subrun-2": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action2/1",
            ),
        ],
    }
    service = DummyService.create(phases=phases)

    c = Controller(client_coro=service, workers=2, max_system_retries=2)

    # Submit first node
    node1 = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )
    final_node1 = await c.submit_action(node1)
    assert final_node1.started
    assert final_node1.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED

    # Submit second node
    node2 = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-2",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )
    final_node2 = await c.submit_action(node2)
    assert final_node2.started
    assert final_node2.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED
    assert final_node2.realized_outputs_uri

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_submit_with_failure_phase():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_FAILED,
                execution_pb2.ExecutionError(message="Task failed due to timeout"),
                None,
            ),
        ],
        "subrun-2": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action2/1",
            ),
        ],
    }
    service = DummyService.create(phases=phases)

    c = Controller(client_coro=service, workers=2, max_system_retries=2)

    # Submit first node (expected to fail)
    node1 = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )
    final_node1 = await c.submit_action(node1)
    assert final_node1.started
    assert final_node1.phase == run_definition_pb2.Phase.PHASE_FAILED
    assert final_node1.err.message == "Task failed due to timeout"

    # Submit second node (expected to succeed)
    node2 = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-2",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )
    final_node2 = await c.submit_action(node2)
    assert final_node2.started
    assert final_node2.phase == run_definition_pb2.Phase.PHASE_SUCCEEDED

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_multiple_submits_with_mixed_phases():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action1/1",
            ),
        ],
        "subrun-2": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_FAILED,
                execution_pb2.ExecutionError(message="Task failed due to error"),
                None,
            ),
        ],
        "subrun-3": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (
                run_definition_pb2.Phase.PHASE_SUCCEEDED,
                None,
                "s3://bucket/run-id/sub-action3/1",
            ),
        ],
    }
    service = DummyService.create(phases=phases)

    nodes = []
    for k, v in phases.items():
        nodes.append(
            Action(
                action_id=identifier_pb2.ActionIdentifier(
                    name=k,
                    run=run_id,
                ),
                parent_action_name=parent_action_name,
                task=task_definition_pb2.TaskSpec(),
                inputs_uri="input_uri",
                run_output_base="output-base",
            )
        )

    c = Controller(client_coro=service, workers=2, max_system_retries=2)

    # Submit all nodes
    futs = [c.submit_action(n) for n in nodes]
    final_nodes = await asyncio.gather(*futs)

    # Validate results
    assert final_nodes[0].started
    assert final_nodes[0].phase == run_definition_pb2.Phase.PHASE_SUCCEEDED

    assert final_nodes[1].started
    assert final_nodes[1].phase == run_definition_pb2.Phase.PHASE_FAILED
    assert final_nodes[1].err.message == "Task failed due to error"

    assert final_nodes[2].started
    assert final_nodes[2].phase == run_definition_pb2.Phase.PHASE_SUCCEEDED

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_submit_with_failure_phase_no_err():
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_FAILED, None, None),
        ],
    }
    service = DummyService.create(phases=phases)

    c = Controller(client_coro=service, workers=2, max_system_retries=2)

    # Submit first node (expected to fail)
    node1 = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )
    final_node1 = await c.submit_action(node1)
    assert final_node1.started
    assert final_node1.phase == run_definition_pb2.Phase.PHASE_FAILED
    assert final_node1.err is None

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()


@pytest.mark.asyncio
async def test_cancel_queued_action():
    """Test that cancelling a queued action properly invokes AbortQueuedAction"""
    parent_action_name = "parent_action"
    run_id = identifier_pb2.RunIdentifier(
        name="root_run",
    )
    phases = {
        "subrun-1": [
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
            (run_definition_pb2.Phase.PHASE_RUNNING, None, None),
        ],
    }

    # Create service and track calls to AbortQueuedAction
    abort_called = False

    class TrackingService(DummyService):
        async def AbortQueuedAction(self, req, **kwargs):
            nonlocal abort_called
            abort_called = True
            return await super().AbortQueuedAction(req, **kwargs)

    async def create_tracking_service():
        return TrackingService(phases=phases)

    c = Controller(client_coro=create_tracking_service(), workers=2, max_system_retries=2)

    # Create and enqueue an action
    action = Action(
        action_id=identifier_pb2.ActionIdentifier(
            name="subrun-1",
            run=run_id,
        ),
        parent_action_name=parent_action_name,
        task=task_definition_pb2.TaskSpec(),
        inputs_uri="input_uri",
        run_output_base="run-base",
    )

    # Start submitting the action but don't wait for completion
    submit_task = asyncio.create_task(c.submit_action(action))

    # Give it time to enqueue and start
    await asyncio.sleep(0.3)

    # Mark the action as started (simulating that it has been enqueued)
    action.mark_started()

    # Cancel the action
    await c.cancel_action(action)

    # Verify that AbortQueuedAction was called
    assert abort_called, "AbortQueuedAction should have been called"

    # Clean up
    submit_task.cancel()
    try:
        await submit_task
    except asyncio.CancelledError:
        pass

    await c._finalize_parent_action(run_id=run_id, parent_action_name=parent_action_name)
    await c.stop()
