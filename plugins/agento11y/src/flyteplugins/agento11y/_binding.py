"""Giving agento11y the Flyte identifiers it already has slots for.

agento11y groups its records by conversation, and tracks which agent and which version
produced them. Flyte knows all three, and nobody should have to restate them by hand:

    conversation id  <-  the run
    agent name       <-  the task
    agent version    <-  the task version

    The agent name binding assumes one agent per task. A task driving several agents should
    set ``bind_agent_name=False`` so each keeps the name its framework gave it.

Binding them means a Flyte run shows up in Grafana as one conversation, and a redeploy shows
up as a new agent version, so the before-and-after of a prompt change is directly comparable.

This is a :class:`flyte._observe.Observer` which is what makes it work at all: the values
have to be set for the duration of the task body, before any generation starts, and that is
exactly the window a task span covers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING

from flyte._logging import logger

from agento11y.context import with_agent_name, with_agent_version, with_conversation_id

if TYPE_CHECKING:
    from flyte._observe import Recorder, StepInfo, TaskInfo

__all__ = ["FlyteIdentityBinding"]


class FlyteIdentityBinding:
    """Binds Flyte's run, task, and version onto agento11y's context for each task."""

    def __init__(self, *, bind_conversation: bool = True, bind_agent_name: bool = True):
        # Left switchable because an agent that models a real user conversation spanning
        # several runs wants its own id, not one per run.
        self._bind_conversation = bind_conversation
        # Also switchable, because the task is only the right unit of agent identity when it
        # runs one agent. A task that drives several — a planner and a worker, say — would
        # report them all under the task name and lose the distinction between them. Turning
        # this off lets each framework name its own agents instead.
        self._bind_agent_name = bind_agent_name

    @contextlib.contextmanager
    def task_span(self, info: "TaskInfo", recorder: "Recorder") -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            try:
                run_name = info.action.run_name or info.action.name
                if self._bind_conversation and run_name:
                    stack.enter_context(with_conversation_id(run_name))
                if self._bind_agent_name and info.name:
                    stack.enter_context(with_agent_name(info.name))
                if info.version:
                    stack.enter_context(with_agent_version(info.version))
            except Exception:
                # Losing the binding costs grouping in the UI, not the run.
                logger.debug("Could not bind Flyte identity onto agento11y context", exc_info=True)
            yield

    @contextlib.contextmanager
    def step_span(self, info: "StepInfo", recorder: "Recorder") -> Generator[None, None, None]:
        # Trace steps inherit the task's binding; there is nothing per-step to set.
        yield
