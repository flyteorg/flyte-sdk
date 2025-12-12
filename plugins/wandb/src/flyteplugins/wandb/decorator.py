from contextlib import contextmanager
from dataclasses import asdict
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar, cast

import wandb

import flyte
from flyte import TaskTemplate

from .context import get_wandb_context
from .link import Wandb as WandbLink

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def _wandb_run():
    """Context manager for wandb run lifecycle."""
    context_config = get_wandb_context()

    # Convert to wandb.init kwargs
    if context_config:
        config_dict = asdict(context_config)
        # Extract kwargs separately
        extra_kwargs = config_dict.pop("kwargs", None) or {}
        # Merge: explicit fields + extra kwargs (extra_kwargs has lower priority)
        init_kwargs = {
            **extra_kwargs,
            **{k: v for k, v in config_dict.items() if v is not None},
        }
    else:
        init_kwargs = {}

    # Auto-generate ID if not provided
    if "id" not in init_kwargs or init_kwargs["id"] is None:
        init_kwargs["id"] = (
            f"{flyte.ctx().action.run_name}:{flyte.ctx().action.name}"  # TODO: replica index?
        )

    run = wandb.init(**init_kwargs)
    try:
        yield run
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise


def wandb_init(_func: Optional[F] = None) -> F:
    """
    Decorator to automatically initialize wandb for a Flyte task.

    This decorator:
    1. Initializes a wandb run before task execution
    2. Auto-generates unique run ID from Flyte action context (if not provided)
    3. Makes the run available via wandb.run or get_wandb_run()
    4. Automatically finishes the run after task completion
    5. Automatically attaches wandb link to task (pulls config from context)
    """

    def decorator(func: F) -> F:
        if not isinstance(func, TaskTemplate):
            raise TypeError("@wandb_init must be applied to a Flyte task.")

        # Attach wandb link that pulls from context
        func.link = WandbLink()

        # Wrap the task's execute method with wandb_run
        original_execute = func.execute

        if iscoroutinefunction(original_execute):

            async def wrapped_execute(*args, **kwargs):
                with _wandb_run():
                    return await original_execute(*args, **kwargs)

            func.execute = wrapped_execute
        else:

            def wrapped_execute(*args, **kwargs):
                with _wandb_run():
                    return original_execute(*args, **kwargs)

            func.execute = wrapped_execute

        return cast(F, func)

    return decorator(_func)


def get_wandb_run():
    """
    Get the current wandb run.

    This is a convenience wrapper around wandb.run that provides
    a consistent API for accessing the current run.
    """
    return wandb.run
