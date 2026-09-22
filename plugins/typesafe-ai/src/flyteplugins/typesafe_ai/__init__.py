"""Run TypeSafe's System One model (Jev) inside durable Flyte tasks.

Jev answers typed questions in parallel instead of generating text, and returns
calibrated confidence with every answer. This plugin gives those answers a shape
that crosses a Flyte task boundary — `Choice`, `Score` and `Noul` — and a way to
ask a whole battery of them in a single request.

```python
import enum
from dataclasses import dataclass, field

import flyte
from flyteplugins.typesafe_ai import Choice, Noul, Score, ask

env = flyte.TaskEnvironment(
    "triage",
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
)


class Intent(enum.Enum):
    '''Which intent best fits the ticket?'''

    REFUND = "refund"
    '''they want money back'''
    DELIVERY = "delivery"
    '''they are asking where their order is'''


class Severity(enum.IntEnum):
    '''How badly is this customer affected?'''

    NONE = 0
    '''no impact; a question or a comment'''
    MINOR = 1
    '''inconvenient, but they can carry on'''
    SERIOUS = 2
    '''they are blocked'''


@dataclass
class Triage:
    # The enums above document themselves, so these fields need no metadata at all.
    intent: Choice[Intent]
    severity: Score[Severity]
    # A Noul has no vocabulary to document itself with, so it needs both.
    hostile: Noul = field(
        metadata={"question": "Is the customer hostile?", "criteria": {"true": "insults or threats", "false": "civil"}}
    )


@env.task
async def triage(ticket: str) -> Triage:
    return await ask(Triage, {"ticket": ticket})
```
"""

import functools

from flyte.types import TypeEngine

from ._ask import BatteryError, ask, ask_with_info, compile_questions, thresholds
from ._client import API_KEY_ENV, MissingAPIKey, client
from ._types import CRITERIA_KEY, QUESTION_KEY, THRESHOLD_KEY, CallInfo, Choice, Noul, Score


@functools.lru_cache(maxsize=None)
def register_typesafe_ai_types():
    """Register Choice, Score and Noul with the Flyte type engine.

    Called automatically via the `flyte.plugins.types` entry point when
    `flyte.init()` runs with `load_plugin_type_transformers=True` (the default).

    The three answer types are plain dataclasses, so they reuse the existing
    `DataclassTransformer` rather than introducing one of their own. Registering
    them anyway makes the resolution explicit: a parameterized `Choice[Intent]`
    is matched through its origin instead of falling through to the type engine's
    last-resort dataclass branch.
    """
    from flyte.types._type_engine import DataclassTransformer

    transformer = DataclassTransformer()
    for answer_type in (Choice, Score, Noul):
        TypeEngine.register_additional_type(transformer, answer_type)


# Also register at import time, so the types work without flyte.init() -- a unit
# test that only round-trips a battery never calls it.
register_typesafe_ai_types()

__all__ = [
    "API_KEY_ENV",
    "CRITERIA_KEY",
    "QUESTION_KEY",
    "THRESHOLD_KEY",
    "BatteryError",
    "CallInfo",
    "Choice",
    "MissingAPIKey",
    "Noul",
    "Score",
    "ask",
    "ask_with_info",
    "client",
    "compile_questions",
    "register_typesafe_ai_types",
    "thresholds",
]
