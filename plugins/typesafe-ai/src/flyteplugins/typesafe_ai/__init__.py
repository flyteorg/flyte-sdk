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

from ._ask import BatteryError, ask, ask_with_info, compile_questions
from ._client import API_KEY_ENV, MissingAPIKey, client
from ._types import CRITERIA_KEY, QUESTION_KEY, CallInfo, Choice, Noul, Score

__all__ = [
    "API_KEY_ENV",
    "CRITERIA_KEY",
    "QUESTION_KEY",
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
]
