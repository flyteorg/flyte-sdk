"""The three answer types, as plain dataclasses.

TypeSafe's System One model ("Jev") answers typed questions instead of writing
text, and every answer arrives with the calibration that produced it. These
dataclasses are how that answer crosses a Flyte task boundary: Flyte's
`DataclassTransformer` carries them with no registration and no pydantic, and
nesting an `Enum` or `IntEnum` inside one keeps it a real member on the way
back out rather than degrading it to a string or an int.

Note the deliberate name collision with `typesafe_sdk.Choice` / `Score` /
`Noul`: those describe the *question* you ask, these hold the *answer* you get.
You write the ones in this module; the plugin builds the SDK's from your battery.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

C = TypeVar("C", bound=enum.Enum)
S = TypeVar("S", bound=enum.IntEnum)

#: Metadata keys on a battery field. They are the SDK's own words: a question
#: has `instructions` and `criteria`, and the shape of the criteria depends
#: on the question type, exactly as in `typesafe_sdk`.
QUESTION_KEY = "question"
CRITERIA_KEY = "criteria"
#: Where a `bool` field says where to cut the 0..1 answer it is asking for.
THRESHOLD_KEY = "threshold"


@dataclass
class Choice(Generic[C]):
    """One pick from a fixed vocabulary, carrying the calibration it came with."""

    value: C
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)

    def certain(self, threshold: float) -> bool:
        """Is this pick confident enough to act on without a human?"""
        return self.confidence >= threshold

    def runner_up(self) -> Optional[tuple[str, float]]:
        """The second-most-likely option, which is what you show a reviewer."""
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[1] if len(ranked) > 1 else None


@dataclass
class Score(Generic[S]):
    """A position on a rubric.

    Two representations, because both are useful: `value` is the rung that was
    picked (an `IntEnum` member, so it compares and orders), and `position`
    is the unrounded place on the scale that Jev actually returned. Branch on the
    first, sort and threshold on the second.
    """

    value: S
    position: float = 0.0
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)

    def at_least(self, rung: S) -> bool:
        return self.value >= rung


@dataclass
class Noul:
    """Truthfulness in 0..1.

    Deliberately no `__bool__`: `if noul:` would make 0.02 and 0.98 alike,
    and choosing the threshold is the part you want in code where it can be read,
    reviewed and changed.
    """

    value: float = 0.0

    def at(self, threshold: float) -> bool:
        return self.value >= threshold


@dataclass
class CallInfo:
    """What one `system_one` call cost, plus the calibration a plain field dropped.

    `Choice`, `Score` and `Noul` carry their own calibration, so for those the maps
    below are redundant. They exist for the shorthand field types -- a `bool` or a
    `Literal` -- which hold a plain value: the calibration is dropped from *your
    model*, not from the call, and this is where to find it.

    Both maps are `dict[str, float]` so that a task can return a `CallInfo`.
    """

    model: str = ""
    questions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    #: field name -> the raw numeric answer a plain field discarded: a `bool`'s
    #: underlying 0..1, or a `Score`'s unrounded position.
    values: dict[str, float] = field(default_factory=dict)
    #: field name -> calibrated confidence, for pick-one and rubric questions.
    confidence: dict[str, float] = field(default_factory=dict)
