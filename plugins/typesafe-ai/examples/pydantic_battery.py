"""A battery as a pydantic model, and how that differs from Pydantic AI.

Pydantic AI's TypeSafe integration (https://pydantic.dev/docs/ai/models/typesafe/)
puts the *question* in the field type: a `bool` field is a yes/no question, a
`Literal` is a pick-one, an `IntEnum` is a rubric. The answers come back as plain
values and the calibration arrives beside them, in `response.provider_details`.

This plugin puts the *answer* in the field type instead. A field is a
`Choice[Intent]`, not an `Intent`, so the confidence and the full distribution
travel with the value: into your model, through pydantic validators, and across
Flyte task boundaries, with no side channel to carry along.

Both shapes are available. `bool` and `Literal` fields work here too, for filling
a model of plain values -- and because a `bool` has to cut a 0..1 answer somewhere,
it must say where, either on the field or at the call. What a plain field drops is
still on the `CallInfo`, which is this plugin's answer to `provider_details`.

Neither is better in the abstract. Reach for Pydantic AI's shape when you want a
model of plain values and will consult the calibration once, at the call site;
reach for this one when the calibration is part of what downstream code decides
on -- which is what routing on confidence means.

    export TYPESAFE_LOCAL_WHEELS=1
    flyte run --root-dir plugins/typesafe-ai/examples \
        plugins/typesafe-ai/examples/pydantic_battery.py handle
"""

import enum
import pathlib
from typing import Annotated, Literal

import flyte
from _env import env
from pydantic import BaseModel, Field, model_validator

from flyteplugins.typesafe_ai import Choice, Noul, Score, ask_with_info


class Intent(enum.Enum):
    """What is this customer asking for?"""

    REFUND = "refund"
    """they want money back for something already paid for"""
    DELIVERY_STATUS = "delivery status"
    """they want to know where an order is, or when it will arrive"""
    TECHNICAL_ISSUE = "technical issue"
    """something in the product is not working"""
    OTHER = "something else"
    """none of the above fits"""


class Severity(enum.IntEnum):
    """How badly is this customer blocked right now?"""

    NONE = 0
    """no impact; a question or a comment"""
    MINOR = 1
    """inconvenient, but they can carry on"""
    SERIOUS = 2
    """they are blocked and a deadline or payment is involved"""
    BLOCKING = 3
    """they cannot use the product at all, or money is already lost"""


class Triage(BaseModel):
    """Every question here is answered by one call; the model is what comes back."""

    # The enums document themselves, so this field needs nothing at all.
    intent: Choice[Intent]
    # `description` reads naturally as the question.
    severity: Score[Severity] = Field(description="How badly is this customer blocked right now?")
    # `json_schema_extra` carries what a description cannot.
    hostile: Noul = Field(
        description="Is the customer hostile or abusive?",
        json_schema_extra={"criteria": {"true": "insults, threats or slurs", "false": "civil, even if angry"}},
    )
    # `Annotated` works here too, if you prefer the question next to the type.
    refund_requested: Annotated[Noul, "Are they asking for money back?"]

    # Shorthand fields. A bool and a Literal hold a plain value, so they are only
    # asked when the field declares a question -- `conflicted` below stays data.
    # A bool also has to say where to cut the 0..1 answer: here, on the field.
    asks_for_human: bool = Field(
        description="Are they explicitly asking for a human agent?",
        json_schema_extra={"threshold": 0.7},
    )
    channel: Literal["email", "chat", "phone"] = Field(description="Which channel did this arrive on?")

    # Not a question -- a bool the model fills in for itself. Non-question fields
    # need a default, because ask() builds the battery out of the answers.
    conflicted: bool = False

    @model_validator(mode="after")
    def _flag_disagreement(self) -> "Triage":
        """Cross-field checks are the reason to reach for a model.

        The calibration is right here in the fields, so a validator can act on it
        without the caller threading a side channel through.
        """
        asks_for_refund = self.refund_requested.at(0.6)
        picked_refund = self.intent.value is Intent.REFUND
        self.conflicted = asks_for_refund != picked_refund
        return self

    @property
    def route(self) -> str:
        """auto / review / escalate, decided in ordinary Python."""
        if self.hostile.at(0.8):
            return "escalate"
        if self.conflicted or not self.intent.certain(0.85) or self.severity.at_least(Severity.BLOCKING):
            return "review"
        return "auto"


SAMPLE = (
    "I was charged twice for the same order last week and nobody has got back to me. "
    "Please refund the duplicate charge today."
)


@env.task
async def triage(ticket: str) -> Triage:
    """The model crosses the task boundary whole, calibration included."""
    answered, info = await ask_with_info(Triage, {"ticket": ticket}, threshold=0.5)
    # A plain field drops the calibration from the model, never from the call:
    # `values` has the float behind each bool, `confidence` the pick-one certainty.
    print(f"{info.questions} questions, one call: {info.latency_s}s")
    print(f"  behind the plain fields: values={info.values} confidence={info.confidence}")
    return answered


@env.task
async def handle(ticket: str = SAMPLE) -> str:
    t = await triage(ticket)
    return (
        f"route={t.route} intent={t.intent.value.value} (p={t.intent.confidence:.2f}) "
        f"severity={t.severity.value.name}@{t.severity.position:.1f} "
        f"refund_requested={t.refund_requested.value:.2f} conflicted={t.conflicted} "
        f"asks_for_human={t.asks_for_human} channel={t.channel}"
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(handle)
    print(run.name, run.url)
    run.wait()
