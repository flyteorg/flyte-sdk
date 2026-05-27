"""Pydantic v2 discriminated (tagged) unions as task inputs.

``Annotated[Union[A, B], Field(discriminator=...)]`` is a common Pydantic v2
pattern that emits a JSON schema using ``oneOf`` plus a top-level
``discriminator`` object (no top-level ``type`` and no ``anyOf``). This
example exercises that pattern end-to-end through ``flyte.run`` so the type
engine has to roundtrip the union through serialization and reconstruct the
right variant on the receiving side.

It deliberately covers the three flavors of discriminated unions that the
type engine needs to handle correctly:

1. ``Shape`` -- a "classic" string-``Literal`` discriminator with variants
   that have *non-overlapping* fields (``Circle.radius`` vs ``Rectangle.width``).
2. ``Event`` -- an ``Enum``-typed discriminator where the variants have
   *overlapping* field sets. Without a discriminator the type engine would
   not be able to unambiguously pick the right variant from a dict, so this
   exercises the schema-declared ``discriminator.mapping`` dispatch.
3. ``Payment`` -- the same enum-discriminator pattern but used inside a
   ``list[...]`` field, so the type engine has to dispatch per-element.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

import flyte

env = flyte.TaskEnvironment(
    name="inputs_pydantic_union_types",
    image=flyte.Image.from_debian_base(),
)


# ---- Case 1: classic string-Literal discriminator, non-overlapping fields ----


class Circle(BaseModel):
    kind: Literal["circle"] = "circle"
    color: str = ""
    radius: float = 0.0


class Rectangle(BaseModel):
    kind: Literal["rectangle"] = "rectangle"
    color: str = ""
    width: float = 0.0
    height: float = 0.0


Shape = Annotated[Union[Circle, Rectangle], Field(discriminator="kind")]


class Properties(BaseModel):
    shape: Shape


class ShapeReport(BaseModel):
    kind: str
    color: str
    area: float


# ---- Case 2: Enum-typed discriminator with overlapping fields ---------------


class EventKind(str, Enum):
    """An enum-typed discriminator value.

    Note: subclassing ``str`` keeps the JSON-serialized form a plain string
    (``"login"`` rather than ``"EventKind.LOGIN"``). The type engine also
    handles non-``str`` ``Enum``s, but ``str``-enums are by far the most
    common pattern in real-world Pydantic models.
    """

    LOGIN = "login"
    LOGOUT = "logout"


class LoginEvent(BaseModel):
    # Variants share the ``user_id`` and ``timestamp`` fields. The
    # discriminator is the *only* way to distinguish them from a raw dict --
    # this is the case AdilFayyaz called out in the PR review.
    kind: Literal[EventKind.LOGIN] = EventKind.LOGIN
    user_id: str
    timestamp: float
    source_ip: str = ""


class LogoutEvent(BaseModel):
    kind: Literal[EventKind.LOGOUT] = EventKind.LOGOUT
    user_id: str
    timestamp: float
    session_duration_s: float = 0.0


Event = Annotated[Union[LoginEvent, LogoutEvent], Field(discriminator="kind")]


class EventEnvelope(BaseModel):
    event: Event


class EventSummary(BaseModel):
    kind: str
    user_id: str


# ---- Case 3: List of enum-discriminated unions ------------------------------


class PaymentMethod(str, Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"


class CardPayment(BaseModel):
    method: Literal[PaymentMethod.CARD] = PaymentMethod.CARD
    amount: float
    last4: str = ""


class BankTransferPayment(BaseModel):
    method: Literal[PaymentMethod.BANK_TRANSFER] = PaymentMethod.BANK_TRANSFER
    amount: float
    account_id: str = ""


Payment = Annotated[Union[CardPayment, BankTransferPayment], Field(discriminator="method")]


class PaymentBatch(BaseModel):
    payments: list[Payment] = Field(default_factory=list)


class PaymentTotals(BaseModel):
    total: float
    by_method: dict[str, float]


# ---- Tasks -----------------------------------------------------------------


@env.task
def describe_shape(props: Properties) -> ShapeReport:
    """Receive a Pydantic model whose field is a discriminated union."""
    shape = props.shape
    if isinstance(shape, Circle):
        area = 3.141592653589793 * shape.radius * shape.radius
    else:
        area = shape.width * shape.height
    return ShapeReport(kind=shape.kind, color=shape.color, area=area)


@env.task
def summarize_event(envelope: EventEnvelope) -> EventSummary:
    """Dispatch on an enum-typed discriminator with overlapping-field variants."""
    event = envelope.event
    return EventSummary(kind=event.kind.value, user_id=event.user_id)


@env.task
def total_payments(batch: PaymentBatch) -> PaymentTotals:
    """Iterate a list of enum-discriminated variants and aggregate amounts."""
    by_method: dict[str, float] = {}
    for payment in batch.payments:
        by_method[payment.method.value] = by_method.get(payment.method.value, 0.0) + payment.amount
    return PaymentTotals(total=sum(by_method.values()), by_method=by_method)


@env.task
def main(
    props: Properties = Properties(shape=Rectangle(color="blue", width=4.0, height=2.5)),
    envelope: EventEnvelope = EventEnvelope(event=LoginEvent(user_id="u-1", timestamp=0.0)),
    batch: PaymentBatch = PaymentBatch(
        payments=[
            CardPayment(amount=12.50, last4="4242"),
            BankTransferPayment(amount=100.00, account_id="acct-7"),
        ],
    ),
) -> tuple[ShapeReport, EventSummary, PaymentTotals]:
    return (
        describe_shape(props=props),
        summarize_event(envelope=envelope),
        total_payments(batch=batch),
    )


if __name__ == "__main__":
    flyte.init_from_config()

    # Run with all defaults: Rectangle shape + LoginEvent + a mixed payment batch.
    print("Testing with defaults (Rectangle / LoginEvent / mixed payments):")
    r1 = flyte.run(main)
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Exercise the other variant of each discriminated union.
    print("\nTesting with Circle / LogoutEvent / single CardPayment:")
    r2 = flyte.run(
        main,
        props=Properties(shape=Circle(color="red", radius=2.0)),
        envelope=EventEnvelope(
            event=LogoutEvent(user_id="u-1", timestamp=10.0, session_duration_s=10.0),
        ),
        batch=PaymentBatch(payments=[CardPayment(amount=42.00, last4="1234")]),
    )
    print(r2.name)
    print(r2.url)
    r2.wait()
