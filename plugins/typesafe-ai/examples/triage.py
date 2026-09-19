"""Triage a support ticket with one System One call.

The battery below asks **fourteen** questions. That is the point: System One
answers them in parallel and in isolation, so asking fourteen costs about what
asking three costs, and every extra question is one more thing your code can
branch on without another round trip. A generative model would have to write all
fourteen fields out one token at a time.

    export TYPESAFE_LOCAL_WHEELS=1          # while the plugin is unpublished
    flyte run --root-dir plugins/typesafe-ai/examples plugins/typesafe-ai/examples/triage.py handle
"""

import enum
import pathlib
from dataclasses import dataclass, field

import flyte
from _env import env

from flyteplugins.typesafe_ai import Choice, Noul, Score, ask_with_info


class Intent(enum.Enum):
    """What is this customer actually asking for?"""

    REFUND = "refund"
    """they want money back for something already paid for"""
    DELIVERY_STATUS = "delivery status"
    """they want to know where an order is, or when it will arrive"""
    TECHNICAL_ISSUE = "technical issue"
    """something in the product is not working"""
    ACCOUNT_ACCESS = "account access"
    """they cannot get into their account"""
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


@dataclass
class Triage:
    """One ticket, fourteen typed answers, one request."""

    # --- the three that decide what happens -------------------------------
    # Intent and Severity document themselves above -- class docstring for the
    # question, member docstrings for the criteria -- so these need no metadata.
    intent: Choice[Intent]
    severity: Score[Severity]
    hostile: Noul = field(
        metadata={
            "question": "Is the customer hostile or abusive?",
            "criteria": {"true": "insults, threats or slurs", "false": "civil, even if angry"},
        }
    )
    # --- the eleven a human reviewer wants anyway --------------------------
    has_order_reference: Noul = field(
        metadata={"question": "Does the ticket name a specific order or reference number?"}
    )
    money_at_stake: Noul = field(metadata={"question": "Is a payment, refund or charge involved?"})
    deadline_mentioned: Noul = field(
        metadata={"question": "Does the customer mention a deadline or an event they need this for?"}
    )
    already_contacted: Noul = field(metadata={"question": "Have they contacted support about this before?"})
    asks_for_human: Noul = field(metadata={"question": "Are they explicitly asking for a human agent?"})
    threatens_chargeback: Noul = field(
        metadata={"question": "Do they threaten a chargeback, a review or legal action?"}
    )
    needs_account_change: Noul = field(metadata={"question": "Would resolving this require changing their account?"})
    reports_bug: Noul = field(metadata={"question": "Are they reporting something that looks like a product defect?"})
    mentions_competitor: Noul = field(metadata={"question": "Do they mention leaving for a competitor?"})
    non_english: Noul = field(metadata={"question": "Is the ticket written in a language other than English?"})
    resolvable_now: Noul = field(metadata={"question": "Could a well-informed agent resolve this in a single reply?"})


SAMPLE = (
    "Order AC-1042 was supposed to arrive Tuesday and it is still not here. I needed it for my "
    "daughter's recital on Saturday. This is the second time I've written in and nobody has replied. "
    "If it doesn't ship today I'm disputing the charge with my bank."
)


@env.task
async def triage(ticket: str) -> Triage:
    answered, info = await ask_with_info(Triage, {"ticket": ticket})
    print(f"{info.questions} questions, one call: {info.latency_s}s, {info.input_tokens}in/{info.output_tokens}out")
    return answered


@env.task
async def handle(ticket: str = SAMPLE) -> str:
    """Route the ticket in ordinary code, on calibrated numbers rather than vibes."""
    t = await triage(ticket)

    # Guard first: what must never be auto-answered.
    if t.hostile.at(0.8) or t.threatens_chargeback.at(0.7):
        route = "escalate"
    # Then confidence: act only when the pick is clear.
    elif not t.intent.certain(0.85) or t.severity.at_least(Severity.BLOCKING):
        route = "review"
    else:
        route = "auto"

    # Every facet that came back true -- free to read, because they rode along in the same call.
    fired = sorted(name for name, answer in vars(t).items() if isinstance(answer, Noul) and answer.at(0.5))
    return (
        f"route={route} intent={t.intent.value.value} (p={t.intent.confidence:.2f}) "
        f"severity={t.severity.value.name}@{t.severity.position:.1f} fired={fired}"
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(handle)
    print(run.name, run.url)
    run.wait()
