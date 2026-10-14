"""Choice, Score and Noul on their own, without a battery dataclass.

A dataclass is the right shape when the answers travel together as one artifact.
When they do not -- a single gate in the middle of a task, or a set of questions
assembled at runtime -- the same types work standalone.

    export TYPESAFE_LOCAL_WHEELS=1
    flyte run --root-dir plugins/typesafe-ai/examples plugins/typesafe-ai/examples/single.py moderate
"""

import enum
import pathlib
from typing import Annotated

import flyte
from _env import env

from flyteplugins.typesafe_ai import Choice, Noul, Score, ask


class Action(enum.Enum):
    """What should happen to this message?"""

    PUBLISH = "publish"
    """nothing here needs a human"""
    REVIEW = "review"
    """borderline; a moderator should look"""
    BLOCK = "block"
    """clearly against the rules"""


class Harm(enum.IntEnum):
    """How harmful is this message?"""

    NONE = 0
    """harmless"""
    RUDE = 1
    """rude or dismissive, but not targeted"""
    ABUSIVE = 2
    """targeted abuse or harassment"""
    DANGEROUS = 3
    """threats, or encouragement of self-harm or violence"""


MESSAGES = [
    "thanks, this fixed it for me!",
    "whoever wrote this documentation is an absolute clown",
    "i know where you live and i am going to find you",
]


@env.task
async def classify(message: str) -> Choice[Action]:
    """An answer type is a perfectly good task output.

    `Choice[Action]` is a parameterized dataclass, so it crosses the boundary as a
    struct -- the picked member, its confidence and the whole distribution -- and
    the caller gets a real `Choice` back, not a dict.
    """
    return await ask(Choice[Action], {"message": message})


@env.task
async def moderate(message: str = MESSAGES[1]) -> str:
    """Three standalone questions, each asked on its own terms."""

    # 1. A bare question type, answered in its own durable task. The enum's class
    #    docstring is the question and its member docstrings are the criteria.
    action: Choice[Action] = await classify(message)

    # 2. Annotated, when you want to ask this vocabulary a different question than
    #    the one its docstring states.
    harm: Score[Harm] = await ask(
        Annotated[Score[Harm], {"question": "How much harm would this message do if published as-is?"}],
        {"message": message},
    )

    # 3. A Noul has no vocabulary to document itself with, so it always carries its
    #    own question -- and here, its own criteria too.
    directed: Noul = await ask(
        Annotated[
            Noul,
            {
                "question": "Is this aimed at a specific person?",
                "criteria": {"true": "addressed at an individual", "false": "general or about the product"},
            },
        ],
        {"message": message},
    )

    return (
        f"action={action.value.value} (p={action.confidence:.2f}) "
        f"harm={harm.value.name}@{harm.position:.1f} directed={directed.value:.2f}"
    )


@env.task
async def moderate_in_one_call(message: str = MESSAGES[2]) -> str:
    """The same three questions, assembled as a mapping -- so they cost one request.

    Three separate `ask()` calls are three round trips. When the questions are known
    together, hand them over together: System One answers them in parallel, and the
    whole point is that the second and third are nearly free.
    """
    answers = await ask(
        {
            "action": Choice[Action],
            "harm": Score[Harm],
            "directed": Annotated[Noul, "Is this aimed at a specific person?"],
        },
        {"message": message},
    )
    return (
        f"action={answers['action'].value.value} "
        f"harm={answers['harm'].value.name} "
        f"directed={answers['directed'].value:.2f}"
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(moderate_in_one_call)
    print(run.name, run.url)
    run.wait()
