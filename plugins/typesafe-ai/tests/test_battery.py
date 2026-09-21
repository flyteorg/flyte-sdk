"""The battery compiles to the right questions and comes back as typed answers."""

import enum
from dataclasses import dataclass, field
from typing import Annotated

import pytest
import typesafe_sdk as ts

from flyteplugins.typesafe_ai import (
    BatteryError,
    Choice,
    MissingAPIKey,
    Noul,
    Score,
    ask_with_info,
    client,
    compile_questions,
)
from flyteplugins.typesafe_ai._client import API_KEY_ENV


class Intent(enum.Enum):
    """What is this customer asking for?"""

    REFUND = "refund"
    """they want money back"""
    DELIVERY = "delivery"
    """they are asking where their order is"""
    OTHER = "other"


class Severity(enum.IntEnum):
    """How badly is this customer blocked?"""

    NONE = 0
    """no impact; a question or a comment"""
    MINOR = 1
    """inconvenient, but they can carry on"""
    SERIOUS = 2
    """they are blocked and a deadline is involved"""
    BLOCKING = 3
    """they cannot use the product at all"""


@dataclass
class Triage:
    intent: Choice[Intent] = field(metadata={"question": "Which intent fits?"})
    severity: Score[Severity] = field(metadata={"question": "How bad is it?"})
    hostile: Noul = field(
        metadata={"question": "Hostile?", "criteria": {"true": "insults or threats", "false": "civil"}}
    )


class _FakeResponse:
    model = "jev-test"

    class usage:
        input_tokens = 120
        output_tokens = 30

    def __init__(self, answers):
        self.answers = answers


class _FakeClient:
    """Records what was asked and replies with the SDK's answer objects."""

    def __init__(self):
        self.asked = None

    async def system_one(self, state, questions):
        self.asked = (state, questions)
        return _FakeResponse(
            {
                "intent": ts.ChoiceAnswer(
                    choice="REFUND", confidence=0.91, probabilities={"REFUND": 0.91, "OTHER": 0.09}
                ),
                "severity": ts.ScoreAnswer(score=2.4, confidence=0.77, legend={}, probabilities={}),
                "hostile": ts.NoulAnswer(noul=0.03),
            }
        )

    async def aclose(self):
        pass


# --------------------------------------------------------------- compilation


def test_questions_come_from_field_metadata():
    qs = compile_questions(Triage)
    assert set(qs) == {"intent", "severity", "hostile"}
    assert qs["intent"].instructions == "Which intent fits?"
    assert qs["hostile"].criteria["true"] == "insults or threats"


def test_choice_criteria_are_inferred_from_member_docstrings():
    """The enum documents itself, so the field does not repeat it."""
    qs = compile_questions(Triage)
    # keyed by member NAME -- the string Flyte puts on the wire for an enum
    assert qs["intent"].criteria == {
        "REFUND": "they want money back",
        "DELIVERY": "they are asking where their order is",
        "OTHER": None,  # undocumented members simply carry no description
    }


def test_score_criteria_are_positional_and_inferred():
    qs = compile_questions(Triage)
    assert qs["severity"].criteria == [
        "no impact; a question or a comment",
        "inconvenient, but they can carry on",
        "they are blocked and a deadline is involved",
        "they cannot use the product at all",
    ]


def test_question_falls_back_to_the_enum_class_docstring():
    @dataclass
    class Minimal:
        intent: Choice[Intent] = field(metadata={"question": ""})

    assert compile_questions(Minimal)["intent"].instructions == "What is this customer asking for?"


def test_noul_must_be_given_a_question():
    """A Noul has no vocabulary to document itself with -- this is the one that needs you."""

    @dataclass
    class Bad:
        hostile: Noul = field(metadata={"question": ""})

    with pytest.raises(BatteryError, match="has no question"):
        compile_questions(Bad)


def test_explicit_criteria_override_the_docstrings():
    @dataclass
    class Override:
        intent: Choice[Intent] = field(
            metadata={"question": "which?", "criteria": {Intent.REFUND: "money back, explicitly"}}
        )
        severity: Score[Severity] = field(
            metadata={"question": "how bad?", "criteria": ["a", "b", "c", "d"]},
        )

    qs = compile_questions(Override)
    assert qs["intent"].criteria["REFUND"] == "money back, explicitly"
    assert qs["intent"].criteria["DELIVERY"] == "they are asking where their order is"  # still inferred
    assert qs["severity"].criteria == ["a", "b", "c", "d"]


def test_bad_criteria_are_rejected_with_the_field_named():
    @dataclass
    class Unknown:
        intent: Choice[Intent] = field(metadata={"question": "which?", "criteria": {"NOPE": "x"}})

    with pytest.raises(BatteryError, match="'NOPE' is not a member"):
        compile_questions(Unknown)

    @dataclass
    class WrongLength:
        severity: Score[Severity] = field(metadata={"question": "how bad?", "criteria": ["only", "two"]})

    with pytest.raises(BatteryError, match="one entry per member"):
        compile_questions(WrongLength)


def test_non_contiguous_rubric_is_rejected():
    class Sparse(enum.IntEnum):
        LOW = 1
        HIGH = 5

    @dataclass
    class Bad:
        s: Score[Sparse] = field(metadata={"question": "how much?"})

    with pytest.raises(BatteryError, match="one rung per step"):
        compile_questions(Bad)


def test_unaskable_field_is_rejected():
    @dataclass
    class Bad:
        note: str = field(metadata={"question": "what is the note?"})

    with pytest.raises(BatteryError, match="not a question"):
        compile_questions(Bad)


def test_battery_without_questions_is_rejected():
    @dataclass
    class Empty:
        x: int = 0

    with pytest.raises(BatteryError, match="no question fields"):
        compile_questions(Empty)


# ------------------------------------------------------------------ answering


@pytest.mark.asyncio
async def test_ask_returns_typed_answers():
    fake = _FakeClient()
    answered, info = await ask_with_info(Triage, {"ticket": "where is my order"}, client=fake)

    assert answered.intent.value is Intent.REFUND  # the member, not "REFUND"
    assert answered.intent.certain(0.9) and not answered.intent.certain(0.95)
    assert answered.intent.runner_up() == ("OTHER", 0.09)

    assert answered.severity.value is Severity.SERIOUS  # rounded rung
    assert answered.severity.position == 2.4  # and the unrounded position
    assert answered.severity.at_least(Severity.MINOR)

    assert answered.hostile.at(0.8) is False
    assert answered.hostile.value == 0.03

    assert info.questions == 3
    assert (info.input_tokens, info.output_tokens) == (120, 30)
    assert fake.asked[0] == {"ticket": "where is my order"}


@pytest.mark.asyncio
async def test_all_questions_ride_in_a_single_request():
    fake = _FakeClient()
    await ask_with_info(Triage, {"ticket": "hi"}, client=fake)
    assert len(fake.asked[1]) == 3  # not three requests -- three questions in one


# ----------------------------------------------------------- flyte boundaries


@pytest.mark.asyncio
async def test_battery_crosses_a_task_boundary():
    """The answers are ordinary dataclasses, so Flyte carries them with nothing registered."""
    from flyte.types import TypeEngine

    answered = Triage(
        intent=Choice(Intent.DELIVERY, 0.88, {"DELIVERY": 0.88}),
        severity=Score(Severity.MINOR, 1.2, 0.7, {}),
        hostile=Noul(0.01),
    )
    lt = TypeEngine.to_literal_type(Triage)
    back = await TypeEngine.to_python_value(await TypeEngine.to_literal(answered, Triage, lt), Triage)

    assert back == answered
    assert back.intent.value is Intent.DELIVERY  # still the enum member
    assert back.severity.value is Severity.MINOR
    assert back.severity.position == 1.2


@pytest.mark.asyncio
async def test_bare_int_enum_is_a_usable_task_output():
    """Relies on the IntEnum support added to flyte's EnumTransformer."""
    from flyte.types import TypeEngine

    lt = TypeEngine.to_literal_type(Severity)
    lv = await TypeEngine.to_literal(Severity.BLOCKING, Severity, lt)
    assert lv.scalar.primitive.string_value == "BLOCKING"
    assert await TypeEngine.to_python_value(lv, Severity) is Severity.BLOCKING


# ----------------------------------------------------------------- the secret


def test_missing_key_explains_the_fix(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(MissingAPIKey, match="flyte create secret"):
        client()


def test_documented_secret_spelling_mounts_the_expected_env_var():
    """The README tells people to write this; it should mount what the SDK reads."""
    import flyte

    assert flyte.Secret(key="TYPESAFE_API_KEY", as_env_var=API_KEY_ENV).as_env_var == API_KEY_ENV
    assert flyte.Secret(key="TYPESAFE_API_KEY").as_env_var == API_KEY_ENV  # derived from the key alone


# ------------------------------------------------- docstrings as the default


@dataclass
class Documented:
    """Every field here relies entirely on what the enums already say."""

    intent: Choice[Intent]
    severity: Score[Severity]


def test_a_question_field_needs_no_metadata_at_all():
    qs = compile_questions(Documented)
    assert set(qs) == {"intent", "severity"}
    assert qs["intent"].instructions == "What is this customer asking for?"  # Intent's class docstring
    assert qs["severity"].instructions == "How badly is this customer blocked?"
    assert qs["intent"].criteria["REFUND"] == "they want money back"


def test_field_metadata_overrides_the_docstring():
    @dataclass
    class Overridden:
        intent: Choice[Intent] = field(metadata={"question": "asked a different way"})

    assert compile_questions(Overridden)["intent"].instructions == "asked a different way"


def test_ordinary_fields_are_carried_but_never_asked():
    @dataclass
    class WithProvenance:
        intent: Choice[Intent]
        source: str = "email"  # not a question, just data the task carries

    assert set(compile_questions(WithProvenance)) == {"intent"}


# ------------------------------------------- standalone, without a dataclass


@pytest.mark.asyncio
async def test_a_single_question_needs_no_dataclass():
    class _One(_FakeClient):
        async def system_one(self, state, questions):
            self.asked = (state, questions)
            return _FakeResponse({"answer": ts.ChoiceAnswer(choice="DELIVERY", confidence=0.8, probabilities={})})

    fake = _One()
    answered = (await ask_with_info(Choice[Intent], {"ticket": "where is it"}, client=fake))[0]
    assert isinstance(answered, Choice)
    assert answered.value is Intent.DELIVERY
    assert fake.asked[1]["answer"].instructions == "What is this customer asking for?"


@pytest.mark.asyncio
async def test_annotated_carries_the_question_outside_a_dataclass():
    class _One(_FakeClient):
        async def system_one(self, state, questions):
            self.asked = (state, questions)
            return _FakeResponse({"answer": ts.NoulAnswer(noul=0.9)})

    fake = _One()
    hostile = Annotated[Noul, {"question": "Is this abusive?", "criteria": {"true": "slurs", "false": "civil"}}]
    answered = (await ask_with_info(hostile, {"ticket": "you thieves"}, client=fake))[0]

    assert isinstance(answered, Noul) and answered.at(0.8)
    asked = fake.asked[1]["answer"]
    assert asked.instructions == "Is this abusive?"
    assert asked.criteria["true"] == "slurs"


@pytest.mark.asyncio
async def test_a_mapping_is_an_ad_hoc_battery_in_one_call():
    class _Two(_FakeClient):
        async def system_one(self, state, questions):
            self.asked = (state, questions)
            return _FakeResponse(
                {
                    "intent": ts.ChoiceAnswer(choice="REFUND", confidence=0.7, probabilities={}),
                    "hostile": ts.NoulAnswer(noul=0.1),
                }
            )

    fake = _Two()
    answered = (
        await ask_with_info(
            {"intent": Choice[Intent], "hostile": Annotated[Noul, "Is the customer hostile?"]},
            {"ticket": "refund please"},
            client=fake,
        )
    )[0]

    assert answered["intent"].value is Intent.REFUND
    assert answered["hostile"].value == 0.1
    assert len(fake.asked[1]) == 2  # one call, two questions


def test_annotated_metadata_loses_to_field_metadata():
    """The dataclass field is the more specific place to say it, so it wins."""

    @dataclass
    class Both:
        intent: Annotated[Choice[Intent], {"question": "from the annotation"}] = field(
            metadata={"question": "from the field"}
        )

    assert compile_questions(Both)["intent"].instructions == "from the field"


def test_standalone_noul_without_a_question_is_rejected():
    with pytest.raises(BatteryError, match="no vocabulary to document itself"):
        compile_questions(Noul)


@pytest.mark.asyncio
async def test_a_single_answer_is_a_usable_task_output():
    """Choice[Intent] at the top level is a struct, not a pickle.

    Relies on the generic-dataclass resolution added to flyte's type engine: the
    parameterized alias used to fall through to FlytePickle.
    """
    from flyte.types import TypeEngine

    assert TypeEngine.get_transformer(Choice[Intent]).name == "Object-Dataclass-Transformer"

    lt = TypeEngine.to_literal_type(Choice[Intent])
    val = Choice(Intent.REFUND, 0.91, {"REFUND": 0.91})
    back = await TypeEngine.to_python_value(await TypeEngine.to_literal(val, Choice[Intent], lt), Choice[Intent])
    assert back == val and back.value is Intent.REFUND

    # and it coerces from a plain dict, like any other dataclass input
    coerced = await TypeEngine.to_python_value(
        await TypeEngine.to_literal({"value": "delivery"}, Choice[Intent], lt), Choice[Intent]
    )
    assert coerced.value is Intent.DELIVERY and coerced.confidence == 0.0


# ------------------------------------------------ regressions found in review


def test_an_undocumented_enum_does_not_leak_cpythons_docstring():
    """inspect.getdoc() inherits: an undocumented enum would otherwise report
    "Create a collection of name/value pairs." (3.12+) as its question."""
    from flyteplugins.typesafe_ai._docs import class_doc

    class Undocumented(enum.Enum):
        A = "a"

    class UndocumentedInt(enum.IntEnum):
        A = 0

    assert class_doc(Undocumented) is None
    assert class_doc(UndocumentedInt) is None

    @dataclass
    class Bad:
        pick: Choice[Undocumented]

    with pytest.raises(BatteryError, match="has no question"):
        compile_questions(Bad)


def test_non_question_field_without_a_default_is_rejected_clearly():
    @dataclass
    class WithRequired:
        intent: Choice[Intent]
        source: str  # never asked, and nothing to fall back on

    with pytest.raises(BatteryError, match="needs a default"):
        compile_questions(WithRequired)


@pytest.mark.asyncio
async def test_score_probabilities_are_keyed_by_member_name():
    """The SDK keys a Score distribution by rung index; Choice keys by name. Match them."""

    class _Scored(_FakeClient):
        async def system_one(self, state, questions):
            self.asked = (state, questions)
            return _FakeResponse(
                {
                    "intent": ts.ChoiceAnswer(choice="REFUND", confidence=0.7, probabilities={}),
                    "severity": ts.ScoreAnswer(score=2.0, confidence=0.8, legend={}, probabilities={0: 0.1, 2: 0.9}),
                    "hostile": ts.NoulAnswer(noul=0.0),
                }
            )

    answered, _ = await ask_with_info(Triage, {"ticket": "x"}, client=_Scored())
    assert answered.severity.probabilities == {"NONE": 0.1, "SERIOUS": 0.9}


# ------------------------------------------------------------- registration


def test_types_are_registered_with_the_type_engine():
    """Importing the plugin is enough; flyte.init() does it too via the entry point."""
    from flyte.types import TypeEngine

    from flyteplugins.typesafe_ai import register_typesafe_ai_types

    register_typesafe_ai_types()  # idempotent: lru_cache plus a non-overriding register

    for answer_type in (Choice, Score, Noul):
        assert answer_type in TypeEngine._REGISTRY

    # A parameterized answer resolves through its origin in the registry rather than
    # the type engine's last-resort dataclass branch.
    assert TypeEngine.get_transformer(Choice[Intent]) is TypeEngine._REGISTRY[Choice]
    assert TypeEngine.get_transformer(Score[Severity]) is TypeEngine._REGISTRY[Score]
    assert TypeEngine.get_transformer(Noul) is TypeEngine._REGISTRY[Noul]


def test_entry_point_is_declared():
    """The `flyte.plugins.types` entry point is what makes flyte.init() pick these up."""
    from importlib.metadata import entry_points

    declared = {ep.name: ep.value for ep in entry_points(group="flyte.plugins.types")}
    assert declared.get("typesafe_ai") == "flyteplugins.typesafe_ai:register_typesafe_ai_types"


def test_registration_does_not_clobber_a_user_transformer():
    """register_additional_type() only fills an empty slot, so a user's choice wins."""
    from flyte.types import TypeEngine, TypeTransformer

    from flyteplugins.typesafe_ai import register_typesafe_ai_types

    class _Mine(TypeTransformer):
        def __init__(self):
            super().__init__(name="mine", t=Noul)

        def get_literal_type(self, t):
            raise NotImplementedError

        async def to_literal(self, *a):
            raise NotImplementedError

        async def to_python_value(self, *a):
            raise NotImplementedError

    original = TypeEngine._REGISTRY[Noul]
    TypeEngine.register_additional_type(_Mine(), Noul, override=True)
    try:
        register_typesafe_ai_types.cache_clear()
        register_typesafe_ai_types()
        assert TypeEngine._REGISTRY[Noul].name == "mine"
    finally:
        TypeEngine.register_additional_type(original, Noul, override=True)
        register_typesafe_ai_types.cache_clear()
