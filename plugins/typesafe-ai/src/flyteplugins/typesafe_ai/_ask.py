"""Turn typed questions into one System One call, and back again.

The whole point of System One is that questions are answered in parallel and in
isolation, so asking forty of them costs about what asking three costs. That only
pays off if asking forty is as easy to write as asking three.

There are three ways to say what you want, and they compile to the same request:

```python
await ask(Triage, state)                                       # a battery dataclass
await ask(Choice[Intent], state)                               # a single question
await ask({"intent": Choice[Intent], "hostile": Noul}, state)  # an ad-hoc battery
```

In every form the vocabulary documents itself: an enum's class docstring is the
question and its member docstrings are the criteria. Override either with metadata
-- `Annotated[Choice[Intent], {"question": ...}]` anywhere, or
`field(metadata={...})` on a dataclass field, which wins over the annotation
because it is the more specific place to say it.
"""

from __future__ import annotations

import dataclasses
import enum
import time
import typing
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from ._client import client as make_client
from ._docs import class_doc, member_docs
from ._types import CRITERIA_KEY, QUESTION_KEY, CallInfo, Choice, Noul, Score

B = TypeVar("B")

#: What you can hand to ask(): a battery dataclass, one question type, or a mapping.
Askable = Union[Type[Any], Mapping[str, Any]]

#: The name a single, unnamed question is asked under.
SINGLE = "answer"


class BatteryError(TypeError):
    """The questions cannot be compiled. Always a problem in your code."""


@dataclasses.dataclass(frozen=True)
class _Spec:
    """One question, however it was declared."""

    name: str
    kind: str  # choice | score | noul
    type: Any  # the bare Choice[...] / Score[...] / Noul
    enum_cls: Optional[Type[enum.Enum]]
    meta: Mapping[str, Any]


def _unwrap(tp: Any) -> Tuple[Any, Mapping[str, Any]]:
    """Split Annotated[X, {...}] into X and the metadata mapping it carries."""
    if typing.get_origin(tp) is not None and hasattr(tp, "__metadata__"):
        bare = typing.get_args(tp)[0]
        meta: Dict[str, Any] = {}
        for extra in tp.__metadata__:
            if isinstance(extra, Mapping):
                meta.update(extra)
            elif isinstance(extra, str):
                meta.setdefault(QUESTION_KEY, extra)
        return bare, meta
    return tp, {}


def _kind(tp: Any) -> Optional[str]:
    origin = typing.get_origin(tp) or tp
    if origin is Noul:
        return "noul"
    if origin is Choice:
        return "choice"
    if origin is Score:
        return "score"
    return None


def _enum_arg(tp: Any) -> Optional[Type[enum.Enum]]:
    args = typing.get_args(tp)
    if args and isinstance(args[0], type) and issubclass(args[0], enum.Enum):
        return args[0]
    return None


def _spec(name: str, tp: Any, extra_meta: Optional[Mapping[str, Any]] = None) -> _Spec:
    bare, annotated_meta = _unwrap(tp)
    kind = _kind(bare)
    if kind is None:
        raise BatteryError(
            f"'{name}' is typed {tp!r}, which is not a question. It must be "
            "Choice[SomeEnum], Score[SomeIntEnum] or Noul."
        )
    # field metadata beats the annotation: it is the more specific place to say it.
    meta = {**annotated_meta, **(extra_meta or {})}
    enum_cls = _enum_arg(bare)
    if kind == "choice" and enum_cls is None:
        raise BatteryError(f"Choice '{name}' needs an Enum type argument, e.g. Choice[Intent].")
    if kind == "score" and (enum_cls is None or not issubclass(enum_cls, enum.IntEnum)):
        raise BatteryError(f"Score '{name}' needs an IntEnum type argument, e.g. Score[Severity].")
    return _Spec(name=name, kind=kind, type=bare, enum_cls=enum_cls, meta=meta)


def _specs(askable: Askable) -> list[_Spec]:
    """Normalise every accepted form into one list of questions."""
    if isinstance(askable, Mapping):
        return [_spec(name, tp) for name, tp in askable.items()]

    # Check for a question type before the dataclass branch: the answer types are
    # themselves dataclasses, so a bare `Noul` would otherwise look like an empty battery.
    if _kind(_unwrap(askable)[0]) is not None:
        return [_spec(SINGLE, askable)]

    if dataclasses.is_dataclass(askable) and isinstance(askable, type):
        hints = typing.get_type_hints(askable, include_extras=True)
        specs = []
        for f in dataclasses.fields(askable):
            tp = hints[f.name]
            bare, _ = _unwrap(tp)
            if _kind(bare) is None:
                if QUESTION_KEY in f.metadata:  # a question was intended, the type is wrong
                    raise BatteryError(
                        f"Field '{f.name}' is typed {tp!r}, which is not a question. It must be "
                        "Choice[SomeEnum], Score[SomeIntEnum] or Noul."
                    )
                continue  # an ordinary field, carried along but never asked
            specs.append(_spec(f.name, tp, f.metadata))
        if not specs:
            raise BatteryError(
                f"{askable.__name__} has no question fields. A question field is typed "
                "Choice[SomeEnum], Score[SomeIntEnum] or Noul."
            )
        # ask() builds the battery out of answers alone, so any other field has to be
        # able to default itself. Without this the failure is a bare TypeError from
        # __init__ that never mentions the battery.
        unfillable = [
            f.name
            for f in dataclasses.fields(askable)
            if f.name not in {sp.name for sp in specs}
            and f.init
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        if unfillable:
            raise BatteryError(
                f"{askable.__name__} has non-question field(s) {unfillable} with no default. "
                "ask() constructs the battery from the answers, so every field it does not ask "
                "for needs a default."
            )
        return specs

    return [_spec(SINGLE, askable)]  # not a battery and not a question: _spec explains why


def _instructions(spec: _Spec) -> str:
    """The question text: what was written here, or what the vocabulary already says."""
    asked = spec.meta.get(QUESTION_KEY)
    if isinstance(asked, str) and asked.strip():
        return asked
    inherited = class_doc(spec.enum_cls) if spec.enum_cls is not None else None
    if inherited:
        return inherited
    hint = (
        "A Noul has no vocabulary to document itself with, so it always needs one."
        if spec.kind == "noul"
        else f"Give {spec.enum_cls.__name__} a class docstring, or say it here."  # type: ignore[union-attr]
    )
    raise BatteryError(f"'{spec.name}' has no question. {hint}")


def _described(enum_cls: Type[enum.Enum]) -> Dict[str, Optional[str]]:
    docs = member_docs(enum_cls)
    return {m.name: docs.get(m.name) for m in enum_cls}


def _choice_criteria(spec: _Spec) -> Dict[str, Optional[str]]:
    """Keyed by member name -- the same string Flyte puts on the wire for an enum."""
    enum_cls = typing.cast(Type[enum.Enum], spec.enum_cls)
    criteria = _described(enum_cls)
    given = spec.meta.get(CRITERIA_KEY)
    if given is None:
        return criteria
    if not isinstance(given, Mapping):
        raise BatteryError(f"Choice '{spec.name}' needs its criteria as a mapping of option -> description.")
    for key, text in given.items():
        member = key.name if isinstance(key, enum.Enum) else str(key)
        if member not in criteria:
            raise BatteryError(f"Choice '{spec.name}': '{member}' is not a member of {enum_cls.__name__}.")
        criteria[member] = text
    return criteria


def _score_criteria(spec: _Spec) -> list:
    """Positional, one rung per step -- which is what the SDK's Score takes."""
    enum_cls = typing.cast(Type[enum.IntEnum], spec.enum_cls)
    members = sorted(enum_cls, key=lambda m: m.value)
    expected = list(range(len(members)))
    if [m.value for m in members] != expected:
        raise BatteryError(
            f"Score '{spec.name}' uses {enum_cls.__name__}, whose values are {[m.value for m in members]}. "
            f"A rubric is positional, so the members must be {expected} -- one rung per step, starting at zero."
        )
    given = spec.meta.get(CRITERIA_KEY)
    if isinstance(given, Mapping):
        given = {(k.name if isinstance(k, enum.Enum) else str(k)): v for k, v in given.items()}
        unknown = set(given) - {m.name for m in members}
        if unknown:
            raise BatteryError(f"Score '{spec.name}': {sorted(unknown)} are not members of {enum_cls.__name__}.")
    elif isinstance(given, Sequence) and not isinstance(given, str):
        if len(given) != len(members):
            raise BatteryError(
                f"Score '{spec.name}' has {len(given)} criteria for {len(members)} rungs. "
                "A positional rubric needs one entry per member."
            )
        return list(given)
    elif given is not None:
        raise BatteryError(f"Score '{spec.name}' needs its criteria as a sequence of rungs, or a mapping.")
    else:
        given = {}

    docs = _described(enum_cls)
    return [given.get(m.name) or docs.get(m.name) or m.name.replace("_", " ").lower() for m in members]


def compile_questions(askable: Askable) -> Dict[str, Any]:
    """Build the SDK's question objects from any accepted form."""
    import typesafe_sdk as ts

    questions: Dict[str, Any] = {}
    for spec in _specs(askable):
        instructions = _instructions(spec)
        if spec.kind == "noul":
            given = spec.meta.get(CRITERIA_KEY)
            criteria = None
            if given is not None:
                if not isinstance(given, Mapping):
                    raise BatteryError(f"Noul '{spec.name}' needs criteria like {{'true': ..., 'false': ...}}.")
                criteria = ts.NoulCriteria(true=given.get("true"), false=given.get("false"))
            questions[spec.name] = ts.Noul(instructions=instructions, criteria=criteria)
        elif spec.kind == "choice":
            questions[spec.name] = ts.Choice(instructions=instructions, criteria=_choice_criteria(spec))
        else:
            questions[spec.name] = ts.Score(instructions=instructions, criteria=_score_criteria(spec))
    return questions


def _by_member_name(enum_cls: Type[enum.Enum], probabilities: Dict[str, float]) -> Dict[str, float]:
    """Re-key a rung-indexed distribution by member name, leaving unknown keys alone."""
    by_value = {str(m.value): m.name for m in enum_cls}
    return {by_value.get(k, k): v for k, v in probabilities.items()}


def _probabilities(raw: Any) -> Dict[str, float]:
    if not raw:
        return {}
    return {str(k): float(v) for k, v in dict(raw).items()}


def _answer(spec: _Spec, answer: Any) -> Any:
    if answer is None:
        raise BatteryError(f"System 1 returned no answer for '{spec.name}'.")
    if spec.kind == "noul":
        return Noul(value=float(answer.noul))
    enum_cls = typing.cast(Type[enum.Enum], spec.enum_cls)
    confidence = float(getattr(answer, "confidence", 0.0) or 0.0)
    probabilities = _probabilities(getattr(answer, "probabilities", None))
    if spec.kind == "choice":
        return Choice(value=enum_cls[answer.choice], confidence=confidence, probabilities=probabilities)
    position = float(answer.score)
    rung = min(max(round(position), 0), len(list(enum_cls)) - 1)
    return Score(
        value=enum_cls(rung),
        position=position,
        confidence=confidence,
        # The SDK keys a Score's distribution by rung index; name them, so that both
        # Choice.probabilities and Score.probabilities read as {member name: p}.
        probabilities=_by_member_name(enum_cls, probabilities),
    )


def _assemble(askable: Askable, specs: list[_Spec], answers: Mapping[str, Any]) -> Any:
    """Shape the answers like the thing that was asked."""
    built = {spec.name: _answer(spec, answers.get(spec.name)) for spec in specs}
    if isinstance(askable, Mapping):
        return built
    if dataclasses.is_dataclass(askable) and isinstance(askable, type):
        return askable(**built)
    return built[SINGLE]


def _answers_of(resp: Any) -> Dict[str, Any]:
    answers = getattr(resp, "answers", None)
    if answers is None:  # older/newer shapes expose the three groups instead
        answers = {**(resp.nouls or {}), **(resp.choices or {}), **(resp.scores or {})}
    return dict(answers)


async def ask_with_info(
    askable: Askable,
    state: Any,
    *,
    model: Optional[str] = None,
    client: Any = None,
) -> Tuple[Any, CallInfo]:
    """Answer everything in one call, and report what the call cost."""
    specs = _specs(askable)
    questions = compile_questions(askable)
    own = client is None
    client = client or make_client(model=model)
    started = time.perf_counter()
    try:
        resp = await client.system_one(state=state, questions=questions)
    finally:
        if own:
            await client.aclose()
    usage = getattr(resp, "usage", None)
    info = CallInfo(
        model=str(getattr(resp, "model", "") or ""),
        questions=len(questions),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        latency_s=round(time.perf_counter() - started, 3),
    )
    return _assemble(askable, specs, _answers_of(resp)), info


async def ask(askable: Askable, state: Any, *, model: Optional[str] = None, client: Any = None) -> Any:
    """Answer a battery, a single question, or a mapping of them -- in one call.

    ```python
    triage = await ask(Triage, {"ticket": text})                  # -> Triage
    intent = await ask(Choice[Intent], {"ticket": text})          # -> Choice[Intent]
    both = await ask({"intent": Choice[Intent], "hot": Noul}, s)  # -> dict
    ```
    """
    answered, _ = await ask_with_info(askable, state, model=model, client=client)
    return answered
