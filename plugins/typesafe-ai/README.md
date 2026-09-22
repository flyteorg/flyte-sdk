# flyteplugins-typesafe-ai

Run [TypeSafe](https://docs.typesafe.ai/introduction)'s System One model ("Jev")
inside durable Flyte tasks.

Jev does not write text. It answers typed questions — in parallel, in isolation,
and with calibrated confidence attached to every answer. The documented property
that makes it worth building around is that *"adding questions barely changes the
response time"*, so the right move is to ask many small questions in one call and
compose the result in code you can read and change.

This plugin supplies the two things that takes on Flyte: a shape for the answers
that survives a task boundary, and a way to ask a whole battery at once.

```bash
pip install flyteplugins-typesafe-ai
```

## Quickstart

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
    """What is this customer asking for?"""

    REFUND = "refund"
    """they want money back"""
    DELIVERY = "delivery status"
    """they are asking where their order is"""
    OTHER = "something else"


class Severity(enum.IntEnum):
    """How badly are they blocked?"""

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
    # The enums document themselves: class docstring -> question, member docstrings -> criteria.
    intent: Choice[Intent]
    severity: Score[Severity]
    # A Noul has no vocabulary to document itself with, so it carries both.
    hostile: Noul = field(
        metadata={
            "question": "Is the customer hostile?",
            "criteria": {"true": "insults or threats", "false": "civil"},
        }
    )
    money_at_stake: Noul = field(metadata={"question": "Is a payment or refund involved?"})


@env.task
async def handle(ticket: str) -> str:
    t = await ask(Triage, {"ticket": ticket})        # one request, four answers

    if t.hostile.at(0.8):                            # thresholds live in your code
        return "escalate"
    if not t.intent.certain(0.85):
        return "review"
    return f"auto: {t.intent.value.value}, severity {t.severity.value.name}"
```

## The three answer types

| Type | Holds | Useful members |
| --- | --- | --- |
| `Choice[SomeEnum]` | the picked member, `confidence`, `probabilities` | `.certain(threshold)`, `.runner_up()` |
| `Score[SomeIntEnum]` | the picked rung, the unrounded `position`, `confidence` | `.at_least(rung)` |
| `Noul` | truthfulness in 0..1 | `.at(threshold)` |

`Choice` comes back as the **enum member**, not a string, and `Score` keeps both
representations on purpose: `value` is the rung you branch on, `position` is where
on the scale the answer actually landed, which is what you sort and threshold by.

`Noul` deliberately has no `__bool__`. `if noul:` would treat 0.02 and 0.98 alike,
and picking the threshold is the part that belongs in reviewable code.

These are plain dataclasses, so **pydantic is not required** and no bespoke type
transformer exists — they reuse Flyte's built-in `DataclassTransformer`. The
plugin registers them with the type engine through the standard
`flyte.plugins.types` entry point, so `flyte.init()` picks them up, and importing
the package registers them too.

## Declaring questions

The vocabulary documents itself. An enum's **class docstring** is the question and
its **member docstrings** are the criteria, so a documented enum needs nothing at
the call site:

```python
@dataclass
class Triage:
    intent: Choice[Intent]        # question and criteria both come from Intent
    severity: Score[Severity]
```

Override either in ordinary `dataclasses.field` metadata, under two keys named
after the SDK's own arguments:

| key | meaning |
| --- | --- |
| `question` | the instructions for this question |
| `criteria` | the same shape `typesafe_sdk` takes for that question type |

```python
intent: Choice[Intent] = field(metadata={"question": "asked a different way"})
```

`criteria` follows the SDK exactly: a mapping keyed by enum **member name** for a
`Choice`, a **positional** sequence of rungs for a `Score` (so the `IntEnum` must
number its rungs `0..n-1` — a gap is rejected with an error that says so), and
`{"true": ..., "false": ...}` for a `Noul`.

`Noul` is the one that always needs you: it has no vocabulary to document itself
with, so a `Noul` without a question is an error naming the field.

Member docstrings are not stored on the object at runtime — `Severity.NONE.__doc__`
returns the *class* docstring it inherits — so they are read by parsing the source,
the same way pydantic implements `use_attribute_docstrings`. That makes them
best-effort: where the source is not available (a REPL, `exec`, some frozen
deployments) the criterion falls back to the member name rather than failing.

## Batteries can be pydantic models too

A battery is a dataclass *or* a pydantic `BaseModel`; both compile to one call.
On a model, the question comes from `Field(description=...)`, and
`json_schema_extra` carries what a description cannot:

```python
class Triage(BaseModel):
    intent: Choice[Intent]                       # enum docstrings say it all
    severity: Score[Severity] = Field(description="How badly are they blocked?")
    hostile: Noul = Field(
        description="Is the customer hostile?",
        json_schema_extra={"criteria": {"true": "insults or threats", "false": "civil"}},
    )
```

### Plain fields: `bool` and `Literal`

You can also write the Pydantic AI shape — a plain value in the field, calibration
out of band — when that is what you want, for instance to fill a model you already
have:

| field type | asks | comes back as |
| --- | --- | --- |
| `bool` | a Noul | `True`/`False`, cut at a threshold you declare |
| `Literal["a", "b"]` | a Choice over those options | the option itself |

Two rules keep this honest.

**A plain field is only a question when the field says so.** `Choice`, `Score` and
`Noul` are questions by their type alone; a `bool` or `Literal` needs a `question`
in its metadata. Otherwise `flag: bool = False` — bookkeeping far more often than a
question — would start being asked.

**A `bool` must say where it cuts.** The answer is a 0..1 value and something has to
turn it into two states; that decision belongs in your code, so there is no implicit
default:

```python
hostile: bool = field(metadata={"question": "Hostile?", "threshold": 0.8})   # per field
...
facets = await ask(Facets, state, threshold=0.5)     # for every bool that has no own
```

Neither, and it raises — before the request, so you do not pay for a call to find
out. Requiring it also makes a typo loud: `{"treshold": 0.8}` fails rather than
quietly meaning 0.5.

**Nothing is lost from the call, only from the model.** `CallInfo` carries what a
plain field dropped — this plugin's answer to `provider_details`:

```python
facets, info = await ask_with_info(Facets, state, threshold=0.5)

facets.hostile              # False   -- your model
info.values["hostile"]      # 0.62    -- the float it was cut from
info.confidence["tier"]     # 0.77    -- for Literal fields
```

Both maps are `dict[str, float]`, so a task can return a `CallInfo` unchanged.

### How this differs from Pydantic AI

[Pydantic AI's TypeSafe integration](https://pydantic.dev/docs/ai/models/typesafe/)
puts the **question** in the field type: a `bool` field is a yes/no question, a
`Literal` is a pick-one, an `IntEnum` is a rubric. The answers come back as plain
values, and the calibration arrives beside them in `response.provider_details`.

This plugin puts the **answer** in the field type. A field is a `Choice[Intent]`
rather than an `Intent`, so the confidence and the full distribution travel *with*
the value — into your model, through pydantic validators, and across Flyte task
boundaries, with no side channel to carry along. `examples/pydantic_battery.py`
shows a `model_validator` acting on the calibration directly, which is the thing
that gets awkward when it lives in a separate response object.

Neither shape is better in the abstract. Prefer Pydantic AI's when you want a model
of plain values and will consult the calibration once at the call site; prefer this
one when the calibration is part of what downstream code decides on — which is what
routing on confidence means.

## Three ways to ask

`ask()` takes any of these and compiles them into a **single** `system_one` call:

```python
triage = await ask(Triage, state)                      # a battery (dataclass or
                                                       # BaseModel) -> Triage
intent = await ask(Choice[Intent], state)              # one question        -> Choice[Intent]
answers = await ask({"intent": Choice[Intent],         # an ad-hoc battery   -> dict
                     "hostile": Noul}, state)
```

Outside a dataclass there is no field to hang metadata on, so `Annotated` carries
it instead — a mapping, or a bare string when all you have is the question:

```python
await ask(Annotated[Score[Harm], {"question": "How much harm would this do?"}], state)
await ask(Annotated[Noul, "Is this aimed at a specific person?"], state)
```

Both forms work on a dataclass field too. If a field has metadata *and* an
`Annotated` annotation, the field metadata wins — it is the more specific place to
say it.

Prefer one call to several: three separate `ask()` calls are three round trips,
while a battery or a mapping asks everything at once, which is the property the
whole design rests on. Use `ask_with_info()` when you want the model name, question
count, token usage and latency back alongside the answers.

The name collision with `typesafe_sdk.Choice` / `Score` / `Noul` is deliberate and
one-directional: those describe the **question**, these hold the **answer**. You
write the ones in this package; the plugin builds the SDK's from your battery.

## The API key

The SDK reads the key from `TYPESAFE_API_KEY`, so mount your secret as that env
var. There is no helper for this — it is a plain `flyte.Secret`:

```python
env = flyte.TaskEnvironment(
    "triage",
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
)
```

`flyte.Secret` derives `as_env_var` from the key by upper-casing it and swapping
`-` for `_`, so a secret named `TYPESAFE_API_KEY` mounts correctly from
`flyte.Secret(key="TYPESAFE_API_KEY")` alone. Spelling `as_env_var` out is worth
the extra words: it is the string you will grep for when a task cannot find the key.

If your secret is stored under a different name, point the key at it and keep the
mount:

```python
flyte.Secret(key="my-org-typesafe-key", as_env_var="TYPESAFE_API_KEY")
```

Create the secret once:

```bash
flyte create secret TYPESAFE_API_KEY --value <your key>
```

If the key is missing, the failure happens **at the point of use** — in the task
that actually calls System One — with a message naming the declaration and the CLI
command. It is deliberately not an import-time check: a module's tasks are imported
together on the dataplane, so an import-time raise would take down tasks that never
touch System One, and a task that merely passes answers along needs no key at all.

## Examples

- [`examples/triage.py`](examples/triage.py) — fourteen questions in one call, then
  confidence-gated routing in ordinary Python
- [`examples/fanout.py`](examples/fanout.py) — a backlog of tickets, one durable
  Flyte task each, one System One call inside each
- [`examples/single.py`](examples/single.py) — `Choice`, `Score` and `Noul` used on
  their own, without a battery dataclass
- [`examples/pydantic_battery.py`](examples/pydantic_battery.py) — the same battery
  as a pydantic `BaseModel`, with a validator acting on the calibration, and how
  that differs from Pydantic AI

Both run against a cluster:

```bash
flyte run --root-dir plugins/typesafe-ai/examples plugins/typesafe-ai/examples/triage.py handle
```

While this package is unpublished, set `TYPESAFE_LOCAL_WHEELS=1` and build the
wheels first with `make dist && make dist-plugins`.

## Answers as task inputs and outputs

`Choice`, `Score` and `Noul` are plain dataclasses, so Flyte carries them with
nothing registered — including on their own, not just inside a battery:

```python
@env.task
async def classify(message: str) -> Choice[Action]:
    return await ask(Choice[Action], {"message": message})
```

They also get the dict coercion every dataclass input gets, so a caller may pass
`{"value": "refund", "confidence": 0.91}` where a `Choice[Intent]` is expected and
omitted fields fall back to their defaults. Note that an enum nested in a dataclass
is spelled by its **value** (`"refund"`), not its name — that is mashumaro's
convention for dataclass fields, and it differs from the name-based spelling Flyte
uses for a bare enum at the top level.

Inference is never implicit. A dict arriving for a `Choice[Intent]` is the
*serialized answer*, never a state to go ask about — the two are indistinguishable
by shape, and replay depends on the serialized reading winning. If you want the
interface to say "System 1 produces this", make it a task: you get durability,
caching and retries, and the call stays visible in the run graph.

## A note on IntEnum

`Score` takes an `IntEnum` because a rubric is ordered. Flyte's enum transformer
used to accept string-valued enums only; it now supports `IntEnum` as well,
serialized by member name like every other enum, so `severity: Severity` also works
as a bare task input or output. `Flag`/`IntFlag` remain unsupported, with a message
explaining why: a composite member like `READ|WRITE` has a name but cannot be looked
up by it, so it cannot come back.

A second core change makes `Choice[Intent]` work as a task type at all: a
parameterized dataclass is a generic *alias*, which `dataclasses.is_dataclass()`
rejects, so it used to fall through the type engine to pickle. The alias now
resolves to its origin for structural checks while the alias itself is kept for
decoding, which is what binds the type variable.
