"""Prove the composition rules — and the arm parity — without calling any model.

Two checks, both offline.

**1. The composition rules reproduce ground truth.** The composed arms do not ask
a model for a verdict — they ask a dozen atomic questions and *compose* the
verdict in Python. That means the rule itself can be wrong, and if it is, the
benchmark would blame the model for a bug in our code. So: feed each task's rule
the answers a perfect System 1 would give (``EXPECTED_SIGNALS``, which lives next
to the eval cases) and assert it lands on that case's ground-truth label *and*
tool.

**2. The two composed arms are the same pipeline.** ``system2_structured`` exists
to answer "could System 2 have filled that schema itself?", and it can only
answer it if the JSON adapter is faithful — if
:meth:`~tasks._base.TaskSpec.answers_from_json` dropped a signal, mis-clamped a
severity or mis-spelled a label, the challenger would lose for reasons that have
nothing to do with the model, and the benchmark would quietly confirm its own
hypothesis. So: hand the adapter the JSON a *perfect* System 2 would emit and
assert it derives the identical decision the perfect Jev battery derives —
label, tool and routing tier. Any gap between the arms in a real run is then the
model, which is the only thing worth measuring.

Both run in well under a second — no API key, no cluster, no tokens:

    python examples/typesafe_ai/logic_check.py
"""

from __future__ import annotations

import sys

from _system1 import JevDecision
from tasks import TASKS


def expected_signals(task_key: str) -> dict:
    import importlib

    return importlib.import_module(f"tasks.{task_key}").EXPECTED_SIGNALS


def oracle_decision(task, case, expected: tuple) -> JevDecision:
    """What Jev would return for this case if it answered every question right."""
    dec = JevDecision(n_questions=len(task.signals) + 3)
    for signal in task.signals:
        dec.nouls[signal.name] = 1.0 if signal.name in expected else 0.0
    dec.choices["label"] = case.label  # the routing Choice, answered correctly
    dec.confidence["label"] = 0.93
    dec.choices["tool"] = case.tool
    dec.scores["severity"] = 2.0
    dec.confidence["severity"] = 0.85
    return dec


def oracle_json(task, case, expected: tuple) -> dict:
    """The JSON a perfect System 2 would emit for this case.

    Spellings are deliberately mixed — real booleans, ``"yes"``/``"no"`` strings,
    a severity written as ``"2 — significant"`` — because a faithful adapter has
    to survive how models actually write JSON, and a check that only fed it
    pristine values would prove nothing about the arm it is defending.
    """
    # Deliberately heterogeneous: mypy would otherwise infer dict[str, bool] from
    # the first branch and reject the spellings this check exists to exercise.
    signals: dict[str, object] = {}
    for i, signal in enumerate(task.signals):
        hit = signal.name in expected
        if i % 3 == 0:  # real JSON booleans
            signals[signal.name] = hit
        elif i % 3 == 1:  # word spellings
            signals[signal.name] = "yes" if hit else "no"
        else:  # numeric spellings
            signals[signal.name] = 1 if hit else 0
    return {
        "label": case.label,
        "tool": case.tool,
        "severity": "2 — significant",
        "confidence": 0.93,
        "signals": signals,
    }


def check_arm_parity(verbose: bool = True) -> int:
    """Assert the structured arm derives exactly what the Jev arm derives."""
    failures = 0
    for task_key, task in TASKS.items():
        table = expected_signals(task_key)
        if verbose:
            print(f"== {task.label} — arm parity")
        for case in task.cases:
            expected = table.get(case.id, ())
            jev = task.derive(oracle_decision(task, case, expected))
            s2 = task.derive(task.answers_from_json(oracle_json(task, case, expected)))
            same = (jev.label, jev.tool, jev.route) == (s2.label, s2.tool, s2.route)
            # The battery must also arrive intact: a perfect answer sheet that the
            # adapter scores as incomplete would understate the challenger.
            filled = task.count_returned(oracle_json(task, case, expected))
            complete = filled == task.battery_size()
            ok = same and complete
            failures += not ok
            if verbose or not ok:
                print(
                    f"  {'ok  ' if ok else 'FAIL'} {case.id:>4}  "
                    f"jev=({jev.label}, {jev.tool}, {jev.route})  "
                    f"s2=({s2.label}, {s2.tool}, {s2.route})  "
                    f"battery={filled}/{task.battery_size()}"
                )
    return failures


def check(verbose: bool = True) -> int:
    failures = 0
    for task_key, task in TASKS.items():
        table = expected_signals(task_key)
        if verbose:
            print(f"== {task.label} ({len(task.signals) + 3} questions per call)")
        for case in task.cases:
            decision = task.derive(oracle_decision(task, case, table.get(case.id, ())))
            ok = decision.label == case.label and decision.tool == case.tool
            failures += not ok
            if verbose or not ok:
                print(
                    f"  {'ok  ' if ok else 'FAIL'} {case.id:>4}  "
                    f"{decision.label:<16} (want {case.label:<16}) "
                    f"{decision.tool:<18} (want {case.tool:<18}) "
                    f"route={decision.route:<9} {decision.reason}"
                )
    print(f"\n{'all composition rules reproduce ground truth' if not failures else f'{failures} rule failures'}")
    return failures


def main(verbose: bool = True) -> int:
    failures = check(verbose)
    print()
    parity = check_arm_parity(verbose)
    verdict = (
        "both composed arms derive identically from a perfect battery" if not parity else f"{parity} parity failures"
    )
    print(f"\n{verdict}")
    return failures + parity


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
