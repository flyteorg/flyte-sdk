"""Prove the composition rules are right, without calling any model.

The with-System-1 arm does not ask Jev for a verdict — it asks a dozen atomic
questions and *composes* the verdict in Python.  That means the rule itself can
be wrong, and if it is, the benchmark would blame the model for a bug in our
code.  So: feed each task's composition rule the answers a perfect System 1
would give (``EXPECTED_SIGNALS``, which lives next to the eval cases) and assert
it lands on that case's ground-truth label *and* tool.

Runs offline in well under a second — no API key, no cluster, no tokens:

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


if __name__ == "__main__":
    sys.exit(1 if check() else 0)
