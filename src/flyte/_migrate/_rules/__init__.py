"""
Migration rules, in the order they run.

Each rule sees the output of the previous one. Order matters: launch plans must be collected before decorators are
rewritten (they become `triggers=`), and workflow bodies must be planned before generic call rewrites touch them.
"""

from typing import List, Type

from .._cst import Rule
from ._calls import CallsRule
from ._image import ImageRule
from ._launch_plan import LaunchPlanRule
from ._leftovers import LeftoversRule
from ._symbols import SymbolsRule
from ._tasks import TasksRule
from ._workflow import WorkflowRule

#: Rules users can select with `--rules` / `--exclude-rules`.
SELECTABLE_RULES: List[Type[Rule]] = [LaunchPlanRule, WorkflowRule, CallsRule, TasksRule, ImageRule, SymbolsRule]

__all__ = ["SELECTABLE_RULES", "LeftoversRule"]
