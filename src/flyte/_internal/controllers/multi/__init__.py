"""Multiprocessing local controller subpackage.

Selected via mode ``"local-multi"`` (CLI flag ``--local-multi``). The
controller dispatches each non-root task body to a worker process from a
``ProcessPoolExecutor``, giving real CPU parallelism for local fan-out
workflows. The root task body itself runs in-process so its
``await child(...)`` calls can fan out across the pool.
"""

from ._controller import LocalMultiController, install_signal_handlers

__all__ = ["LocalMultiController", "install_signal_handlers"]
