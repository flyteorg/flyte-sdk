"""The crewai adapter must satisfy the shared agent-adapter contract.

This is the enforcement: if the adapter drifts from the common format (missing
``tool``/``run_agent``, wrong ``run_agent`` keyword surface, or a tool
task that isn't wired to the resolver), CI fails here.
"""

from flyteplugins.agents.core.testing import assert_adapter_conforms

import flyteplugins.agents.crewai as adapter


def test_crewai_adapter_conforms():
    assert_adapter_conforms(adapter)
