"""The mistral adapter must satisfy the shared agent-adapter contract.

Mistral is a third distinct shape — a server-side Conversations API with a
client-driven dispatch loop — so passing the *same* conformance check confirms
the common format holds across OpenAI, Claude and Mistral alike.
"""

from flyteplugins.agents.core.testing import assert_adapter_conforms

import flyteplugins.agents.mistral as adapter


def test_mistral_adapter_conforms():
    assert_adapter_conforms(adapter)
