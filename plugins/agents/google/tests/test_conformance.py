"""The Google ADK adapter follows the shared agent-adapter format."""

from flyteplugins.agents.core.testing import assert_adapter_conforms

import flyteplugins.agents.google as adapter


def test_google_adapter_conforms():
    assert_adapter_conforms(adapter)
