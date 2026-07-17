"""The claude adapter must satisfy the shared agent-adapter contract.

The Claude SDK's tool/loop shape is completely different from OpenAI's (in-process
MCP tools, a runtime-owned loop), so passing the *same* conformance check is the
proof that the common format holds across SDKs.
"""

from flyteplugins.agents.core.testing import assert_adapter_conforms

import flyteplugins.agents.claude as adapter


def test_claude_adapter_conforms():
    assert_adapter_conforms(adapter)
