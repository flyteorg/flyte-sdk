"""Tests for AutoCoderAgent construction in flyteplugins.codegen.auto_coder_agent."""

import pytest

from flyteplugins.codegen.auto_coder_agent import AutoCoderAgent


class TestAutoCoderAgentConstruction:
    def test_model_is_required(self):
        """AutoCoderAgent requires a model argument."""
        with pytest.raises(TypeError):
            AutoCoderAgent()

    def test_model_set(self):
        agent = AutoCoderAgent(model="gpt-4.1")
        assert agent.model == "gpt-4.1"

    def test_default_values(self):
        agent = AutoCoderAgent(model="gpt-4.1")
        assert agent.name == "auto-coder"
        assert agent.system_prompt is None
        assert agent.api_key is None
        assert agent.api_base is None
        assert agent.litellm_params is None
        assert agent.base_packages is None
        assert agent.resources is None
        assert agent.image_config is None
        assert agent.skip_tests is False
        assert agent.timeout is None
        assert agent.env_vars is None
        assert agent.secrets is None
        assert agent.cache == "auto"
        assert agent.use_agent_sdk is False

    def test_max_iterations_default(self):
        agent = AutoCoderAgent(model="test")
        assert agent.max_iterations == 10

    def test_sandbox_retries_default(self):
        agent = AutoCoderAgent(model="test")
        assert agent.sandbox_retries == 0

    def test_max_sample_rows_default(self):
        agent = AutoCoderAgent(model="test")
        assert agent.max_sample_rows == 100

    def test_agent_sdk_max_turns_default(self):
        agent = AutoCoderAgent(model="test")
        assert agent.agent_sdk_max_turns == 50

    def test_custom_values(self):
        agent = AutoCoderAgent(
            model="claude-sonnet-4-20250514",
            name="my-agent",
            max_iterations=5,
            sandbox_retries=2,
            skip_tests=True,
            base_packages=["pandas", "numpy"],
            cache="disable",
            use_agent_sdk=True,
            agent_sdk_max_turns=100,
        )
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.name == "my-agent"
        assert agent.max_iterations == 5
        assert agent.sandbox_retries == 2
        assert agent.skip_tests is True
        assert agent.base_packages == ["pandas", "numpy"]
        assert agent.cache == "disable"
        assert agent.use_agent_sdk is True
        assert agent.agent_sdk_max_turns == 100
