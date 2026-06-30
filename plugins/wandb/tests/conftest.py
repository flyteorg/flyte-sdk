"""Pytest configuration and fixtures for wandb plugin tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_wandb_run():
    """Create a mock wandb run object."""
    mock_run = MagicMock()
    mock_run.id = "test-run-id"
    mock_run.project = "test-project"
    mock_run.entity = "test-entity"
    mock_run.name = "test-run-name"
    mock_run.tags = ["test-tag"]
    mock_run.config = {"test_param": "test_value"}
    mock_run.url = "https://wandb.ai/test-entity/test-project/runs/test-run-id"
    return mock_run


@pytest.fixture
def mock_flyte_context():
    """Create a mock Flyte context."""
    mock_ctx = MagicMock()
    mock_ctx.data = {}
    mock_ctx.custom_context = {}
    mock_ctx.action = MagicMock()
    mock_ctx.action.name = "test-action"
    mock_ctx.action.run_name = "test-run"
    mock_ctx.action.project = "test-project"
    mock_ctx.action.domain = "development"
    return mock_ctx


@pytest.fixture
def sample_wandb_config_dict():
    """Sample wandb config dictionary."""
    return {
        "wandb_project": "test-project",
        "wandb_entity": "test-entity",
        "wandb_tags": '["tag1", "tag2"]',
        "wandb_config": '{"learning_rate": 0.01}',
        "wandb_mode": "offline",
    }


@pytest.fixture
def sample_wandb_sweep_config_dict():
    """Sample wandb sweep config dictionary."""
    return {
        "wandb_sweep_method": "random",
        "wandb_sweep_metric": '{"name": "loss", "goal": "minimize"}',
        "wandb_sweep_parameters": '{"lr": {"min": 0.001, "max": 0.1}}',
        "wandb_sweep_project": "test-project",
        "wandb_sweep_entity": "test-entity",
    }


@pytest.fixture
def sample_link_context():
    """Sample context dict for link generation."""
    return {
        "wandb_project": "context-project",
        "wandb_entity": "context-entity",
        "_wandb_run_id": "parent-run-123",
    }


@pytest.fixture
def sample_sweep_link_context():
    """Sample context dict for sweep link generation."""
    return {
        "wandb_project": "context-project",
        "wandb_entity": "context-entity",
        "_wandb_sweep_id": "sweep-abc-123",
    }
