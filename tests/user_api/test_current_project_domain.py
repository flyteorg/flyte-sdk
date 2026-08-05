from pathlib import Path

import pytest

from flyte import _initialize as init_module
from flyte._initialize import (
    _InitConfig,
    current_domain,
    current_project,
)
from flyte.errors import InitializationError


class TestCurrentProject:
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        init_module._init_config = None
        yield
        init_module._init_config = None

    def test_current_project_when_initialized(self):
        init_module._init_config = _InitConfig(root_dir=Path("/test"), project="my-project")
        assert current_project() == "my-project"

    def test_current_project_raises_when_not_initialized(self):
        with pytest.raises(InitializationError):
            current_project()

    def test_current_project_raises_when_project_none(self):
        init_module._init_config = _InitConfig(root_dir=Path("/test"), project=None)
        with pytest.raises(InitializationError):
            current_project()


class TestCurrentDomain:
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        init_module._init_config = None
        yield
        init_module._init_config = None

    def test_current_domain_when_initialized(self):
        init_module._init_config = _InitConfig(root_dir=Path("/test"), domain="development")
        assert current_domain() == "development"

    def test_current_domain_raises_when_not_initialized(self):
        with pytest.raises(InitializationError):
            current_domain()

    def test_current_domain_raises_when_domain_none(self):
        init_module._init_config = _InitConfig(root_dir=Path("/test"), domain=None)
        with pytest.raises(InitializationError):
            current_domain()
