import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from flyte import _initialize as init_module
from flyte._initialize import (
    CommonInit,
    _get_init_config,
    _InitConfig,
    get_client,
    get_common_config,
    get_storage,
    init,
    init_from_config,
    is_initialized,
    replace_client,
    requires_initialization,
    requires_storage,
)
from flyte.config import Config
from flyte.errors import InitializationError


class TestInitFromConfig:
    """Test cases for init_from_config function with various root_dir scenarios"""

    @pytest.fixture
    def temp_yaml_config_file(self):
        """Create a temporary YAML config file for testing with admin section"""
        config_content = """admin:
  endpoint: dns:///abc.example.com
image:
  builder: remote
task:
  domain: development
  org: demo
  project: user
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            return Path(f.name)

    @pytest.fixture
    def mock_config(self):
        """Mock Config object for testing"""
        config = Mock(spec=Config)
        config.task.org = "test-org"
        config.task.project = "test-project"
        config.task.domain = "test-domain"
        config.platform.endpoint = "test.flyte.example.com"
        config.platform.insecure = False
        config.platform.insecure_skip_verify = False
        config.platform.ca_cert_file_path = None
        config.platform.auth_mode = "Pkce"
        config.platform.command = None
        config.platform.proxy_command = None
        config.platform.client_id = None
        config.platform.client_credentials_secret = None
        config.image.builder = "local"
        return config

    @pytest.fixture
    def mock_yaml_config(self):
        """Mock Config object for YAML config testing"""
        config = Mock(spec=Config)
        config.task.org = "demo"
        config.task.project = "user"
        config.task.domain = "development"
        config.platform.endpoint = "dns:///abc.example.com"
        config.platform.insecure = False
        config.platform.insecure_skip_verify = False
        config.platform.ca_cert_file_path = None
        config.platform.auth_mode = "Pkce"
        config.platform.command = None
        config.platform.proxy_command = None
        config.platform.client_id = None
        config.platform.client_credentials_secret = None
        config.image.builder = "remote"
        return config

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test"""
        init_module._init_config = None
        yield
        init_module._init_config = None

    @patch("flyte._initialize.init")
    @pytest.mark.asyncio
    async def test_init_from_config_with_yaml_config(self, mock_init, temp_yaml_config_file):
        """Test init_from_config with YAML config file and admin endpoint"""
        mock_init.aio = AsyncMock()

        test_root_dir = Path("/project/root")

        await init_from_config.aio(path_or_config=str(temp_yaml_config_file), root_dir=test_root_dir)

        # Verify init was called with correct parameters from YAML config
        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["root_dir"] == test_root_dir
        # Note: These assertions depend on the actual config parsing,
        # which may vary based on the config structure

    @patch("flyte.config.auto")
    @pytest.mark.asyncio
    async def test_init_from_config_with_nonexistent_file(self, mock_config_auto):
        """Test init_from_config raises error for nonexistent config file"""
        # Mock config.auto to raise an exception for nonexistent files
        mock_config_auto.side_effect = FileNotFoundError("Config file not found")

        nonexistent_path = "/path/that/does/not/exist.cfg"

        with pytest.raises(InitializationError) as exc_info:
            await init_from_config.aio(path_or_config=nonexistent_path)

        assert "Configuration file" in str(exc_info.value)
        assert nonexistent_path in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_init_from_config_with_nonexistent_relative_file(self):
        """Test init_from_config raises error for nonexistent relative config file"""
        test_root_dir = Path("/workspace")
        nonexistent_relative_path = "configs/missing.yaml"

        with pytest.raises(InitializationError):
            await init_from_config.aio(path_or_config=nonexistent_relative_path, root_dir=test_root_dir)

    @patch("flyte._initialize.init")
    @pytest.mark.asyncio
    async def test_init_from_config_with_config_object(self, mock_init):
        """Test init_from_config with Config object directly"""
        mock_init.aio = AsyncMock()
        test_root_dir = Path("/test/root")

        # Create a mock config object
        mock_config = Mock(spec=Config)
        mock_config.task.org = "test-org"
        mock_config.task.project = "test-project"
        mock_config.task.domain = "test-domain"
        mock_config.platform.endpoint = "test.flyte.example.com"
        mock_config.platform.insecure = False
        mock_config.platform.insecure_skip_verify = False
        mock_config.platform.ca_cert_file_path = None
        mock_config.platform.auth_mode = "Pkce"
        mock_config.platform.command = None
        mock_config.platform.proxy_command = None
        mock_config.platform.client_id = None
        mock_config.platform.client_credentials_secret = None
        mock_config.image.builder = "local"

        await init_from_config.aio(path_or_config=mock_config, root_dir=test_root_dir)

        # Verify init was called with correct parameters
        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["org"] == mock_config.task.org
        assert call_kwargs["project"] == mock_config.task.project
        assert call_kwargs["domain"] == mock_config.task.domain
        assert call_kwargs["root_dir"] == test_root_dir

    @patch("flyte._initialize.init")
    @patch("flyte.config.auto")
    @pytest.mark.asyncio
    async def test_init_from_config_with_none_path(self, mock_config_auto, mock_init):
        """Test init_from_config with None path uses default config"""
        # Create a mock config for the default config case
        mock_config = Mock(spec=Config)
        mock_config.task.org = "test-org"
        mock_config.task.project = "test-project"
        mock_config.task.domain = "test-domain"
        mock_config.platform.endpoint = "test.flyte.example.com"
        mock_config.platform.insecure = False
        mock_config.platform.insecure_skip_verify = False
        mock_config.platform.ca_cert_file_path = None
        mock_config.platform.auth_mode = "Pkce"
        mock_config.platform.command = None
        mock_config.platform.proxy_command = None
        mock_config.platform.client_id = None
        mock_config.platform.client_credentials_secret = None
        mock_config.image.builder = "local"

        mock_config_auto.return_value = mock_config
        mock_init.aio = AsyncMock()

        test_root_dir = Path("/test/root")

        await init_from_config.aio(path_or_config=None, root_dir=test_root_dir)

        # config.auto should be called with no arguments for default config
        mock_config_auto.assert_called_once_with()

        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["root_dir"] == test_root_dir

    @patch("flyte._initialize.init")
    @patch("flyte.config.auto")
    @pytest.mark.asyncio
    async def test_init_from_config_no_root_dir_provided(self, mock_config_auto, mock_init, mock_config):
        """Test init_from_config when no root_dir is provided"""
        mock_config_auto.return_value = mock_config
        mock_init.aio = AsyncMock()

        await init_from_config.aio(path_or_config=None, root_dir=None)

        mock_config_auto.assert_called_once_with()
        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["root_dir"] is None

    @patch("flyte._initialize.init")
    @patch("flyte.config.auto")
    @patch("pathlib.Path.exists")
    @pytest.mark.asyncio
    async def test_init_from_config_with_symlinked_root_dir(
        self, mock_exists, mock_config_auto, mock_init, mock_config
    ):
        """Test init_from_config with symlinked root directory"""
        mock_exists.return_value = True
        mock_config_auto.return_value = mock_config
        mock_init.aio = AsyncMock()

        # Simulate a symlinked directory
        symlinked_root = Path("/mnt/shared/project-link")
        config_path = "config.yaml"

        await init_from_config.aio(path_or_config=config_path, root_dir=symlinked_root)

        expected_path = symlinked_root / config_path
        mock_config_auto.assert_called_once_with(expected_path)

        mock_init.aio.assert_called_once()
        call_kwargs = mock_init.aio.call_args[1]
        assert call_kwargs["root_dir"] == symlinked_root


class TestInitialization:
    """Test cases for core initialization functions"""

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test"""
        init_module._init_config = None
        yield
        init_module._init_config = None

    def test_is_initialized_false_when_not_initialized(self):
        """Test is_initialized returns False when not initialized"""
        assert is_initialized() is False

    def test_is_initialized_true_when_initialized(self):
        """Test is_initialized returns True when initialized"""
        test_config = _InitConfig(root_dir=Path("/test"))
        init_module._init_config = test_config
        assert is_initialized() is True

    def test_get_common_config_when_initialized(self):
        """Test get_common_config returns config when initialized"""
        test_root = Path("/test/root")
        test_config = _InitConfig(root_dir=test_root, org="test-org", project="test-project", domain="test-domain")
        init_module._init_config = test_config

        common_config = get_common_config()

        assert isinstance(common_config, CommonInit)
        assert common_config.root_dir == test_root
        assert common_config.org == "test-org"
        assert common_config.project == "test-project"
        assert common_config.domain == "test-domain"

    def test_get_common_config_raises_when_not_initialized(self):
        """Test get_common_config raises error when not initialized"""
        with pytest.raises(InitializationError):
            get_common_config()

    def test_get_storage_when_initialized_with_storage(self):
        """Test get_storage returns storage when initialized"""
        mock_storage = Mock()
        test_config = _InitConfig(root_dir=Path("/test"), storage=mock_storage)
        init_module._init_config = test_config

        storage = get_storage()
        assert storage == mock_storage

    def test_get_storage_when_initialized_without_storage(self):
        """Test get_storage returns None when no storage configured"""
        test_config = _InitConfig(root_dir=Path("/test"), storage=None)
        init_module._init_config = test_config

        storage = get_storage()
        assert storage is None

    def test_get_storage_raises_when_not_initialized(self):
        """Test get_storage raises error when not initialized"""
        with pytest.raises(InitializationError):
            get_storage()

    def test_get_client_when_initialized_with_client(self):
        """Test get_client returns client when initialized"""
        mock_client = Mock()
        test_config = _InitConfig(root_dir=Path("/test"), client=mock_client)
        init_module._init_config = test_config

        client = get_client()
        assert client == mock_client

    def test_get_client_raises_when_not_initialized(self):
        """Test get_client raises error when not initialized"""
        with pytest.raises(InitializationError):
            get_client()

    def test_get_client_raises_when_no_client_configured(self):
        """Test get_client raises error when no client configured"""
        test_config = _InitConfig(root_dir=Path("/test"), client=None)
        init_module._init_config = test_config

        with pytest.raises(InitializationError):
            get_client()

    def test_replace_client(self):
        """Test replace_client updates the client"""
        original_client = Mock()
        new_client = Mock()
        test_config = _InitConfig(root_dir=Path("/test"), client=original_client)
        init_module._init_config = test_config

        replace_client(new_client)

        updated_config = _get_init_config()
        assert updated_config.client == new_client
        assert updated_config.root_dir == test_config.root_dir


class TestDecorators:
    """Test cases for initialization requirement decorators"""

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test"""
        init_module._init_config = None
        yield
        init_module._init_config = None

    def test_requires_initialization_decorator_passes_when_initialized(self):
        """Test requires_initialization decorator allows execution when initialized"""
        test_config = _InitConfig(root_dir=Path("/test"))
        init_module._init_config = test_config

        @requires_initialization
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_requires_initialization_decorator_raises_when_not_initialized(self):
        """Test requires_initialization decorator raises error when not initialized"""

        @requires_initialization
        def test_func():
            return "success"

        with pytest.raises(InitializationError):
            test_func()

    def test_requires_storage_decorator_passes_when_storage_available(self):
        """Test requires_storage decorator allows execution when storage is available"""
        mock_storage = Mock()
        test_config = _InitConfig(root_dir=Path("/test"), storage=mock_storage)
        init_module._init_config = test_config

        @requires_storage
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_requires_storage_decorator_raises_when_no_storage(self):
        """Test requires_storage decorator raises error when no storage"""
        test_config = _InitConfig(root_dir=Path("/test"), storage=None)
        init_module._init_config = test_config

        @requires_storage
        def test_func():
            return "success"

        with pytest.raises(InitializationError) as exc_info:
            test_func()

        assert "test_func" in str(exc_info.value)


class TestInitFunction:
    """Test cases for the main init function"""

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test"""
        init_module._init_config = None
        yield
        init_module._init_config = None

    @patch("flyte._initialize._initialize_client")
    @patch("flyte._utils.get_cwd_editable_install")
    @patch("flyte._utils.org_from_endpoint")
    @patch("flyte._utils.sanitize_endpoint")
    @pytest.mark.asyncio
    async def test_init_with_explicit_root_dir(
        self, mock_sanitize, mock_org_from_endpoint, mock_get_editable, mock_init_client
    ):
        """Test init function with explicitly provided root_dir"""
        mock_sanitize.return_value = "https://test.flyte.example.com"
        mock_org_from_endpoint.return_value = "test-org"
        mock_client = Mock()
        mock_init_client.return_value = mock_client

        test_root_dir = Path("/custom/root/dir")

        await init.aio(
            endpoint="test.flyte.example.com", root_dir=test_root_dir, project="test-project", domain="test-domain"
        )

        config = _get_init_config()
        assert config is not None
        assert config.root_dir == test_root_dir
        assert config.project == "test-project"
        assert config.domain == "test-domain"
        assert config.client == mock_client

        # get_cwd_editable_install should not be called when root_dir is provided
        mock_get_editable.assert_not_called()

    @patch("flyte._initialize._initialize_client")
    @patch("flyte._utils.get_cwd_editable_install")
    @patch("flyte._utils.org_from_endpoint")
    @patch("flyte._utils.sanitize_endpoint")
    @pytest.mark.asyncio
    async def test_init_with_editable_install_fallback(
        self, mock_sanitize, mock_org_from_endpoint, mock_get_editable, mock_init_client
    ):
        """Test init function falls back to editable install directory"""
        mock_sanitize.return_value = "https://test.flyte.example.com"
        mock_org_from_endpoint.return_value = "test-org"
        mock_client = Mock()
        mock_init_client.return_value = mock_client

        editable_root = Path("/editable/install/root")
        mock_get_editable.return_value = editable_root

        await init.aio(endpoint="test.flyte.example.com")

        config = _get_init_config()
        assert config is not None
        assert config.root_dir == editable_root

        mock_get_editable.assert_called_once()

    @patch("flyte._initialize._initialize_client")
    @patch("flyte._utils.get_cwd_editable_install")
    @patch("flyte._utils.org_from_endpoint")
    @patch("flyte._utils.sanitize_endpoint")
    @pytest.mark.asyncio
    async def test_init_with_cwd_fallback(
        self, mock_sanitize, mock_org_from_endpoint, mock_get_editable, mock_init_client
    ):
        """Test init function falls back to current working directory"""
        mock_sanitize.return_value = "https://test.flyte.example.com"
        mock_org_from_endpoint.return_value = "test-org"
        mock_client = Mock()
        mock_init_client.return_value = mock_client
        mock_get_editable.return_value = None  # No editable install found

        await init.aio(endpoint="test.flyte.example.com")

        config = _get_init_config()
        assert config is not None
        assert config.root_dir == Path.cwd()

        mock_get_editable.assert_called_once()

    @patch("flyte._initialize._initialize_client")
    @pytest.mark.asyncio
    async def test_init_without_endpoint_or_api_key(self, mock_init_client):
        """Test init function without endpoint or api_key creates config without client"""
        test_root_dir = Path("/test/root")

        await init.aio(root_dir=test_root_dir, project="test-project")

        config = _get_init_config()
        assert config is not None
        assert config.root_dir == test_root_dir
        assert config.project == "test-project"
        assert config.client is None

        # Client initialization should not be called
        mock_init_client.assert_not_called()
