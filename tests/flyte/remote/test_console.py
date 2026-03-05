from flyte.remote._client.controlplane import Console


class TestConsole:
    """Test the Console class."""

    def test_console_http_domain_dns_localhost_insecure(self):
        """Test localhost DNS endpoint with insecure mode."""
        console = Console("dns:///localhost:8090", insecure=True)
        assert console._http_domain == "http://localhost:8080"

    def test_console_http_domain_http_localhost(self):
        """Test HTTP localhost endpoint."""
        console = Console("http://localhost", insecure=True)
        assert console._http_domain == "http://localhost:8080"

    def test_console_http_domain_dns_secure(self):
        """Test DNS endpoint with secure mode."""
        console = Console("dns:///example.com", insecure=False)
        assert console._http_domain == "https://example.com"

    def test_console_http_domain_https(self):
        """Test HTTPS endpoint."""
        console = Console("https://example.com", insecure=False)
        assert console._http_domain == "https://example.com"

    def test_console_task_url(self):
        """Test task URL construction."""
        console = Console("https://example.com", insecure=False)
        url = console.task_url(project="myproject", domain="development", task_name="mytask")
        assert url == "https://example.com/v2/domain/development/project/myproject/tasks/mytask"

    def test_console_run_url(self):
        """Test run URL construction."""
        console = Console("https://example.com", insecure=False)
        url = console.run_url(project="myproject", domain="development", run_name="run123")
        assert url == "https://example.com/v2/domain/development/project/myproject/runs/run123"

    def test_console_app_url(self):
        """Test app URL construction."""
        console = Console("https://example.com", insecure=False)
        url = console.app_url(project="myproject", domain="development", app_name="myapp")
        assert url == "https://example.com/v2/domain/development/project/myproject/apps/myapp"

    def test_console_trigger_url(self):
        """Test trigger URL construction."""
        console = Console("https://example.com", insecure=False)
        url = console.trigger_url(
            project="myproject", domain="development", task_name="mytask", trigger_name="mytrigger"
        )
        assert url == "https://example.com/v2/domain/development/project/myproject/triggers/mytask/mytrigger"

    def test_console_properties(self):
        """Test Console properties."""
        console = Console("https://example.com", insecure=True)
        assert console.endpoint == "https://example.com"
        assert console.insecure is True

    def test_console_insecure_mode(self):
        """Test Console with insecure mode enabled."""
        console = Console("dns:///example.com", insecure=True)
        assert console._http_domain == "http://example.com"
        url = console.task_url(project="test", domain="dev", task_name="task1")
        assert url.startswith("http://")

    def test_console_secure_mode(self):
        """Test Console with secure mode (default)."""
        console = Console("dns:///example.com")
        assert console._http_domain == "https://example.com"
        url = console.task_url(project="test", domain="dev", task_name="task1")
        assert url.startswith("https://")
