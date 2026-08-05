from types import SimpleNamespace

from flyteplugins.trackio import Trackio


class TestTrackioLink:
    """Tests for the Trackio link class."""

    @staticmethod
    def _mock_context(monkeypatch, *, project=None, server_url=None, space_id=None):
        monkeypatch.setattr(
            "flyteplugins.trackio._link.get_trackio_context",
            lambda: SimpleNamespace(
                project=project,
                server_url=server_url,
                space_id=space_id,
            ),
        )

    def test_self_hosted_project_link(self, monkeypatch):
        """Test self-hosted Trackio project URL generation."""
        self._mock_context(monkeypatch)

        link = Trackio(
            server_url="https://trackio.example.com",
            project="demo",
        )

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://trackio.example.com/projects/demo"

    def test_self_hosted_without_project(self, monkeypatch):
        """Test self-hosted Trackio URL without project."""
        self._mock_context(monkeypatch)

        link = Trackio(server_url="https://trackio.example.com")

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://trackio.example.com"

    def test_space_link(self, monkeypatch):
        """Test Hugging Face Space URL generation."""
        self._mock_context(monkeypatch)

        link = Trackio(space_id="user/my-space")

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://huggingface.co/spaces/user/my-space"

    def test_docs_fallback(self, monkeypatch):
        """Test fallback to Trackio documentation."""
        self._mock_context(monkeypatch)

        link = Trackio()

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://huggingface.co/docs/trackio"

    def test_context_values_are_used(self, monkeypatch):
        """Test values from Trackio context are used."""
        self._mock_context(
            monkeypatch,
            project="ctx-project",
            server_url="https://ctx.example.com",
        )

        link = Trackio()

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://ctx.example.com/projects/ctx-project"

    def test_decorator_values_take_precedence(self, monkeypatch):
        """Test decorator arguments override Trackio context."""
        self._mock_context(
            monkeypatch,
            project="ctx-project",
            server_url="https://ctx.example.com",
            space_id="ctx/space",
        )

        link = Trackio(
            project="decorator-project",
            server_url="https://decorator.example.com",
        )

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://decorator.example.com/projects/decorator-project"

    def test_server_url_takes_precedence_over_space(self, monkeypatch):
        """Test self-hosted server takes precedence over Space."""
        self._mock_context(monkeypatch)

        link = Trackio(
            server_url="https://trackio.example.com",
            space_id="user/my-space",
        )

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://trackio.example.com"

    def test_trailing_slash_is_removed(self, monkeypatch):
        """Test trailing slash is removed from server URL."""
        self._mock_context(monkeypatch)

        link = Trackio(
            server_url="https://trackio.example.com/",
            project="demo",
        )

        uri = link.get_link(
            run_name="run",
            project="flyte-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod",
        )

        assert uri == "https://trackio.example.com/projects/demo"
