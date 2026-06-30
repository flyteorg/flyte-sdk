"""
Unit tests for flyte.cli._serve module.

Tests for the CLI serve command including the --local flag.
"""

from flyte.cli._serve import ServeArguments


class TestServeArguments:
    """Tests for ServeArguments dataclass."""

    def test_default_values(self):
        """Verify default values for ServeArguments."""
        args = ServeArguments()
        assert args.local is False
        assert args.follow is False
        assert args.copy_style == "loaded_modules"
        assert args.root_dir is None

    def test_local_flag(self):
        """Verify --local flag is captured."""
        args = ServeArguments(local=True)
        assert args.local is True

    def test_from_dict(self):
        """Verify from_dict correctly handles the local flag."""
        args = ServeArguments.from_dict({"local": True, "follow": False, "env_var": []})
        assert args.local is True

    def test_from_dict_with_extra_keys(self):
        """Verify from_dict ignores unknown keys."""
        args = ServeArguments.from_dict({"local": True, "unknown_key": "value"})
        assert args.local is True

    def test_options_include_local(self):
        """Verify options() includes the --local flag."""
        options = ServeArguments.options()
        option_names = [opt.name for opt in options]
        assert "local" in option_names

    def test_env_var_option(self):
        """Verify env-var option."""
        args = ServeArguments(env_var=["KEY=VALUE", "FOO=BAR"])
        assert args.env_var == ["KEY=VALUE", "FOO=BAR"]


class TestServeAppCommandLocal:
    """Tests for local serving via the CLI command."""

    def test_serve_args_local_flag_integration(self):
        """
        GOAL: Verify that ServeArguments with local=True signals local mode.
        """
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from flyte._image import Image
        from flyte.app import AppEnvironment
        from flyte.cli._serve import ServeAppCommand

        # Create an app env with a server decorator
        app_env = AppEnvironment(
            name="cli-test-local-flag",
            image=Image.from_base("python:3.11"),
            port=18095,
        )

        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format, *args):
                pass

        @app_env.server
        def serve_func():
            server = HTTPServer(("127.0.0.1", 18095), TestHandler)
            server.serve_forever()

        serve_args = ServeArguments(local=True)

        # Verify the local flag is set and command is created correctly
        assert serve_args.local is True

        cmd = ServeAppCommand(
            obj_name="cli-test-local-flag",
            obj=app_env,
            serve_args=serve_args,
            help="Test",
        )
        assert cmd.serve_args.local is True
