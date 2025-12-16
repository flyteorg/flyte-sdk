"""A plain Python HTTP server example - the simplest possible app."""

import flyte
import flyte.app
from pathlib import Path

from http.server import HTTPServer, BaseHTTPRequestHandler


class SimpleHandler(BaseHTTPRequestHandler):
    """A simple HTTP server handler."""

    def do_GET(self):

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Hello from Plain Python Server!</h1>")

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')

        else:
            self.send_response(404)
            self.end_headers()


file_name = Path(__file__).name
app_env = flyte.app.AppEnvironment(
    name="plain-python-server",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    args=["python", file_name, "--server"],
    port=8080,
    resources=flyte.Resources(cpu="1", memory="512Mi"),
    requires_auth=False,
)

if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        server = HTTPServer(("0.0.0.0", 8080), SimpleHandler)
        print("Server running on port 8080")
        server.serve_forever()
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        app = flyte.serve(app_env)
        print(f"App URL: {app.url}")
