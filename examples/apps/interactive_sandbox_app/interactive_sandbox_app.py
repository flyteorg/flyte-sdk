"""Interactive Sandbox App - A Flyte app for running shell commands with sandboxing.

This app uses seccomp-bpf + Landlock for unprivileged sandboxing that works in
restricted Kubernetes environments without requiring special capabilities.

Security Features:
    - seccomp-bpf: Blocks dangerous syscalls (setuid, mount, ptrace, etc.)
    - Landlock: Restricts filesystem access (requires Linux 5.13+)
    - Resource limits: Prevents CPU/memory/process exhaustion
    - Clean environment: Only safe environment variables passed to commands

Environment Variables:
    - SANDBOX_MODE: "full" (default) or "none" to disable
    - ALLOW_NETWORK: "true" or "false" (default: false)
    - COMMAND_TIMEOUT: timeout in seconds (default: 30)
    - MAX_MEMORY_MB: max memory per command in MB (default: 512)
    - MAX_PROCESSES: max processes per command (default: 50)

API Endpoints:
    - GET /health - Health check
    - GET /sandbox/status - Current sandbox configuration
    - GET /sandbox/info - System sandbox capabilities
    - GET /sandbox/test - Run security tests
    - POST /run?command=... - Run a command
    - POST /run/advanced - Run with custom settings
"""

import flyte
from flyte.app import AppEnvironment

image = (
    flyte.Image.from_debian_base(name="interactive-sandbox-image")
    .with_apt_packages("curl", "ca-certificates", "libseccomp2", "libseccomp-dev")
    .with_commands(["ldconfig"])
    .with_pip_packages("fastapi", "uvicorn", "pyseccomp", "landlock")
)

interactive_sandbox_app = AppEnvironment(
    name="interactive-sandbox-app",
    description="A FastAPI app for running shell commands with sandboxing.",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    include=["fastapi_app.py", "sandbox.py"],
    args=["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"],
)


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app_handle = flyte.serve(interactive_sandbox_app)
    print(f"App URL: {app_handle.url}")
