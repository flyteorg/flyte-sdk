"""Interactive Sandbox App - A Flyte app for running shell commands with sandboxing.

This app uses seccomp-bpf + Landlock for unprivileged sandboxing that works in
restricted Kubernetes environments without requiring special capabilities.

Security Features:
    - seccomp-bpf: Blocks dangerous syscalls (setuid, mount, ptrace, etc.)
    - Landlock: Restricts filesystem access (requires Linux 5.13+)
    - Resource limits: Prevents CPU/memory/process exhaustion
    - Clean environment: Only safe environment variables passed to commands

Environment Variables:
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
"""

import flyte
from flyte.app import AppEnvironment, Parameter

image = (
    flyte.Image.from_debian_base(name="interactive-sandbox-image")
    .with_apt_packages("curl", "ca-certificates", "libseccomp2", "libseccomp-dev")
    .with_pip_packages("fastapi>=0.128.3", "uvicorn>=0.34.0", "pyseccomp", "landlock")
)

interactive_sandbox_app = AppEnvironment(
    name="interactive-sandbox-app-0",
    description="A FastAPI app for running shell commands with sandboxing.",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    include=["fastapi_app.py", "sandbox.py"],
    args=["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"],
    parameters=[
        Parameter(name="extra_read_paths", value="[]", env_var="EXTRA_READ_PATHS"),
        Parameter(name="extra_write_paths", value="[]", env_var="EXTRA_WRITE_PATHS"),
        Parameter(name="max_memory_mb", value="512", env_var="MAX_MEMORY_MB"),
        Parameter(name="max_processes", value="50", env_var="MAX_PROCESSES"),
        Parameter(name="timeout", value="30", env_var="COMMAND_TIMEOUT"),
        Parameter(name="allow_network", value="false", env_var="ALLOW_NETWORK"),
    ],
)


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    app_handle = flyte.serve(interactive_sandbox_app)
    print(f"App URL: {app_handle.url}")

    # Allowed to run: ✅
    # curl -X POST "https://broken-sunset-fcb75.apps.demo.hosted.unionai.cloud/run?command=echo+%27hello%27"

    # Not allowed to run: ❌
    # Network access:
    # curl -X POST "https://broken-sunset-fcb75.apps.demo.hosted.unionai.cloud/run?command=curl+-s+https://example.com"
    #
    # Write to protected path (/etc):
    # curl -X POST "https://broken-sunset-fcb75.apps.demo.hosted.unionai.cloud/run?command=touch+/etc/test_file"
    #
    # Read /proc/self/environ (restricted):
    # curl -X POST "https://broken-sunset-fcb75.apps.demo.hosted.unionai.cloud/run?command=cat+/proc/self/environ"
    #
    # Attempt to mount (blocked syscall):
    # curl -X POST "https://broken-sunset-fcb75.apps.demo.hosted.unionai.cloud/run?command=mount+-t+tmpfs+tmpfs+/mnt"
