"""FastAPI app for running commands with seccomp + Landlock sandboxing."""

import os
import shlex
import subprocess
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sandbox import (
    LANDLOCK_AVAILABLE,
    SAFE_ENV,
    SECCOMP_AVAILABLE,
    check_landlock_support,
    check_seccomp_support,
    get_sandbox_info,
    run_command_sandboxed,
)


app = FastAPI(
    title="Interactive Sandbox",
    description="A FastAPI app for running shell commands with unprivileged sandboxing.",
    version="1.0.0",
)

SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "full").lower()
DEFAULT_TIMEOUT = float(os.environ.get("COMMAND_TIMEOUT", "30"))
ALLOW_NETWORK = os.environ.get("ALLOW_NETWORK", "false").lower() == "true"
MAX_MEMORY_MB = int(os.environ.get("MAX_MEMORY_MB", "512"))
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", "50"))


class CommandResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int


class SandboxStatus(BaseModel):
    sandbox_mode: str
    landlock_module_available: bool
    landlock_kernel_support: bool
    seccomp_module_available: bool
    seccomp_kernel_support: bool
    allow_network: bool
    default_timeout: float
    max_memory_mb: int
    max_processes: int
    extra_read_paths: Optional[List[str]]
    extra_write_paths: Optional[List[str]]
    safe_environment: dict


class SystemInfo(BaseModel):
    platform: str
    kernel_release: str
    landlock: dict
    seccomp: dict
    resource_limits_available: bool


class CommandRequest(BaseModel):
    command: str = Field(..., description="Shell command to execute")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/sandbox/status", response_model=SandboxStatus)
async def sandbox_status() -> SandboxStatus:
    return SandboxStatus(
        sandbox_mode=SANDBOX_MODE,
        landlock_module_available=LANDLOCK_AVAILABLE,
        landlock_kernel_support=check_landlock_support(),
        seccomp_module_available=SECCOMP_AVAILABLE,
        seccomp_kernel_support=check_seccomp_support(),
        allow_network=ALLOW_NETWORK,
        default_timeout=DEFAULT_TIMEOUT,
        max_memory_mb=MAX_MEMORY_MB,
        max_processes=MAX_PROCESSES,
        extra_read_paths=EXTRA_READ_PATHS,
        extra_write_paths=EXTRA_WRITE_PATHS,
        safe_environment=SAFE_ENV,
    )


@app.get("/sandbox/info", response_model=SystemInfo)
async def sandbox_info() -> SystemInfo:
    info = get_sandbox_info()
    return SystemInfo(**info)


@app.post("/run", response_model=CommandResponse)
async def run_command(command: str) -> CommandResponse:
    """Run a command in a sandboxed subprocess."""
    if not command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")

    if SANDBOX_MODE == "none":
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            shell=False,
            timeout=DEFAULT_TIMEOUT,
            env=SAFE_ENV,
        )
    else:
        result = run_command_sandboxed(
            command=command,
            allow_network=ALLOW_NETWORK,
            timeout=DEFAULT_TIMEOUT,
            max_memory_mb=MAX_MEMORY_MB,
            max_processes=MAX_PROCESSES,
        )

    return CommandResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


@app.post("/run/advanced", response_model=CommandResponse)
async def run_command_advanced(request: CommandRequest) -> CommandResponse:
    """Run a command with custom sandbox settings."""
    if not request.command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")

    timeout = request.timeout or DEFAULT_TIMEOUT
    allow_network = (
        request.allow_network if request.allow_network is not None else ALLOW_NETWORK
    )
    max_memory = request.max_memory_mb or MAX_MEMORY_MB
    max_procs = request.max_processes or MAX_PROCESSES

    if SANDBOX_MODE == "none":
        try:
            result = subprocess.run(
                shlex.split(request.command),
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
                env=SAFE_ENV,
            )
        except subprocess.TimeoutExpired:
            return CommandResponse(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                returncode=-1,
            )
    else:
        result = run_command_sandboxed(
            command=request.command,
            allow_network=allow_network,
            extra_read_paths=request.extra_read_paths,
            extra_write_paths=request.extra_write_paths,
            timeout=timeout,
            max_memory_mb=max_memory,
            max_processes=max_procs,
        )

    return CommandResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


@app.get("/sandbox/test")
async def test_sandbox() -> dict:
    """Run security tests to verify sandbox restrictions."""
    tests = {}

    result = run_command_sandboxed("echo 'sandbox test'", timeout=5)
    tests["basic_execution"] = {
        "passed": result.returncode == 0 and "sandbox test" in result.stdout,
        "stdout": result.stdout.strip(),
        "returncode": result.returncode,
    }

    result = run_command_sandboxed("head -1 /etc/passwd", timeout=5)
    tests["read_allowed_path"] = {
        "passed": result.returncode == 0,
        "description": "Should be able to read /etc/passwd",
        "returncode": result.returncode,
    }

    result = run_command_sandboxed(
        "touch /tmp/sandbox_test && rm /tmp/sandbox_test", timeout=5
    )
    tests["write_tmp"] = {
        "passed": result.returncode == 0,
        "description": "Should be able to write to /tmp",
        "returncode": result.returncode,
    }

    result = run_command_sandboxed("touch /etc/sandbox_test", timeout=5)
    tests["write_etc_blocked"] = {
        "passed": result.returncode != 0,
        "description": "Should NOT be able to write to /etc",
        "returncode": result.returncode,
    }

    if not ALLOW_NETWORK:
        result = run_command_sandboxed(
            "curl -s --connect-timeout 2 https://example.com", timeout=5
        )
        tests["network_blocked"] = {
            "passed": result.returncode != 0,
            "description": "Network access should be blocked",
            "returncode": result.returncode,
        }

    result = run_command_sandboxed("cat /proc/self/environ", timeout=5)
    tests["proc_environ_blocked"] = {
        "passed": result.returncode != 0 or not result.stdout,
        "description": "/proc access should be restricted",
        "returncode": result.returncode,
    }

    passed = sum(1 for t in tests.values() if t.get("passed", False))
    total = len(tests)

    return {
        "summary": f"{passed}/{total} tests passed",
        "sandbox_mode": SANDBOX_MODE,
        "landlock_available": check_landlock_support(),
        "seccomp_available": check_seccomp_support(),
        "tests": tests,
    }
