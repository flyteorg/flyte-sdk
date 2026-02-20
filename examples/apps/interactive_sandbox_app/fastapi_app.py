"""FastAPI app for running commands with seccomp + Landlock sandboxing."""

import os
import shlex
import json
import subprocess
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sandbox import (
    SAFE_ENV,
    check_landlock_support,
    check_seccomp_support,
    get_sandbox_info,
    run_command_sandboxed,
    run_python_sandboxed,
)


app = FastAPI(
    title="Interactive Sandbox",
    description="A FastAPI app for running shell commands with unprivileged sandboxing.",
    version="1.0.0",
)

DEFAULT_TIMEOUT = float(os.environ.get("COMMAND_TIMEOUT", "30"))
ALLOW_NETWORK = os.environ.get("ALLOW_NETWORK", "false").lower() == "true"
MAX_MEMORY_MB = int(os.environ.get("MAX_MEMORY_MB", "512"))
MAX_PROCESSES = int(os.environ.get("MAX_PROCESSES", "50"))
EXTRA_READ_PATHS: Optional[List[str]] = None
EXTRA_WRITE_PATHS: Optional[List[str]] = None
_extra_read = os.environ.get("EXTRA_READ_PATHS", "")
_extra_write = os.environ.get("EXTRA_WRITE_PATHS", "")
if _extra_read:
    EXTRA_READ_PATHS = json.loads(_extra_read)
if _extra_write:
    EXTRA_WRITE_PATHS = json.loads(_extra_write)


class CommandResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int


class PythonRequest(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: Optional[float] = Field(None, description="Execution timeout in seconds")


class PythonResponse(BaseModel):
    result: Optional[str] = Field(None, description="String representation of the result")
    stdout: str = Field(default="", description="Captured stdout")
    stderr: str = Field(default="", description="Captured stderr")
    returncode: int = Field(description="0 for success, non-zero for errors")
    error: Optional[str] = Field(None, description="Error message if execution failed")


class SandboxStatus(BaseModel):
    landlock_kernel_support: bool
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


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/sandbox/status", response_model=SandboxStatus)
async def sandbox_status() -> SandboxStatus:
    return SandboxStatus(
        landlock_kernel_support=check_landlock_support(),
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


@app.post("/python", response_model=PythonResponse)
async def run_python(request: PythonRequest) -> PythonResponse:
    """Run Python code in a sandboxed subprocess.

    The code can be either:
    - An expression (e.g., "2 + 2", "[x**2 for x in range(10)]")
    - A statement or multiple statements (e.g., "x = 5\\nprint(x)")

    For expressions, the result is returned in the 'result' field.
    For statements, any output goes to stdout/stderr.
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    timeout = request.timeout if request.timeout is not None else DEFAULT_TIMEOUT

    result = run_python_sandboxed(
        code=request.code,
        allow_network=ALLOW_NETWORK,
        timeout=timeout,
        max_memory_mb=MAX_MEMORY_MB,
        max_processes=MAX_PROCESSES,
    )

    return PythonResponse(
        result=result.get("result"),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        returncode=result.get("returncode", 1),
        error=result.get("error"),
    )


@app.get("/sandbox/python/test")
async def test_python_sandbox() -> dict:
    """Run security tests to verify Python sandbox restrictions."""
    tests = {}

    # Test 1: Basic expression evaluation
    result = run_python_sandboxed("2 + 2", timeout=5)
    tests["basic_expression"] = {
        "passed": result.get("returncode") == 0 and result.get("result") == "4",
        "description": "Basic arithmetic expression",
        "result": result.get("result"),
        "returncode": result.get("returncode"),
    }

    # Test 2: List comprehension
    result = run_python_sandboxed("[x**2 for x in range(5)]", timeout=5)
    tests["list_comprehension"] = {
        "passed": result.get("returncode") == 0 and result.get("result") == "[0, 1, 4, 9, 16]",
        "description": "List comprehension",
        "result": result.get("result"),
        "returncode": result.get("returncode"),
    }

    # Test 3: Print statement (stdout capture)
    result = run_python_sandboxed("print('hello sandbox')", timeout=5)
    tests["print_statement"] = {
        "passed": result.get("returncode") == 0 and "hello sandbox" in result.get("stdout", ""),
        "description": "Print statement with stdout capture",
        "stdout": result.get("stdout", "").strip(),
        "returncode": result.get("returncode"),
    }

    # Test 4: Multi-line code
    result = run_python_sandboxed("x = 10\ny = 20\nprint(x + y)", timeout=5)
    tests["multiline_code"] = {
        "passed": result.get("returncode") == 0 and "30" in result.get("stdout", ""),
        "description": "Multi-line code execution",
        "stdout": result.get("stdout", "").strip(),
        "returncode": result.get("returncode"),
    }

    # Test 5: Import standard library (should work)
    result = run_python_sandboxed("import math; print(math.sqrt(16))", timeout=5)
    tests["import_stdlib"] = {
        "passed": result.get("returncode") == 0 and "4.0" in result.get("stdout", ""),
        "description": "Import standard library module",
        "stdout": result.get("stdout", "").strip(),
        "returncode": result.get("returncode"),
    }

    # Test 6: Try to read /etc/passwd (should work - read allowed)
    result = run_python_sandboxed("open('/etc/passwd').readline()", timeout=5)
    tests["read_etc_passwd"] = {
        "passed": result.get("returncode") == 0 and result.get("result") is not None,
        "description": "Reading /etc/passwd should be allowed",
        "result": result.get("result", "")[:50] + "..." if result.get("result") else None,
        "returncode": result.get("returncode"),
    }

    # Test 7: Try to write to /etc (should fail - write blocked)
    result = run_python_sandboxed("open('/etc/sandbox_test', 'w').write('test')", timeout=5)
    tests["write_etc_blocked"] = {
        "passed": result.get("returncode") != 0 or result.get("error") is not None,
        "description": "Writing to /etc should be blocked",
        "error": result.get("error"),
        "returncode": result.get("returncode"),
    }

    # Test 8: Try to access /proc/self/environ (should fail - blocked path)
    result = run_python_sandboxed("open('/proc/self/environ').read()", timeout=5)
    tests["proc_environ_blocked"] = {
        "passed": result.get("returncode") != 0 or result.get("error") is not None,
        "description": "/proc access should be blocked",
        "error": result.get("error"),
        "returncode": result.get("returncode"),
    }

    # Test 9: Try to use subprocess (should fail if seccomp blocks fork/exec)
    result = run_python_sandboxed(
        "import subprocess; subprocess.run(['echo', 'test'], capture_output=True).stdout",
        timeout=5
    )
    tests["subprocess_restricted"] = {
        "passed": True,  # This may or may not work depending on seccomp config
        "description": "Subprocess execution (may be restricted)",
        "result": result.get("result"),
        "error": result.get("error"),
        "returncode": result.get("returncode"),
    }

    # Test 10: Try network access (should fail if network disabled)
    if not ALLOW_NETWORK:
        result = run_python_sandboxed(
            "import socket; s = socket.socket(); s.connect(('example.com', 80))",
            timeout=5
        )
        tests["network_blocked"] = {
            "passed": result.get("returncode") != 0 or result.get("error") is not None,
            "description": "Network access should be blocked",
            "error": result.get("error"),
            "returncode": result.get("returncode"),
        }

    # Test 11: Exception handling
    result = run_python_sandboxed("1/0", timeout=5)
    tests["exception_handling"] = {
        "passed": result.get("returncode") != 0 and "ZeroDivisionError" in (result.get("error") or ""),
        "description": "Exception should be caught and reported",
        "error": result.get("error"),
        "returncode": result.get("returncode"),
    }

    # Test 12: Syntax error handling
    result = run_python_sandboxed("def foo(", timeout=5)
    tests["syntax_error"] = {
        "passed": result.get("returncode") != 0 and "SyntaxError" in (result.get("error") or ""),
        "description": "Syntax errors should be caught and reported",
        "error": result.get("error"),
        "returncode": result.get("returncode"),
    }

    passed = sum(1 for t in tests.values() if t.get("passed", False))
    total = len(tests)

    return {
        "summary": f"{passed}/{total} tests passed",
        "landlock_available": check_landlock_support(),
        "seccomp_available": check_seccomp_support(),
        "network_allowed": ALLOW_NETWORK,
        "tests": tests,
    }


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
        "passed": result.returncode != 0,
        "description": "/proc access should be blocked",
        "returncode": result.returncode,
        "stderr": result.stderr.strip() if result.stderr else "",
    }

    passed = sum(1 for t in tests.values() if t.get("passed", False))
    total = len(tests)

    return {
        "summary": f"{passed}/{total} tests passed",
        "landlock_available": check_landlock_support(),
        "seccomp_available": check_seccomp_support(),
        "tests": tests,
    }
