"""Unprivileged sandboxing using seccomp-bpf and Landlock.

This module provides sandboxing that works in restricted Kubernetes environments
without requiring special capabilities like SYS_ADMIN or SETFCAP.
"""

import glob
import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    import ctypes.util

    _original_find_library = ctypes.util.find_library

    def _patched_find_library(name):
        result = _original_find_library(name)
        if result is None and name == "seccomp":
            for pattern in [
                "/usr/lib/*/libseccomp.so*",
                "/lib/*/libseccomp.so*",
                "/usr/lib/libseccomp.so*",
                "/lib/libseccomp.so*",
            ]:
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
        return result

    ctypes.util.find_library = _patched_find_library
    import pyseccomp as seccomp

except (ImportError, RuntimeError, OSError):
    pass


SAFE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TERM": "xterm",
}


@dataclass
class ResourceLimits:
    max_processes: int = 64
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    max_memory: int = 1024 * 1024 * 1024  # 1 GB
    max_cpu_time: int = 30
    max_open_files: int = 1024
    max_data_size: int = 512 * 1024 * 1024  # 512 MB
    max_stack_size: int = 8 * 1024 * 1024  # 8 MB


@dataclass
class SandboxConfig:
    read_paths: List[str] = field(default_factory=lambda: ["/tmp", "/var/inputs"])
    write_paths: List[str] = field(default_factory=lambda: ["/tmp", "/var/outputs"])
    read_write_paths: List[str] = field(default_factory=list)
    device_paths: List[str] = field(
        default_factory=lambda: [
            "/dev/null",
            "/dev/zero",
            "/dev/urandom",
            "/dev/random",
        ]
    )
    blocked_path_patterns: List[str] = field(
        default_factory=lambda: [
            "/proc/",
            "/sys/",
            "/run/secrets",
            "/var/run/secrets",
        ]
    )
    use_seccomp_allowlist: bool = True
    allowed_syscalls: Set[str] = field(
        default_factory=lambda: {
            "read",
            "write",
            "readv",
            "writev",
            "pread64",
            "pwrite64",
            "lseek",
            "close",
            "fstat",
            "stat",
            "lstat",
            "fstatat",
            "newfstatat",
            "statx",
            "open",
            "openat",
            "creat",
            "access",
            "faccessat",
            "faccessat2",
            "readlink",
            "readlinkat",
            "getcwd",
            "chdir",
            "fchdir",
            "dup",
            "dup2",
            "dup3",
            "fcntl",
            "flock",
            "truncate",
            "ftruncate",
            "getdents",
            "getdents64",
            "mkdir",
            "mkdirat",
            "rmdir",
            "unlink",
            "unlinkat",
            "rename",
            "renameat",
            "renameat2",
            "link",
            "linkat",
            "symlink",
            "symlinkat",
            "mmap",
            "munmap",
            "mprotect",
            "mremap",
            "brk",
            "madvise",
            "msync",
            "fork",
            "vfork",
            "clone",
            "clone3",
            "execve",
            "execveat",
            "wait4",
            "waitid",
            "exit",
            "exit_group",
            "getpid",
            "getppid",
            "gettid",
            "getuid",
            "getgid",
            "geteuid",
            "getegid",
            "getgroups",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sigaltstack",
            "kill",
            "tgkill",
            "clock_gettime",
            "clock_getres",
            "gettimeofday",
            "nanosleep",
            "clock_nanosleep",
            "poll",
            "ppoll",
            "select",
            "pselect6",
            "epoll_create",
            "epoll_create1",
            "epoll_ctl",
            "epoll_wait",
            "epoll_pwait",
            "epoll_pwait2",
            "eventfd",
            "eventfd2",
            "pipe",
            "pipe2",
            "getrlimit",
            "prlimit64",
            "getrusage",
            "uname",
            "sysinfo",
            "getrandom",
            "futex",
            "set_robust_list",
            "get_robust_list",
            "set_tid_address",
            "arch_prctl",
            "prctl",
            "ioctl",
            "sched_getaffinity",
            "sched_yield",
            "rseq",
        }
    )
    blocked_syscalls: Set[str] = field(
        default_factory=lambda: {
            "setuid",
            "setgid",
            "setreuid",
            "setregid",
            "setresuid",
            "setresgid",
            "setfsuid",
            "setfsgid",
            "capset",
            "capget",
            "init_module",
            "finit_module",
            "delete_module",
            "mount",
            "umount",
            "umount2",
            "pivot_root",
            "sysfs",
            "statfs",
            "fstatfs",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "iopl",
            "ioperm",
            "ioprio_set",
            "settimeofday",
            "clock_settime",
            "adjtimex",
            "clock_adjtime",
            "reboot",
            "kexec_load",
            "kexec_file_load",
            "swapon",
            "swapoff",
            "unshare",
            "setns",
            "add_key",
            "request_key",
            "keyctl",
            "bpf",
            "perf_event_open",
            "userfaultfd",
            "personality",
            "acct",
            "quotactl",
            "quotactl_fd",
            "nfsservctl",
            "lookup_dcookie",
            "vhangup",
            "modify_ldt",
            "vm86",
            "vm86old",
            "seccomp",
            "memfd_create",
            "memfd_secret",
            "io_uring_setup",
            "io_uring_enter",
            "io_uring_register",
        }
    )
    allow_network: bool = False
    network_syscalls: Set[str] = field(
        default_factory=lambda: {
            "socket",
            "socketpair",
            "connect",
            "accept",
            "accept4",
            "bind",
            "listen",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "sendmmsg",
            "recvmmsg",
            "shutdown",
            "getsockname",
            "getpeername",
            "getsockopt",
            "setsockopt",
        }
    )
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    environment: Optional[Dict[str, str]] = None


def check_blocked_paths(command: List[str], blocked_patterns: List[str]) -> Optional[str]:
    """Check if a command references any blocked path patterns.

    Returns the blocked pattern if found, None otherwise.
    """

    command_str = " ".join(command)

    for pattern in blocked_patterns:
        if pattern in command_str:
            return pattern

    return None


def check_landlock_support() -> bool:
    """Check if Landlock is supported by the kernel (Linux 5.13+)."""
    try:
        import platform

        release = platform.release()
        parts = release.split(".")
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1].split("-")[0])
            return (major, minor) >= (5, 13)
        return False
    except Exception:
        return False


def check_seccomp_support() -> bool:
    """Check if seccomp is supported."""
    try:
        return os.path.exists("/proc/sys/kernel/seccomp")
    except Exception:
        return False


def apply_resource_limits(limits: ResourceLimits) -> Dict[str, bool]:
    """Apply resource limits using setrlimit."""
    results = {}
    limit_map = [
        ("max_processes", resource.RLIMIT_NPROC, limits.max_processes),
        ("max_file_size", resource.RLIMIT_FSIZE, limits.max_file_size),
        ("max_memory", resource.RLIMIT_AS, limits.max_memory),
        ("max_cpu_time", resource.RLIMIT_CPU, limits.max_cpu_time),
        ("max_open_files", resource.RLIMIT_NOFILE, limits.max_open_files),
        ("max_data_size", resource.RLIMIT_DATA, limits.max_data_size),
        ("max_stack_size", resource.RLIMIT_STACK, limits.max_stack_size),
    ]

    for name, rlimit_type, value in limit_map:
        try:
            _, hard = resource.getrlimit(rlimit_type)
            if hard == resource.RLIM_INFINITY or value <= hard:
                new_soft = min(value, hard) if hard != resource.RLIM_INFINITY else value
                resource.setrlimit(rlimit_type, (new_soft, hard))
                results[name] = True
            else:
                results[name] = False
        except (ValueError, resource.error, OSError):
            results[name] = False

    return results


def apply_landlock_restrictions(config: SandboxConfig) -> bool:
    """Apply Landlock filesystem restrictions."""

    try:
        import landlock as ll

        ruleset = ll.Ruleset()
        skip_paths = {"/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/fd"}

        read_dir_paths: Set[str] = set()
        read_file_paths: Set[str] = set()
        write_dir_paths: Set[str] = set()
        write_file_paths: Set[str] = set()
        rw_dir_paths: Set[str] = set()
        rw_file_paths: Set[str] = set()

        def collect_paths(paths: List[str], dir_set: Set[str], file_set: Set[str]) -> None:
            for path in paths:
                if path in skip_paths or not os.path.exists(path):
                    continue
                try:
                    real_path = os.path.realpath(path)
                    if real_path.startswith("/proc"):
                        continue
                    if os.path.isdir(real_path):
                        dir_set.add(real_path)
                    else:
                        file_set.add(real_path)
                except OSError:
                    continue

        collect_paths(config.read_paths, read_dir_paths, read_file_paths)
        collect_paths(config.write_paths, write_dir_paths, write_file_paths)
        collect_paths(config.read_write_paths, rw_dir_paths, rw_file_paths)
        collect_paths(config.device_paths, read_dir_paths, read_file_paths)

        paths_added = 0

        read_file_access = ll.FSAccess.ReadFile
        read_dir_access = ll.FSAccess.ReadFile | ll.FSAccess.ReadDir
        write_file_access = ll.FSAccess.WriteFile | ll.FSAccess.Truncate
        write_dir_access = (
            ll.FSAccess.WriteFile
            | ll.FSAccess.Truncate
            | ll.FSAccess.RemoveFile
            | ll.FSAccess.RemoveDir
            | ll.FSAccess.MakeDir
        )
        rw_file_access = read_file_access | write_file_access
        rw_dir_access = read_dir_access | write_dir_access

        for path in sorted(read_dir_paths):
            try:
                ruleset.allow(path, read_dir_access)
                paths_added += 1
            except Exception:
                pass

        for path in sorted(read_file_paths):
            try:
                ruleset.allow(path, read_file_access)
                paths_added += 1
            except Exception:
                pass

        for path in sorted(write_dir_paths):
            try:
                ruleset.allow(path, write_dir_access)
                paths_added += 1
            except Exception:
                pass

        for path in sorted(write_file_paths):
            try:
                ruleset.allow(path, write_file_access)
                paths_added += 1
            except Exception:
                pass

        for path in sorted(rw_dir_paths):
            try:
                ruleset.allow(path, rw_dir_access)
                paths_added += 1
            except Exception:
                pass

        for path in sorted(rw_file_paths):
            try:
                ruleset.allow(path, rw_file_access)
                paths_added += 1
            except Exception:
                pass

        if paths_added == 0:
            print("Warning: No paths could be added to Landlock ruleset", file=sys.stderr)
            return False

        try:
            ruleset.apply()
        except Exception as apply_error:
            if ") = 0" in str(apply_error):
                return True
            raise

        return True

    except Exception as e:
        if ") = 0" in str(e):
            return True
        print(f"Failed to apply Landlock restrictions: {e}", file=sys.stderr)
        return False


def apply_seccomp_restrictions(config: SandboxConfig) -> bool:
    """Apply seccomp syscall filtering."""
    try:
        if config.use_seccomp_allowlist:
            f = seccomp.SyscallFilter(seccomp.ERRNO(1))
            for syscall_name in config.allowed_syscalls:
                try:
                    f.add_rule(seccomp.ALLOW, syscall_name)
                except Exception:
                    pass
            if config.allow_network:
                for syscall_name in config.network_syscalls:
                    try:
                        f.add_rule(seccomp.ALLOW, syscall_name)
                    except Exception:
                        pass
        else:
            f = seccomp.SyscallFilter(seccomp.ALLOW)
            for syscall_name in config.blocked_syscalls:
                try:
                    f.add_rule(seccomp.ERRNO(1), syscall_name)
                except Exception:
                    pass
            if not config.allow_network:
                for syscall_name in config.network_syscalls:
                    try:
                        f.add_rule(seccomp.ERRNO(1), syscall_name)
                    except Exception:
                        pass

        f.load()
        return True

    except Exception as e:
        print(f"Failed to apply seccomp restrictions: {e}", file=sys.stderr)
        return False


def create_sandbox(config: Optional[SandboxConfig] = None) -> dict:
    """Create a sandbox with the given configuration."""
    if config is None:
        config = SandboxConfig()

    results = {
        "landlock_supported": check_landlock_support(),
        "landlock_applied": False,
        "seccomp_supported": check_seccomp_support(),
        "seccomp_applied": False,
        "resource_limits": {},
    }

    results["resource_limits"] = apply_resource_limits(config.resource_limits)

    if results["landlock_supported"]:
        results["landlock_applied"] = apply_landlock_restrictions(config)

    if results["seccomp_supported"]:
        results["seccomp_applied"] = apply_seccomp_restrictions(config)

    return results


def run_sandboxed(
    command: List[str],
    config: Optional[SandboxConfig] = None,
    timeout: Optional[float] = 30.0,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a command in a sandboxed subprocess."""
    if config is None:
        config = SandboxConfig()

    blocked = check_blocked_paths(command, config.blocked_path_patterns)
    if blocked:
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=f"Access denied: path pattern '{blocked}' is blocked by sandbox policy\n",
        )

    env = config.environment if config.environment is not None else SAFE_ENV.copy()

    sandbox_script = f"""
import sys
import os
import resource

limits = [
    (resource.RLIMIT_FSIZE, {config.resource_limits.max_file_size}),
    (resource.RLIMIT_CPU, {config.resource_limits.max_cpu_time}),
    (resource.RLIMIT_NOFILE, {config.resource_limits.max_open_files}),
    (resource.RLIMIT_DATA, {config.resource_limits.max_data_size}),
    (resource.RLIMIT_STACK, {config.resource_limits.max_stack_size}),
]
for rlimit_type, value in limits:
    try:
        soft, hard = resource.getrlimit(rlimit_type)
        new_value = min(value, hard) if hard != resource.RLIM_INFINITY else value
        resource.setrlimit(rlimit_type, (new_value, hard))
    except:
        pass

sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})

try:
    from sandbox import apply_landlock_restrictions, apply_seccomp_restrictions, SandboxConfig

    config = SandboxConfig(
        read_paths={config.read_paths!r},
        write_paths={config.write_paths!r},
        read_write_paths={config.read_write_paths!r},
        device_paths={config.device_paths!r},
        allow_network={config.allow_network!r},
        use_seccomp_allowlist=False,
    )

    try:
        apply_landlock_restrictions(config)
    except Exception as e:
        print(f"Landlock: {{e}}", file=sys.stderr)

    try:
        apply_seccomp_restrictions(config)
    except Exception as e:
        print(f"Seccomp: {{e}}", file=sys.stderr)

except ImportError as e:
    print(f"Sandbox import error: {{e}}", file=sys.stderr)
    sys.exit(1)

import subprocess
cmd = {command!r}
try:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd={cwd!r})
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    sys.exit(proc.returncode)
except Exception as e:
    print(f"Execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        return subprocess.run(
            [sys.executable, "-c", sandbox_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            args=command,
            returncode=-1,
            stdout=e.stdout or "",
            stderr=f"Command timed out after {timeout} seconds\n" + (e.stderr or ""),
        )
    except Exception as e:
        return subprocess.CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr=f"Failed to run sandboxed command: {e}\n",
        )


def run_command_sandboxed(
    command: str,
    allow_network: bool = False,
    extra_read_paths: Optional[List[str]] = None,
    extra_write_paths: Optional[List[str]] = None,
    timeout: float = 30.0,
    cwd: Optional[str] = None,
    max_memory_mb: int = 512,
    max_processes: int = 50,
    use_isolated_tmp: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command in a sandbox.

    Args:
        command: Shell command to execute.
        allow_network: Whether to allow network syscalls.
        extra_read_paths: Additional paths to allow reading.
        extra_write_paths: Additional paths to allow writing.
        timeout: Maximum execution time in seconds.
        cwd: Working directory for the command.
        max_memory_mb: Maximum memory in megabytes.
        max_processes: Maximum number of processes.
        use_isolated_tmp: If True, create an isolated temp directory instead of
            using shared /tmp. This prevents symlink attacks and interference
            between sandboxed processes.
    """
    import shlex
    import tempfile
    import uuid

    resource_limits = ResourceLimits(
        max_memory=max_memory_mb * 1024 * 1024,
        max_processes=max_processes,
        max_cpu_time=int(timeout) + 5,
    )

    config = SandboxConfig(
        allow_network=allow_network,
        resource_limits=resource_limits,
    )

    isolated_tmp_dir = None
    if use_isolated_tmp:
        isolated_tmp_dir = os.path.join(tempfile.gettempdir(), f"sandbox-{uuid.uuid4()}")
        os.makedirs(isolated_tmp_dir, mode=0o700, exist_ok=True)
        config.write_paths = [isolated_tmp_dir]
        if config.environment is None:
            config.environment = SAFE_ENV.copy()
        config.environment["TMPDIR"] = isolated_tmp_dir
        config.environment["HOME"] = isolated_tmp_dir

    if extra_read_paths:
        config.read_paths.extend(extra_read_paths)
    if extra_write_paths:
        config.write_paths.extend(extra_write_paths)

    try:
        config.read_paths.append(os.getcwd())
    except OSError:
        pass

    try:
        return run_sandboxed(
            command=shlex.split(command),
            config=config,
            timeout=timeout,
            cwd=cwd,
        )
    finally:
        if isolated_tmp_dir and os.path.exists(isolated_tmp_dir):
            import shutil

            try:
                shutil.rmtree(isolated_tmp_dir)
            except OSError:
                pass


def check_blocked_paths_in_code(code: str, blocked_patterns: List[str]) -> Optional[str]:
    """Check if Python code references any blocked path patterns.

    Returns the blocked pattern if found, None otherwise.
    """
    for pattern in blocked_patterns:
        if pattern in code:
            return pattern
    return None


def run_python_sandboxed(
    code: str,
    allow_network: bool = False,
    extra_read_paths: Optional[List[str]] = None,
    extra_write_paths: Optional[List[str]] = None,
    timeout: float = 30.0,
    max_memory_mb: int = 512,
    max_processes: int = 50,
) -> Dict:
    """Run Python code in a sandbox.

    The code can be either an expression or statements. For expressions,
    the result is captured and returned. For statements, stdout/stderr
    are captured.

    Args:
        code: Python code to execute.
        allow_network: Whether to allow network syscalls.
        extra_read_paths: Additional paths to allow reading.
        extra_write_paths: Additional paths to allow writing.
        timeout: Maximum execution time in seconds.
        max_memory_mb: Maximum memory in megabytes.
        max_processes: Maximum number of processes.

    Returns:
        Dict with keys: result, stdout, stderr, returncode, error
    """
    import json
    import tempfile
    import uuid

    resource_limits = ResourceLimits(
        max_memory=max_memory_mb * 1024 * 1024,
        max_processes=max_processes,
        max_cpu_time=int(timeout) + 5,
    )

    config = SandboxConfig(
        allow_network=allow_network,
        resource_limits=resource_limits,
    )

    blocked = check_blocked_paths_in_code(code, config.blocked_path_patterns)
    if blocked:
        return {
            "result": None,
            "stdout": "",
            "stderr": f"Access denied: path pattern '{blocked}' is blocked by sandbox policy\n",
            "returncode": 1,
            "error": f"SecurityError: Access to '{blocked}' is blocked by sandbox policy",
        }

    isolated_tmp_dir = os.path.join(tempfile.gettempdir(), f"sandbox-{uuid.uuid4()}")
    os.makedirs(isolated_tmp_dir, mode=0o700, exist_ok=True)
    config.write_paths = [isolated_tmp_dir]
    if config.environment is None:
        config.environment = SAFE_ENV.copy()
    config.environment["TMPDIR"] = isolated_tmp_dir
    config.environment["HOME"] = isolated_tmp_dir

    if extra_read_paths:
        config.read_paths.extend(extra_read_paths)
    if extra_write_paths:
        config.write_paths.extend(extra_write_paths)

    try:
        config.read_paths.append(os.getcwd())
    except OSError:
        pass

    escaped_code = json.dumps(code)

    python_script = f"""
import sys
import os
import resource
import json

limits = [
    (resource.RLIMIT_FSIZE, {config.resource_limits.max_file_size}),
    (resource.RLIMIT_CPU, {config.resource_limits.max_cpu_time}),
    (resource.RLIMIT_NOFILE, {config.resource_limits.max_open_files}),
    (resource.RLIMIT_DATA, {config.resource_limits.max_data_size}),
    (resource.RLIMIT_STACK, {config.resource_limits.max_stack_size}),
]
for rlimit_type, value in limits:
    try:
        soft, hard = resource.getrlimit(rlimit_type)
        new_value = min(value, hard) if hard != resource.RLIM_INFINITY else value
        resource.setrlimit(rlimit_type, (new_value, hard))
    except:
        pass

sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})

try:
    from sandbox import apply_landlock_restrictions, apply_seccomp_restrictions, SandboxConfig

    sandbox_config = SandboxConfig(
        read_paths={config.read_paths!r},
        write_paths={config.write_paths!r},
        read_write_paths={config.read_write_paths!r},
        device_paths={config.device_paths!r},
        allow_network={config.allow_network!r},
        use_seccomp_allowlist=False,
    )

    try:
        apply_landlock_restrictions(sandbox_config)
    except Exception as e:
        print(f"Landlock: {{e}}", file=sys.stderr)

    try:
        apply_seccomp_restrictions(sandbox_config)
    except Exception as e:
        print(f"Seccomp: {{e}}", file=sys.stderr)

except ImportError as e:
    print(f"Sandbox import error: {{e}}", file=sys.stderr)
    sys.exit(1)

from io import StringIO
import contextlib

code = {escaped_code}
result_data = {{"result": None, "error": None}}

stdout_capture = StringIO()
stderr_capture = StringIO()

try:
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        try:
            compiled = compile(code, "<sandbox>", "eval")
            result = eval(compiled)
            result_data["result"] = repr(result)
        except SyntaxError:
            exec(compile(code, "<sandbox>", "exec"))
except Exception as e:
    result_data["error"] = f"{{type(e).__name__}}: {{e}}"

result_data["stdout"] = stdout_capture.getvalue()
result_data["stderr"] = stderr_capture.getvalue()

print("__SANDBOX_RESULT__" + json.dumps(result_data))
"""

    env = config.environment if config.environment is not None else SAFE_ENV.copy()

    try:
        proc_result = subprocess.run(
            [sys.executable, "-c", python_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )

        stdout = proc_result.stdout
        stderr = proc_result.stderr

        if "__SANDBOX_RESULT__" in stdout:
            marker_pos = stdout.find("__SANDBOX_RESULT__")
            json_str = stdout[marker_pos + len("__SANDBOX_RESULT__") :]
            pre_output = stdout[:marker_pos]
            try:
                result_data = json.loads(json_str.strip())
                combined_stdout = pre_output + result_data.get("stdout", "")
                combined_stderr = stderr + result_data.get("stderr", "")
                return {
                    "result": result_data.get("result"),
                    "stdout": combined_stdout,
                    "stderr": combined_stderr,
                    "returncode": 0 if result_data.get("error") is None else 1,
                    "error": result_data.get("error"),
                }
            except json.JSONDecodeError:
                pass

        return {
            "result": None,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc_result.returncode,
            "error": "Failed to parse sandbox result" if proc_result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "result": None,
            "stdout": e.stdout or "",
            "stderr": f"Execution timed out after {timeout} seconds\n" + (e.stderr or ""),
            "returncode": -1,
            "error": f"TimeoutError: Execution timed out after {timeout} seconds",
        }
    except Exception as e:
        return {
            "result": None,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if isolated_tmp_dir and os.path.exists(isolated_tmp_dir):
            import shutil

            try:
                shutil.rmtree(isolated_tmp_dir)
            except OSError:
                pass


def get_sandbox_info() -> dict:
    """Get information about sandbox capabilities on this system."""
    import platform

    return {
        "platform": platform.system(),
        "kernel_release": platform.release(),
        "landlock": {
            "kernel_support": check_landlock_support(),
        },
        "seccomp": {
            "kernel_support": check_seccomp_support(),
        },
        "resource_limits_available": True,
    }


if __name__ == "__main__":
    import json

    print("=== Sandbox System Information ===")
    print(json.dumps(get_sandbox_info(), indent=2))

    print("\n=== Test: Run 'echo hello' ===")
    result = run_command_sandboxed("echo hello")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    print(f"returncode: {result.returncode}")

    print("\n=== Test: Try to read /etc/passwd (should work) ===")
    result = run_command_sandboxed("head -1 /etc/passwd")
    print(f"stdout: {result.stdout}")
    print(f"returncode: {result.returncode}")

    print("\n=== Test: Try to write to /etc (should fail) ===")
    result = run_command_sandboxed("touch /etc/test_file")
    print(f"stderr: {result.stderr}")
    print(f"returncode: {result.returncode}")

    print("\n=== Test: Try to access /proc/self/environ (should fail) ===")
    result = run_command_sandboxed("cat /proc/self/environ")
    print(f"stderr: {result.stderr[:200] if result.stderr else 'none'}...")
    print(f"returncode: {result.returncode}")
    assert result.returncode != 0, "/proc access should be blocked"

    print("\n=== Test: Network access (should fail) ===")
    result = run_command_sandboxed("curl -s https://example.com", timeout=5)
    print(f"stderr: {result.stderr[:200] if result.stderr else 'none'}...")
    print(f"returncode: {result.returncode}")
