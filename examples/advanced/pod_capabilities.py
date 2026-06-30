# /// script
# requires-python = "==3.12"
# dependencies = [
#    "kubernetes",
#    "flyte",
# ]
#
# [tool.uv.sources]
# flyte = { path = "../..", editable = true }
# ///
"""
Pod capability helpers — ``allow_fuse`` / ``allow_nested_sandboxing`` / ``from_spec``.

Demonstrates requesting pod privileges through ``PodTemplate`` capability
helpers instead of hand-writing Kubernetes security contexts, then verifying
*inside the running pods* that the grants actually landed:

1. ``verify_sandboxing`` runs with ``allow_nested_sandboxing()`` — the bwrap
   prerequisite bundle (CAP_SYS_ADMIN + AppArmor unconfined +
   allowPrivilegeEscalation=false). It checks the capability bitmask,
   no_new_privs, the AppArmor profile, and actually exercises the syscalls the
   grant exists for: ``unshare(CLONE_NEWNS)`` (needs CAP_SYS_ADMIN past
   seccomp) and ``unshare(CLONE_NEWUSER)``.
2. ``verify_fuse`` runs with ``from_spec(...).allow_fuse()`` and proves the
   device-cgroup gate is open by ``open("/dev/fuse")`` — the one check that
   fails without ``privileged`` on a stock cluster.

Usage:
    uv run examples/advanced/pod_capabilities.py
"""

import os
import subprocess

from kubernetes.client import V1Container, V1PodSpec

import flyte

# The helpers are new in this SDK version, so bake the local checkout into the
# image (a plain PyPI `flyte` would lack them at runtime too). `kubernetes` is
# not a flyte runtime dependency — this module imports it (for V1PodSpec), and
# the task module is imported in-container, so the image needs it explicitly.
# Pip layer goes BEFORE with_local_v2 so the local SDK wheel wins.
image = (
    flyte.Image.from_debian_base(install_flyte=False, name="pod-capabilities")
    .with_pip_packages("kubernetes")
    .with_local_v2()
)

# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #

# Nested-sandboxing grant: exactly what the bwrap sandbox backend needs.
sandbox_env = flyte.TaskEnvironment(
    name="cap-sandboxing",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)

# FUSE grant, composed onto a user-supplied pod spec via from_spec().
fuse_env = flyte.TaskEnvironment(
    name="cap-fuse",
    image=image,
    resources=flyte.Resources(cpu="500m", memory="512Mi"),
    pod_template=flyte.PodTemplate.from_spec(V1PodSpec(containers=[V1Container(name="primary")])).allow_fuse(),
)

sandbox_env.add_dependency(fuse_env)

# --------------------------------------------------------------------------- #
# In-pod verification helpers
# --------------------------------------------------------------------------- #

_CAP_SYS_ADMIN_BIT = 21
_CLONE_NEWNS = 0x00020000
_CLONE_NEWUSER = 0x10000000


def _proc_status(field: str) -> str:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return line.split(":", 1)[1].strip()
    return ""


def _has_cap_sys_admin(mask_field: str) -> bool:
    mask = int(_proc_status(mask_field) or "0", 16)
    return bool(mask & (1 << _CAP_SYS_ADMIN_BIT))


def _apparmor_profile() -> str:
    try:
        with open("/proc/self/attr/current") as f:
            return f.read().strip().rstrip("\x00") or "unconfined"
    except OSError:
        return "n/a (AppArmor not enabled on this node)"


def _unshare_works(flag: int) -> bool:
    # unshare() alters the calling process, so probe in a throwaway child.
    code = f"import ctypes, sys; sys.exit(0 if ctypes.CDLL(None, use_errno=True).unshare({flag}) == 0 else 1)"
    return subprocess.run(["python", "-c", code], check=False).returncode == 0


@sandbox_env.task
async def verify_sandboxing() -> dict[str, str]:
    """Verify the allow_nested_sandboxing() grants from inside the pod."""
    results = {
        "cap_sys_admin_effective": str(_has_cap_sys_admin("CapEff")),
        "no_new_privs": _proc_status("NoNewPrivs"),  # "1" == allowPrivilegeEscalation: false
        "apparmor_profile": _apparmor_profile(),
        # The syscalls bwrap needs: blocked by the default seccomp profile
        # unless CAP_SYS_ADMIN is present, and by the default AppArmor profile
        # unless unconfined.
        "unshare_mountns": str(_unshare_works(_CLONE_NEWNS)),
        "unshare_userns": str(_unshare_works(_CLONE_NEWUSER)),
    }
    print(f"sandboxing grants: {results}")
    assert results["cap_sys_admin_effective"] == "True", "CAP_SYS_ADMIN missing from effective set"
    assert results["no_new_privs"] == "1", "allowPrivilegeEscalation=false did not land (NoNewPrivs != 1)"
    assert results["unshare_mountns"] == "True", "unshare(CLONE_NEWNS) blocked — seccomp/AppArmor still in the way"
    return results


@fuse_env.task
async def verify_fuse() -> dict[str, str]:
    """Verify the allow_fuse() grants from inside the pod."""
    results = {"dev_fuse_exists": str(os.path.exists("/dev/fuse"))}
    try:
        # The real test: open() on the char device is gated by the device
        # cgroup allowlist, which only `privileged` bypasses on stock clusters.
        fd = os.open("/dev/fuse", os.O_RDWR)
        os.close(fd)
        results["dev_fuse_openable"] = "True"
    except OSError as e:
        results["dev_fuse_openable"] = f"False ({e})"
    results["cap_sys_admin_effective"] = str(_has_cap_sys_admin("CapEff"))
    print(f"fuse grants: {results}")
    assert results["dev_fuse_exists"] == "True", "/dev/fuse hostPath volume did not land"
    assert results["dev_fuse_openable"] == "True", "open(/dev/fuse) failed — device cgroup still blocks it"
    return results


@sandbox_env.task
async def main() -> dict[str, dict[str, str]]:
    return {
        "sandboxing": await verify_sandboxing(),
        "fuse": await verify_fuse(),
    }


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    run.wait()
