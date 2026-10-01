"""
Record and check which modules Flyte loads on its startup-critical paths.

Startup time is dominated by imports, so each profile below pins the set of modules a
path is allowed to load. A profile is a sorted list holding every `flyte.*` module that ends up
in `sys.modules` and the name of every third-party package. The standard library is left out: it
is cheap, and what it loads differs between platforms.

Profiles:

* `flyte`    - `import flyte`, what every user script pays.
* `runtime`  - the `a0` entrypoint executing a task in a container (inputs, controller, outputs).
* `cli_run`  - `flyte run --local file.py task`.

Run through the Makefile, which uses an environment holding only Flyte and its required
dependencies at the versions in `uv.lock`. Optional packages in a development environment, or a
different version of a dependency, would otherwise change the lists:

    make check-import-profile     # fail if any path loads something new (or stopped loading something)
    make update-import-profile    # accept the current module sets
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent.parent / "import_profiles"

# The image is named explicitly: resolving the default image depends on whether the installed Flyte
# is a development build and on whether a local `dist/` folder exists, which would make the
# recorded modules differ between a checkout, a release branch and CI.
_TASK_FILE = """\
import flyte

env = flyte.TaskEnvironment(name="import_profile", image="python:3.13-slim")


@env.task
def main() -> int:
    return 1
"""

# An endpoint nothing listens on. Clients connect lazily, so the paths below complete without a backend.
_ENDPOINT = "localhost:1"

# Runs inside the profiled interpreter, before anything else: dump the loaded modules at exit.
_PREAMBLE = """\
import atexit, json, os, sys

def _dump():
    with open(os.environ["_IMPORT_PROFILE_OUT"], "w") as fh:
        json.dump(sorted(sys.modules), fh)

atexit.register(_dump)
"""

_RUNTIME_ARGS = [
    "a0",
    "--inputs",
    "inputs.pb",
    "--outputs-path",
    "outputs",
    "--version",
    "v1",
    "--run-base-dir",
    "base",
    "--raw-data-path",
    "raw",
    "--name",
    "a0",
    "--run-name",
    "r1",
    "--project",
    "p",
    "--domain",
    "d",
    "--org",
    "o",
    "--resolver",
    "flyte._internal.resolvers.default.DefaultTaskResolver",
    "mod",
    "noop_task",
    "instance",
    "main",
]

_CLI_RUN_ARGS = ["flyte", "--config", "config.yaml", "run", "--local", "noop_task.py", "main"]

PROFILES: dict[str, str] = {
    "flyte": "import flyte",
    "runtime": (
        f"sys.argv = {['a0', *_RUNTIME_ARGS]!r}\nfrom flyte._bin.runtime import _pass_through\n_pass_through()"
    ),
    "cli_run": f"sys.argv = {_CLI_RUN_ARGS!r}\nfrom flyte.cli.main import main\nmain()",
}


# Third-party packages recorded down to their first subpackage, because a single subpackage of
# theirs is expensive enough to keep off these paths (`mashumaro.jsonschema`, unused service stubs).
# Everything else is recorded by its top-level name, so a dependency upgrade that reshuffles its
# internals does not change the profiles.
_DETAILED = frozenset({"flyteidl2", "mashumaro"})


def _prepare_workdir(workdir: Path, profile: str) -> None:
    (workdir / "noop_task.py").write_text(_TASK_FILE)
    (workdir / "inputs.pb").write_bytes(b"")  # an empty Inputs message
    if profile == "cli_run":
        (workdir / "config.yaml").write_text(f"admin:\n  endpoint: dns:///{_ENDPOINT}\n  insecure: true\n")
    for d in ("outputs", "base", "raw"):
        (workdir / d).mkdir()


def _summarize(modules: list[str]) -> list[str]:
    kept = set()
    for name in modules:
        parts = name.split(".")
        top = parts[0]
        if top in sys.stdlib_module_names or top.startswith("_") or top in ("noop_task", "sitecustomize"):
            continue
        if top == "flyte":
            kept.add(name)
        elif top in _DETAILED and len(parts) > 1 and not parts[1].startswith("_"):
            kept.add(".".join(parts[:2]))
        else:
            kept.add(top)
    return sorted(kept)


def collect(profile: str) -> list[str]:
    """Run one profile in a fresh interpreter and return the summary of the modules it loaded."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        _prepare_workdir(workdir, profile)
        out = workdir / "modules.json"
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("FLYTE", "UCTL", "_U_", "_F_", "_UNION")) and k != "PYTHONPATH"
        }
        env.update(
            _IMPORT_PROFILE_OUT=str(out),
            HOME=tmp,  # no user-level config or local run database
        )
        if profile == "runtime":
            # What the platform sets on a task container.
            env.update(
                _U_EP_OVERRIDE=_ENDPOINT,
                _U_INSECURE="1",
                FLYTE_INTERNAL_EXECUTION_PROJECT="p",
                FLYTE_INTERNAL_EXECUTION_DOMAIN="d",
            )
        proc = subprocess.run(
            [sys.executable, "-c", _PREAMBLE + PROFILES[profile]],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"profile {profile!r} exited with {proc.returncode}\n"
                f"--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}"
            )
        if profile == "runtime" and not (workdir / "outputs" / "outputs.pb").exists():
            raise RuntimeError(f"profile 'runtime' did not produce outputs\n--- stderr\n{proc.stderr}")
        return _summarize(json.loads(out.read_text()))


def _profile_path(profile: str) -> Path:
    return PROFILE_DIR / f"{profile}.txt"


def update(profiles: list[str]) -> int:
    for profile in profiles:
        modules = collect(profile)
        _profile_path(profile).write_text("\n".join(modules) + "\n")
        print(f"{profile}: recorded {len(modules)} entries")
    return 0


def check(profiles: list[str]) -> int:
    failed = False
    for profile in profiles:
        modules = collect(profile)
        expected = _profile_path(profile).read_text().split()
        if modules == expected:
            print(f"{profile}: ok, {len(modules)} entries")
            continue
        failed = True
        print(f"{profile}: import profile mismatch")
        diff = difflib.unified_diff(expected, modules, "recorded", "current", lineterm="", n=0)
        print("\n".join(line for line in diff if not line.startswith("@@")))
    if failed:
        print(
            "\nA '+' line is a module this path now loads at startup. Import it inside the function that"
            "\nneeds it instead. If the change is intended, run `make update-import-profile`."
        )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["check", "update"])
    parser.add_argument("--profile", action="append", choices=sorted(PROFILES), help="default: all profiles")
    args = parser.parse_args()
    profiles = args.profile or list(PROFILES)
    return check(profiles) if args.action == "check" else update(profiles)


if __name__ == "__main__":
    sys.exit(main())
