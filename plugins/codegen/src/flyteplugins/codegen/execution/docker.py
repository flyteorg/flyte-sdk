import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

import flyte
from flyte.errors import InvalidPackageError
from flyte.io import File
from flyte.sandbox import ImageConfig
from flyte.syncify import syncify

logger = logging.getLogger(__name__)

_PYTEST_COMMAND = r"""
set -o pipefail

EXIT_CODE=1

cleanup() {
    echo "$EXIT_CODE" > /var/outputs/exit_code
    sync
}

trap cleanup EXIT

# $1 = solution file, $2 = test file
PYTHONPATH=/var/inputs python -m pytest "$2" -v --tb=short \
    2>&1 | tee /var/outputs/result

EXIT_CODE=${PIPESTATUS[0]}
"""


@dataclass
class RunResult:
    """Result from running tests in a container."""

    output: str
    exit_code: str
    tests_passed: bool


@flyte.trace
async def build_image(
    language: str,
    base_pkgs: list[str],
    detected_packages: list[str],
    detected_system_packages: list[str],
    previously_installed_packages: list[str],
    previously_installed_system_packages: list[str],
    additional_commands: list[str],
    image_name: str,
    current_image: Optional[str],
    image_config: Optional[ImageConfig],
) -> str:
    """Build image with packages using incremental builds when possible.

    Uses flyte.Image for fresh builds. For incremental builds (when current_image
    exists), adds only new packages as layers on top of the existing image.

    Args:
        language: Programming language (e.g. "python").
        base_pkgs: Base packages required.
        detected_packages: Language packages detected from code.
        detected_system_packages: System packages detected from code.
        previously_installed_packages: Packages already in current_image.
        previously_installed_system_packages: System packages already in current_image.
        additional_commands: Additional RUN commands for image build.
        image_name: Name for the image.
        current_image: Current image URI (if exists) for incremental builds.
        image_config: Image configuration.

    Returns:
        Built image URI string.

    Raises:
        InvalidPackageError: If a system package doesn't exist in apt repos.
    """
    all_packages = base_pkgs + detected_packages
    new_packages = [pkg for pkg in all_packages if pkg not in previously_installed_packages]
    new_system_packages = [pkg for pkg in detected_system_packages if pkg not in previously_installed_system_packages]

    if current_image and (new_packages or new_system_packages):
        logger.info(
            f"Incrementally updating image '{image_name}': adding system={new_system_packages}, language={new_packages}"
        )
        image = flyte.Image.from_base(current_image).clone(name=image_name)
        if new_system_packages:
            image = image.with_apt_packages(*new_system_packages)
        if new_packages and language == "python":
            image = image.with_pip_packages(*new_packages)
    else:
        logger.info(
            f"Building image '{image_name}' with packages: system={detected_system_packages}, language={all_packages}"
        )
        config = image_config or ImageConfig()
        image = flyte.Image.from_debian_base(
            install_flyte=False,
            registry=config.registry,
            registry_secret=config.registry_secret,
            python_version=config.python_version,
            name=image_name,
        )
        apt_packages = list(detected_system_packages)
        if "gcc" not in apt_packages:
            apt_packages.extend(["gcc", "g++", "make"])
        if apt_packages:
            image = image.with_apt_packages(*apt_packages)
        if all_packages and language == "python":
            image = image.with_pip_packages(*all_packages)
        if additional_commands:
            image = image.with_commands(additional_commands)

    try:
        result = await flyte.build.aio(image)
        return result.uri
    except Exception as e:
        error_msg = str(e)
        if "Unable to locate package" in error_msg or "has no installation candidate" in error_msg:
            match = re.search(r"(?:Unable to locate package|Package '?)([\w.+-]+)", error_msg)
            if match:
                logger.error(f"Image build failed: Invalid system package '{match.group(1)}'")
                raise InvalidPackageError(match.group(1), error_msg) from e
        logger.error(f"Image build failed: {error_msg}")
        raise


@syncify
async def run_tests(
    code: str,
    tests: str,
    image: str,
    name: str,
    resources: Optional[flyte.Resources] = None,
    block_network: bool = True,
    retries: int = 0,
    timeout: Optional[int] = None,
    env_vars: Optional[dict[str, str]] = None,
    secrets: Optional[list] = None,
    cache: str = "auto",
    _attempt: int = 1,
) -> RunResult:
    """Run pytest tests against code in an isolated container.

    Args:
        code: Complete Python code to test (including imports).
        tests: Test code string (pytest-compatible).
        image: Pre-built image URI.
        name: Base name for the container task.
        resources: CPU / memory resources for the container.
        block_network: Block outbound network inside the container. Defaults to ``True``.
        retries: Number of task retries on failure.
        timeout: Task timeout in seconds.
        env_vars: Environment variables available inside the container.
        secrets: Flyte secrets to mount.
        cache: Cache behaviour — ``"auto"``, ``"override"``, or ``"disable"``.
        _attempt: Differentiates repeated calls with the same base name.

    Returns:
        :class:`RunResult` with ``output``, ``exit_code``, and ``tests_passed``.
    """
    code_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    code_file.write(f"{code}\n")
    code_file.close()

    test_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    test_file.write(tests)
    test_file.close()

    sandbox = flyte.sandbox.create(
        name=f"{name}-{_attempt}",
        command=["/bin/bash", "-c", _PYTEST_COMMAND],
        arguments=["_", "/var/inputs/solution.py", "/var/inputs/test_solution.py"],
        inputs={"solution.py": File, "test_solution.py": File},
        outputs={"exit_code": str, "result": str},
        resources=resources,
        block_network=block_network,
        retries=retries,
        timeout=timeout,
        env_vars=env_vars,
        secrets=secrets,
        cache=cache,
    )

    try:
        test_exit_code, test_output = await sandbox.run.aio(
            image=image,
            **{
                "solution.py": await File.from_local(
                    code_file.name,
                    hash_method=hashlib.sha256(code.encode()).hexdigest(),
                ),
                "test_solution.py": await File.from_local(
                    test_file.name,
                    hash_method=hashlib.sha256(tests.encode()).hexdigest(),
                ),
            },
        )
        tests_passed = test_exit_code.strip() == "0"
        return RunResult(output=test_output, exit_code=test_exit_code, tests_passed=tests_passed)
    finally:
        for path in (code_file.name, test_file.name):
            try:
                os.unlink(path)
            except OSError:
                pass
