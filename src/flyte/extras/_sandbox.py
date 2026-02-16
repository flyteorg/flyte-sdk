import hashlib
import json
import logging
import os
import re
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import flyte
from flyte.extras._container import ContainerTask
from flyte.io import Dir, File
from flyte.syncify import syncify

logger = logging.getLogger(__name__)

sandbox_environment = flyte.TaskEnvironment(
    name="sandbox_runtime",
    image=flyte.Image.from_debian_base(
        install_flyte=False
    ),  # Use a minimal base image without Flyte dependencies
)


@dataclass
class ImageConfig:
    """Configuration for Docker image building at runtime."""

    registry: Optional[str] = None
    registry_secret: Optional[str] = None
    python_version: Optional[tuple[int, int]] = None


class InvalidPackageError(Exception):
    """Raised when an invalid system package is detected during image build."""

    def __init__(self, package_name: str, original_error: str):
        self.package_name = package_name
        self.original_error = original_error
        super().__init__(
            f"Invalid system package detected: '{package_name}'. "
            f"This package does not exist in apt repositories. "
            f"Error: {original_error}"
        )


@dataclass
class RunResult:
    """Result from running tests in a container."""

    output: str
    exit_code: str
    tests_passed: bool


@dataclass
class Sandbox:
    """Container environment for running code in isolation.

    Configure the image (packages, resources) once, then call methods
    with different code, inputs and outputs as needed.

    Example::

        sandbox = Sandbox(packages=["pandas", "numpy"])
        image = await sandbox.build.aio()

        task = sandbox.create_task(
            name="process",
            code="import pandas as pd\\n...",
            image=image,
            inputs={"data": File},
            outputs={"result": str},
        )

        result = await sandbox.run_tests.aio(
            code="def add(a, b): return a + b",
            tests="def test_add(): assert add(1, 2) == 3",
            image=image,
        )
        print(result.output, result.exit_code, result.tests_passed)
    """

    packages: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    additional_commands: list[str] = field(default_factory=list)
    resources: Optional[flyte.Resources] = None
    image_config: Optional[ImageConfig] = None
    image_name: Optional[str] = None

    def create_image(self, name: Optional[str] = None) -> flyte.Image:
        """Create an Image spec with pip/apt packages (not yet built).

        Args:
            name: Optional override for image name.

        Returns:
            flyte.Image ready to build.
        """
        spec_name = name or self.image_name or self._default_image_name()
        config = self.image_config or ImageConfig()

        image = flyte.Image.from_debian_base(
            install_flyte=False,
            registry=config.registry,
            registry_secret=config.registry_secret,
            python_version=config.python_version,
            name=spec_name,
        )

        # System packages
        apt_packages = list(self.system_packages)
        if "gcc" not in apt_packages:
            apt_packages.extend(["gcc", "g++", "make"])
        if apt_packages:
            image = image.with_apt_packages(*apt_packages)

        # Pip packages
        if self.packages:
            image = image.with_pip_packages(*self.packages)

        # Additional commands
        if self.additional_commands:
            image = image.with_commands(self.additional_commands)

        return image

    @syncify
    async def build(self) -> str:
        """Build the Docker image and return the image URI.

        Detects invalid system packages and raises InvalidPackageError.

        Returns:
            Built image URI string.

        Raises:
            InvalidPackageError: If a system package doesn't exist in apt repos.
        """
        image = self.create_image()
        try:
            result = await flyte.build.aio(image)
            return result.uri
        except Exception as e:
            error_msg = str(e)
            if (
                "Unable to locate package" in error_msg
                or "has no installation candidate" in error_msg
            ):
                match = re.search(
                    r"(?:Unable to locate package|Package '?)([\w.+-]+)", error_msg
                )
                if match:
                    bad_package = match.group(1)
                    raise InvalidPackageError(bad_package, error_msg) from e
            raise

    def create_task(
        self,
        name: str,
        code: str,
        image: str,
        inputs: Optional[dict[str, type]] = None,
        outputs: Optional[dict[str, type]] = None,
    ) -> Any:
        """Create a ContainerTask that runs code with argparse-style input handling.

        Returns a callable wrapper that automatically provides the script file.
        Outputs are returned as a tuple in the same order as the outputs dict.

        Args:
            name: Name for the container task.
            code: Complete Python code to run (including imports).
            image: Pre-built image URI.
            inputs: Input type declarations (e.g., {"data": File, "threshold": float}).
            outputs: Output type declarations (e.g., {"result": str}).

        Returns:
            Callable task wrapper that returns a tuple of outputs.
        """
        final_inputs = inputs or {}

        # Save code to temp file
        code_file_path = Path(tempfile.gettempdir()) / f"{name}_generated.py"
        code_file_path.write_text(code)
        script_file = str(code_file_path)

        # Build command and arguments
        cli_args = []
        arguments = ["/bin/bash", "/var/inputs/_script"]
        positional_index = 2

        for arg_name, arg_type in final_inputs.items():
            if arg_type in (File, Dir):
                cli_args.extend([f"--{arg_name}", f"${positional_index}"])
                arguments.append(f"/var/inputs/{arg_name}")
                positional_index += 1
            else:
                cli_args.extend([f"--{arg_name}", f"{{{{.inputs.{arg_name}}}}}"])

        python_args = " ".join(cli_args)
        python_cmd = f"python $1 {python_args}" if python_args else "python $1"

        bash_cmd = f"set -o pipefail && {python_cmd}; _exit=$?; mkdir -p /var/outputs; echo $_exit > /var/outputs/exit_code"
        command = ["/bin/bash", "-c", bash_cmd]

        task_inputs = {**final_inputs, "_script": File}

        task_outputs = dict(outputs) if outputs else {}

        task = ContainerTask(
            name=name,
            image=image,
            input_data_dir="/var/inputs",
            output_data_dir="/var/outputs",
            inputs=task_inputs,
            outputs=task_outputs,
            command=command,
            arguments=arguments,
            resources=self.resources or flyte.Resources(cpu=1, memory="1Gi"),
        )

        task.parent_env = weakref.ref(sandbox_environment)
        task.parent_env_name = (
            name  # Use a unique name to avoid reusing the parent environment image
        )

        @syncify
        async def task_wrapper(**kwargs):
            """Wrapper that provides _script and returns tuple."""
            script = await File.from_local(script_file)
            return await task(_script=script, **kwargs)

        task_wrapper._task = task
        return task_wrapper

    @syncify
    async def run(
        self,
        code: str,
        name: Optional[str] = None,
        inputs: Optional[dict[str, type]] = None,
        outputs: Optional[dict[str, type]] = None,
        **kwargs,
    ) -> Any:
        """Full lifecycle: build + create task + execute + collect outputs.

        Args:
            code: Complete Python code to run (including imports).
            name: Base name for the container task.
            inputs: Input type declarations.
            outputs: Output type declarations.
            **kwargs: Input values matching inputs.

        Returns:
            Tuple of typed outputs including exit_code.
        """
        image = await self.build.aio()
        task = self.create_task(
            name=name or f"sandbox-{flyte.ctx().action.name}",
            code=code,
            image=image,
            inputs=inputs,
            outputs=outputs,
        )

        return await task.aio(**kwargs)

    @syncify
    async def run_tests(
        self,
        code: str,
        tests: str,
        name: Optional[str] = None,
        image: Optional[str | flyte.Image] = None,
        _attempt: int = 1,
    ) -> RunResult:
        """Run tests against code in a container.

        Args:
            code: Complete Python code to test (including imports).
            tests: Test code string.
            name: Base name for the test container task.
            image: Pre-built image URI. If None, builds automatically.
            _attempt: Internal parameter for retry logic; not for external use.

        Returns:
            RunResult with output, exit_code, and tests_passed.
        """
        if image is None:
            image = await self.build.aio()

        code_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        code_file.write(f"{code}\n")
        code_file.close()

        test_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        test_file.write(tests)
        test_file.close()

        # Create test container task
        command = [
            "/bin/bash",
            "-c",
            r"""
        set -o pipefail

        EXIT_CODE=1

        cleanup() {
        echo "$EXIT_CODE" > /var/outputs/exit_code
        sync
        }

        trap cleanup EXIT

        # $1 = solution file
        # $2 = test file

        PYTHONPATH=/var/inputs python -m pytest "$2" -v --tb=short \
        2>&1 | tee /var/outputs/result

        EXIT_CODE=${PIPESTATUS[0]}
        """,
        ]

        arguments = [
            "_",
            "/var/inputs/solution.py",
            "/var/inputs/test_solution.py",
        ]

        if name:
            container_name = f"{name}-{flyte.ctx().action.name}"
        else:
            container_name = f"sandbox-test-{flyte.ctx().action.name}-{_attempt}"

        task = ContainerTask(
            name=container_name,
            image=image,
            input_data_dir="/var/inputs",
            output_data_dir="/var/outputs",
            inputs={"solution.py": File, "test_solution.py": File},
            outputs={"exit_code": str, "result": str},
            command=command,
            arguments=arguments,
            resources=self.resources or flyte.Resources(cpu=1, memory="1Gi"),
        )

        task.parent_env = weakref.ref(sandbox_environment)
        task.parent_env_name = container_name  # Use a unique name to avoid reusing the parent environment image

        try:
            test_inputs = {
                "solution.py": await File.from_local(code_file.name),
                "test_solution.py": await File.from_local(test_file.name),
            }
            test_exit_code, test_output = await task(**test_inputs)

            tests_passed = test_exit_code.strip() == "0"
            return RunResult(
                output=test_output, exit_code=test_exit_code, tests_passed=tests_passed
            )
        finally:
            for path in (code_file.name, test_file.name):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _default_image_name(self) -> str:
        spec = {
            "packages": sorted(self.packages),
            "system_packages": sorted(self.system_packages),
        }
        config_hash = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:12]
        return f"sandbox-{config_hash}"
