import datetime
import hashlib
import json
import logging
import os
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import litellm
import pandas as pd
from pydantic import BaseModel, Field

import flyte
import flyte.errors
from flyte.extras import ContainerTask
from flyte.io import Dir, File
from flyte.syncify import syncify

logger = logging.getLogger(__name__)


# Create a TaskEnvironment for all code-gen tasks
# Users should add this as depends_on to their main TaskEnvironment
code_gen_environment = flyte.TaskEnvironment(
    name="code_gen_runtime",
    image=flyte.Image.from_debian_base(install_flyte=False),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

__all__ = [
    "code_gen_eval",
    "CodeGenEvalResult",
    "ImageConfig",
    "code_gen_environment",
    "CodePlan",
    "CodeSolution",
    "ErrorDiagnosis",
]


@dataclass
class ImageConfig:
    """Configuration for Docker image building at runtime."""

    registry: Optional[str] = None
    registry_secret: Optional[str] = None
    python_version: Optional[tuple[int, int]] = None


class CodePlan(BaseModel):
    """Structured plan for the code solution."""

    description: str = Field(description="Overall description of the solution")
    approach: str = Field(
        description="High-level approach and algorithm to solve the problem"
    )


class CodeSolution(BaseModel):
    """Structured code solution with dependencies and main code separated."""

    language: str = Field(
        default="python",
        description="Programming language (python)",
    )
    dependencies: str = Field(
        default="",
        description="Import statements or dependency declarations (separated from main code)",
    )
    code: str = Field(
        default="",
        description="Main code without dependency declarations (should be executable when combined with dependencies)",
    )
    system_packages: list[str] = Field(
        default_factory=list,
        description="System packages needed (e.g., gcc, build-essential, curl)",
    )


class CodeGenEvalResult(BaseModel):
    """Result from code generation and evaluation."""

    plan: Optional[CodePlan] = None
    solution: CodeSolution
    tests: Optional[str] = None
    success: bool
    output: str
    exit_code: int
    error: Optional[str] = None
    attempts: int = 1
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    detected_packages: list[str] = Field(
        default_factory=list,
        description="Language packages detected by LLM from imports",
    )
    detected_system_packages: list[str] = Field(
        default_factory=list, description="System packages detected by LLM"
    )
    image: Optional[str] = Field(
        default=None,
        description="The Flyte Image built with all dependencies",
    )
    total_input_tokens: int = Field(
        default=0,
        description="Total input tokens used across all LLM calls",
    )
    total_output_tokens: int = Field(
        default=0,
        description="Total output tokens used across all LLM calls",
    )
    declared_inputs: Optional[dict[str, type]] = Field(
        default=None,
        description="Input types (user-provided or inferred from data)",
    )
    declared_outputs: Optional[dict[str, type]] = Field(
        default=None,
        description="Output types declared by user",
    )
    data_context: Optional[str] = Field(
        default=None,
        description="Extracted data context (schema, stats, patterns, samples) used for code generation",
    )
    original_data: Optional[dict[str, File]] = Field(
        default=None,
        description="Original data converted to Files (for run() method)",
    )

    def as_container_task(
        self,
        name: str = "run_code_on_real_data",
        resources: Optional[flyte.Resources] = None,
    ):
        """Create a ContainerTask that runs the generated code in a container.

        The generated code must write outputs to /var/outputs/{output_name} files.
        Exit code is automatically captured and added to outputs.
        Returns a callable wrapper that automatically provides the script file.

        Args:
            name: Name for the container task
            resources: Optional resources for the task

        Returns:
            Callable task wrapper - Call with your declared inputs.
            Returns tuple of (declared_outputs..., exit_code: int).
            Exit code is always included as the last output automatically.
        """
        if not self.success:
            raise ValueError("Cannot create task from failed code generation")

        if not self.image:
            raise ValueError(
                "No image available - code generation did not build an image"
            )

        final_inputs = self.declared_inputs

        # Validate input types
        supported_input_types = (str, int, float, bool, File, Dir)
        for input_key, input_type in final_inputs.items():
            if input_type not in supported_input_types:
                supported_names = [t.__name__ for t in supported_input_types]
                raise ValueError(
                    f"Unsupported input type for '{input_key}': {input_type}. "
                    f"ContainerTask only supports: {', '.join(supported_names)}"
                )

        # Combine dependencies and code
        full_code = f"{self.solution.dependencies}\n\n{self.solution.code}"

        # Save code to temp file (will be uploaded as File input to container)
        code_file_path = Path(tempfile.gettempdir()) / f"{name}_generated.py"
        code_file_path.write_text(full_code)
        script_file = str(code_file_path)

        # Build command and arguments based on input types
        # - File/Dir inputs: use path syntax (/var/inputs/name) with positional args ($N)
        # - Primitive inputs: use template syntax ({{.inputs.name}})

        cli_args = []
        arguments = ["/bin/bash", "/var/inputs/_script"]  # $0, $1
        positional_index = 2  # Start at $2 (after bash and script)

        for arg_name, arg_type in final_inputs.items():
            if arg_type in (File, Dir):
                # File/Dir: use positional argument with path syntax
                cli_args.extend([f"--{arg_name}", f"${positional_index}"])
                arguments.append(f"/var/inputs/{arg_name}")
                positional_index += 1
            else:
                # Primitives (str, int, float, bool): use template syntax
                cli_args.extend([f"--{arg_name}", f"{{{{.inputs.{arg_name}}}}}"])

        # Build python command
        python_args = " ".join(cli_args)
        python_cmd = f"python $1 {python_args}" if python_args else "python $1"

        # Wrap in bash to capture exit code automatically
        bash_cmd = f"set -o pipefail && {python_cmd}; echo $? > /var/outputs/exit_code"
        command = ["/bin/bash", "-c", bash_cmd]

        # Add script as a File input
        task_inputs = {**final_inputs, "_script": File}

        # Validate and prepare output types
        task_outputs = dict(self.declared_outputs) if self.declared_outputs else {}

        # Validate output types
        supported_output_types = (
            str,
            int,
            float,
            bool,
            datetime.datetime,
            datetime.timedelta,
            File,
            Dir,
        )
        for output_key, output_type in task_outputs.items():
            if output_type not in supported_output_types:
                supported_names = [t.__name__ for t in supported_output_types]
                raise ValueError(
                    f"Unsupported output type for '{output_key}': {output_type}. "
                    f"ContainerTask only supports: {', '.join(supported_names)}"
                )

        # Automatically add exit_code to outputs
        if "exit_code" not in task_outputs:
            task_outputs["exit_code"] = int

        task = ContainerTask(
            name=name,
            image=self.image,
            input_data_dir="/var/inputs",
            output_data_dir="/var/outputs",
            inputs=task_inputs,
            outputs=task_outputs,
            command=command,
            arguments=arguments,
            resources=resources or flyte.Resources(cpu=1, memory="1Gi"),
        )

        # Associate with code_gen_environment
        task.parent_env = weakref.ref(code_gen_environment)
        task.parent_env_name = code_gen_environment.name

        # Create wrapper that automatically provides the script
        def task_wrapper(**kwargs):
            """Wrapper that automatically provides _script input."""
            return task(_script=File.from_local_sync(script_file), **kwargs)

        # Preserve task instance for inspection
        task_wrapper._task = task

        return task_wrapper

    async def run(self, **override_data) -> dict[str, any]:
        """Run generated code on original data (one-off execution).

        This is a convenience method for running the code immediately on the data
        provided during code_gen_eval(). If you reuse the generated code, use as_container_task().

        Args:
            **override_data: Optional data to override original_data values

        Returns:
            Dict of typed outputs including exit_code
        """
        if not self.success:
            raise ValueError("Cannot run failed code generation")

        if not self.original_data:
            raise ValueError(
                "No original data available. "
                "Use as_container_task() for manual data passing."
            )

        # Merge original data with any overrides
        run_data = {**self.original_data, **override_data}

        # Create and run task
        task = self.as_container_task()

        # Execute task with refreshed data
        result_tuple = await task(**run_data)

        # Parse results into dict
        output_names = (
            list(self.declared_outputs.keys()) if self.declared_outputs else []
        )
        output_names.append("exit_code")  # Always included

        outputs = {}
        for i, name in enumerate(output_names):
            outputs[name] = (
                result_tuple[i] if isinstance(result_tuple, tuple) else result_tuple
            )

        return outputs


# Apply syncify after class definition to avoid Pydantic field detection
CodeGenEvalResult.run = syncify(CodeGenEvalResult.run)


class TestFailure(BaseModel):
    """Individual test failure with diagnosis."""

    test_name: str = Field(description="Name of the failing test")
    error_message: str = Field(
        description="The exact final error message from test output (e.g., 'RecursionError: maximum recursion depth exceeded')"
    )
    expected_behavior: str = Field(description="What this test expected to happen")
    actual_behavior: str = Field(description="What actually happened when the code ran")
    root_cause: str = Field(
        description="Why the test failed (quote the EXACT code that's wrong)"
    )
    suggested_fix: str = Field(
        description="Specific code changes using format: Replace `current code` with `new code`"
    )
    error_type: str = Field(
        description="Type of error: 'environment' (missing packages/dependencies), 'logic' (bug in solution code), or 'test_error' (bug in test code)"
    )


class ErrorDiagnosis(BaseModel):
    """Structured diagnosis of execution errors."""

    failures: list[TestFailure] = Field(
        description="Individual test failures with their diagnoses"
    )
    needs_system_packages: list[str] = Field(
        default_factory=list,
        description="System packages needed (e.g., gcc, pkg-config).",
    )
    needs_language_packages: list[str] = Field(
        default_factory=list,
        description="Language packages needed.",
    )
    needs_additional_commands: list[str] = Field(
        default_factory=list,
        description="Additional RUN commands (e.g., apt-get update, mkdir /data, wget files).",
    )


class FixVerification(BaseModel):
    """Verification that fixes were applied to code."""

    all_fixes_applied: bool = Field(
        description="True if ALL suggested fixes are present in the new code"
    )
    applied_fixes: list[str] = Field(
        default_factory=list,
        description="List of fixes that were successfully applied (by test name)",
    )
    missing_fixes: list[str] = Field(
        default_factory=list,
        description="List of fixes that are still missing (by test name)",
    )
    explanation: str = Field(
        description="Brief explanation of what was checked and what's missing (if anything)"
    )


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


class _PackageDetectionResponse(BaseModel):
    """Response format for LLM package detection."""

    packages: list[str] = Field(
        default_factory=list,
        description="List of third-party package names",
    )


class _TestCodeResponse(BaseModel):
    """Response format for LLM test generation."""

    test_code: str = Field(description="Complete test code")


FILE_EXTENSIONS = {"python": ".py"}

PACKAGE_MANAGER_MAP = {"python": "pip package names (excluding standard library)"}

TEST_FRAMEWORKS = {
    "python": {
        "name": "pytest",
        "packages": ["pytest"],
        "system_packages": [],
        "command": "python -m pytest",
    }
}

DEFAULT_SYSTEM_PROMPT = (
    """You are a coding assistant that generates high-quality code in {language}."""
)

STRUCTURED_OUTPUT_REQUIREMENTS = """
IMPORTANT: You must structure your response with:
1. description: Brief explanation of what the code does
2. language: The programming language used
3. dependencies: Only dependency declarations - no other code
4. code: The main code (without dependency declarations, but should be executable when combined with dependencies)
5. system_packages: List of system packages needed (e.g., ["gcc", "build-essential", "curl"]). Leave empty if none needed.

Ensure all code is complete, executable, and follows best practices for the chosen language."""


def _create_image_spec(
    language: str,
    packages: Optional[list[str]],
    system_packages: Optional[list[str]],
    additional_commands: Optional[list[str]],
    image_name: Optional[str],
    image_config: Optional[ImageConfig],
) -> flyte.Image:
    """Create an Image for the specified language with dependencies.

    Args:
        language: Programming language for the container
        packages: Language packages to install in the container image
        system_packages: System packages to install (e.g., gcc, build-essential)
        additional_commands: Additional commands to run during image build
        image_name: Optional custom name for the image (defaults to "code-gen-{language}")
        image_config: Image configuration (registry, registry_secret, python_version)

    Returns:
        Image (not yet built)
    """
    # Start with base image using builder pattern
    spec_name = image_name or f"code-gen-{language}"
    config = image_config or ImageConfig()

    image = flyte.Image.from_debian_base(
        install_flyte=False,
        registry=config.registry,
        registry_secret=config.registry_secret,
        python_version=config.python_version,
        name=spec_name,
    )

    # Add system packages first
    apt_packages = list(system_packages or [])

    # Add common system packages for Python
    if language == "python" and "gcc" not in apt_packages:
        apt_packages.extend(["gcc", "g++", "make"])

    if apt_packages:
        image = image.with_apt_packages(*apt_packages)

    # Add language-specific packages
    packages_to_install = list(packages or [])

    if language == "python":
        # Python: use with_pip_packages
        if packages_to_install:
            image = image.with_pip_packages(*packages_to_install)

    # Add any additional custom commands
    if additional_commands:
        image = image.with_commands(additional_commands)

    return image


def _create_container_task(
    language: str,
    image: flyte.Image | str,
    container_resources: Optional[flyte.Resources],
) -> ContainerTask:
    """Create a ContainerTask.

    Args:
        language: Programming language for the container
        image: Image to use
        container_resources: Resources for the container

    Returns:
        ContainerTask configured with the provided image
    """
    # Get test framework info for the language
    test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])

    command_parts = []
    arguments = []

    if language == "python":
        # Build test command for Python
        test_input_name = "test_solution.py"
        code_input_name = "solution.py"

        # Set PYTHONPATH so tests can import from solution module
        # Separate file references in arguments so both get mounted
        command_parts = [
            "/bin/bash",
            "-c",
            f"set -o pipefail && PYTHONPATH=/var/inputs {test_framework_info['command']} $2 -v --tb=short 2>&1 | tee /var/outputs/result; echo ${{PIPESTATUS[0]}} > /var/outputs/exit_code",
        ]
        arguments = [
            "/bin/bash",
            f"/var/inputs/{code_input_name}",
            f"/var/inputs/{test_input_name}",
        ]

    resources = container_resources or flyte.Resources(cpu=1, memory="1Gi")

    # Add parent action name to container name
    container_name = f"code-eval-{language}-{flyte.ctx().action.name}"

    return ContainerTask(
        name=container_name,
        image=image,
        input_data_dir="/var/inputs",
        output_data_dir="/var/outputs",
        inputs={code_input_name: File, test_input_name: File},
        outputs={"result": str, "exit_code": str},
        command=command_parts,
        arguments=arguments,
        resources=resources,
    )


def _extract_token_usage(response) -> tuple[int, int]:
    """Extract token usage from LLM response.

    Args:
        response: LiteLLM response object

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    try:
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        return input_tokens, output_tokens
    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")
        return 0, 0


def _is_dataframe(obj) -> bool:
    """Check if object is a pandas DataFrame."""
    try:
        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


def _extract_dataframe_context(df, name: str, max_sample_rows: int = 5) -> str:
    """Extract comprehensive context from DataFrame.

    Args:
        df: pandas DataFrame
        name: Name of the data input
        max_sample_rows: Number of sample rows to include

    Returns:
        Formatted string with all extracted context
    """
    context_parts = []

    # 1. Structural Context
    context_parts.append(f"## Data: {name}")
    context_parts.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    context_parts.append(f"\nColumn Types:")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        context_parts.append(f"  - {col}: {dtype} ({null_pct:.1f}% null)")

    # 2. Statistical Context
    context_parts.append(f"\nStatistical Summary:")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        context_parts.append("  Numeric columns:")
        desc = df[numeric_cols].describe()
        for col in numeric_cols:
            stats = desc[col]
            context_parts.append(
                f"    {col}: min={stats['min']:.2g}, max={stats['max']:.2g}, "
                f"mean={stats['mean']:.2g}, median={stats['50%']:.2g}"
            )

    # Categorical/Object columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        context_parts.append("  Categorical columns:")
        for col in cat_cols:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            if unique_count <= 20 and total_count > 0:
                # Show value counts for low-cardinality columns
                top_values = df[col].value_counts().head(5)
                top_str = ", ".join([f"'{k}': {v}" for k, v in top_values.items()])
                context_parts.append(
                    f"    {col}: {unique_count} unique values. Top 5: {{{top_str}}}"
                )
            else:
                context_parts.append(f"    {col}: {unique_count} unique values")

    # DateTime columns
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(date_cols) > 0:
        context_parts.append("  DateTime columns:")
        for col in date_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            context_parts.append(f"    {col}: {min_date} to {max_date}")

    # 3. Behavioral Context (patterns, invariants)
    context_parts.append(f"\nData Patterns:")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        context_parts.append(
            f"  - {dup_count:,} duplicate rows ({dup_count/len(df)*100:.1f}%)"
        )

    # Check for potential ID columns
    for col in df.columns:
        if df[col].nunique() == len(df) and not df[col].isna().any():
            context_parts.append(f"  - '{col}' appears to be a unique identifier")
            break

    # 4. Representative Samples
    context_parts.append(f"\nRepresentative Samples ({max_sample_rows} rows):")

    # Sample strategy: first few + random + edge cases
    sample_indices = []

    # First rows
    sample_indices.extend(range(min(2, len(df))))

    # Random sample
    if len(df) > max_sample_rows:
        remaining = max_sample_rows - len(sample_indices)
        random_indices = df.sample(n=remaining).index.tolist()
        sample_indices.extend(random_indices)
    else:
        sample_indices = list(range(len(df)))

    sample_df = df.iloc[sample_indices[:max_sample_rows]]

    # Format as CSV
    context_parts.append(sample_df.to_csv(index=False))

    return "\n".join(context_parts)


async def _extract_file_context(file: File, name: str, max_sample_rows: int = 5) -> str:
    """Extract comprehensive context from File.

    Args:
        file: File to extract context from
        name: Name of the data input
        max_sample_rows: Number of sample rows to include

    Returns:
        Formatted string with all extracted context
    """
    import pandas as pd

    local_path = await file.download()

    # Detect file type and extract accordingly
    file_ext = Path(local_path).suffix.lower()

    try:
        # Try to parse as structured data
        if file_ext in [".csv", ".tsv"]:
            # CSV/TSV
            delimiter = "\t" if file_ext == ".tsv" else ","
            df = pd.read_csv(
                local_path, delimiter=delimiter, nrows=10000
            )  # Limit to 10k rows for analysis
            return _extract_dataframe_context(df, name, max_sample_rows)

        elif file_ext in [".parquet", ".pq"]:
            # Parquet
            df = pd.read_parquet(local_path)
            if len(df) > 10000:
                df = df.sample(n=10000)  # Sample for analysis
            return _extract_dataframe_context(df, name, max_sample_rows)

        elif file_ext == ".json":
            # JSON - try as JSON lines first, then as single JSON
            try:
                df = pd.read_json(local_path, lines=True, nrows=10000)
                return _extract_dataframe_context(df, name, max_sample_rows)
            except:
                df = pd.read_json(local_path)
                if isinstance(df, pd.DataFrame):
                    return _extract_dataframe_context(df, name, max_sample_rows)
                else:
                    # Single JSON object - show structure
                    context_parts = [
                        f"## Data: {name}",
                        f"Type: JSON object",
                        f"Keys: {list(df.keys()) if hasattr(df, 'keys') else 'N/A'}",
                        f"\nStructure preview:",
                        str(df)[:500],
                    ]
                    return "\n".join(context_parts)

        elif file_ext in [".xlsx", ".xls"]:
            # Excel
            df = pd.read_excel(local_path, nrows=10000)
            return _extract_dataframe_context(df, name, max_sample_rows)

        else:
            # Text file or unknown - extract basic stats
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[:1000]  # First 1000 lines

            context_parts = [
                f"## Data: {name}",
                f"Type: Text file ({file_ext})",
                f"Lines: {len(lines)}",
                f"\nFirst {max_sample_rows} lines:",
                "".join(lines[:max_sample_rows]),
            ]
            return "\n".join(context_parts)

    except Exception as e:
        # Fallback: treat as text
        logger.warning(
            f"Failed to parse {name} as structured data: {e}. Treating as text."
        )
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(2000)

        context_parts = [
            f"## Data: {name}",
            f"Type: Text/Binary ({file_ext})",
            f"Size: {Path(local_path).stat().st_size:,} bytes",
            f"\nContent preview:",
            content,
        ]
        return "\n".join(context_parts)


async def extract_data_context(
    data: dict[str, any],
    max_sample_rows: int = 5,
) -> str:
    """Extract comprehensive context from data inputs.

    Extracts:
    1. Structural context (schema, types, shape)
    2. Statistical context (distributions, ranges)
    3. Behavioral context (patterns, invariants)
    4. Operational context (scale, nulls)
    5. Representative samples

    Args:
        data: Dict of data inputs (File or DataFrame)
        max_sample_rows: Number of sample rows to include

    Returns:
        Formatted string with all data context
    """
    context_parts = []

    for name, value in data.items():
        if isinstance(value, File):
            context = await _extract_file_context(value, name, max_sample_rows)
            context_parts.append(context)
        elif _is_dataframe(value):
            context = _extract_dataframe_context(value, name, max_sample_rows)
            context_parts.append(context)
        else:
            context_parts.append(
                f"## Data: {name}\nType: {type(value)}\n(Unsupported type)"
            )

    return "\n\n" + "=" * 80 + "\n\n".join(context_parts)


def _build_enhanced_prompt(
    prompt: str,
    language: str,
    schema: Optional[str],
    constraints: Optional[list[str]],
    data_samples: Optional[str],
    inputs: Optional[dict[str, type]],
    outputs: Optional[dict[str, type]],
) -> str:
    """Build enhanced prompt with language, schema, constraints, data samples, inputs, and outputs."""
    enhanced_prompt = f"Language: {language}\n\n{prompt}"

    if schema:
        enhanced_prompt += f"\n\nSchema:\n```\n{schema}\n```"

    # Always add script requirement first, then user constraints
    script_constraint = (
        "REQUIRED: Your code must be a runnable Python script with a "
        "if __name__ == '__main__': block that parses command line arguments using argparse. "
    )

    # Add CLI argument requirement based on declared inputs
    if inputs:
        # Build argument list from declared inputs
        args_list = []
        for name, param_type in inputs.items():
            type_name = (
                param_type.__name__
                if hasattr(param_type, "__name__")
                else str(param_type)
            )
            # Clarify that File/Dir inputs are received as string paths
            if "File" in type_name or "Dir" in type_name:
                args_list.append(f"--{name} (str): path to {type_name.lower()}")
            else:
                args_list.append(f"--{name} ({type_name})")
        args_spec = ", ".join(args_list)
        script_constraint += f"Accept these command line arguments: {args_spec}. "

        # Add explicit instruction for File/Dir handling
        has_file_inputs = any(
            "File" in str(t) or "Dir" in str(t) for t in inputs.values()
        )
        if has_file_inputs:
            script_constraint += "File/Dir arguments are STRING PATHS - use them directly with open() or other file operations."
    elif data_samples:
        script_constraint += (
            "Accept appropriate command line arguments to process the data samples."
        )
    else:
        script_constraint += "Include appropriate command line arguments if needed."

    all_constraints = [script_constraint]

    # Add output requirement based on declared outputs
    if outputs:
        output_constraint = ""
        output_parts = []
        for name, output_type in outputs.items():
            type_name = (
                output_type.__name__
                if hasattr(output_type, "__name__")
                else str(output_type)
            )
            if type_name in ("str", "int", "float", "bool"):
                output_parts.append(
                    f"Write {name} ({type_name}) to /var/outputs/{name}"
                )
            elif "File" in type_name:
                output_parts.append(
                    f"Write {name} file path (absolute path as string) to /var/outputs/{name}"
                )
            elif "Dir" in type_name:
                output_parts.append(
                    f"Write {name} directory path (absolute path as string) to /var/outputs/{name}"
                )
            else:
                output_parts.append(
                    f"Write {name} ({type_name}) to /var/outputs/{name}"
                )
        output_constraint += "; ".join(output_parts)
        all_constraints.append(output_constraint)

    if constraints:
        all_constraints.extend(constraints)

    enhanced_prompt += "\n\nConstraints:\n" + "\n".join(
        f"- {c}" for c in all_constraints
    )

    if data_samples:
        enhanced_prompt += f"\n\nData samples:\n```\n{data_samples}\n```"

    return enhanced_prompt


@flyte.trace
async def generate_plan(
    model: str,
    prompt: str,
    language: str,
    schema: Optional[str],
    constraints: Optional[list[str]],
    data_samples: Optional[str],
    inputs: Optional[dict[str, type]],
    outputs: Optional[dict[str, type]],
    litellm_params: Optional[dict],
) -> tuple[CodePlan, int, int]:
    """Generate a structured plan for the code solution before writing code.

    Returns:
        Tuple of (plan, input_tokens, output_tokens)
    """
    base_prompt = _build_enhanced_prompt(
        prompt, language, schema, constraints, data_samples, inputs, outputs
    )

    # Add planning-specific instructions
    planning_prompt = f"""You are planning a {language} solution for the following task:

{base_prompt}

Create a detailed plan including:
1. Overall description of the solution
2. High-level approach and algorithm

The solution will be implemented as a single {language} file.
Focus on clarity and completeness. The plan will guide code generation."""

    # Build params with defaults
    params = {
        "model": model,
        "messages": [{"role": "user", "content": planning_prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }

    # Merge litellm_params (can override anything except response_format)
    params.update(litellm_params or {})

    # Always set response_format last
    params["response_format"] = CodePlan

    try:
        response = await litellm.acompletion(**params)
    except Exception as e:
        # Check if it's an unsupported params error
        if "UnsupportedParamsError" in type(
            e
        ).__name__ or "does not support parameters" in str(e):
            raise flyte.errors.RuntimeUserError(
                f"Model '{model}' does not support structured outputs (response_format parameter). "
                f"Please use a model that supports structured outputs like: gpt-4.1, "
                f"claude-3-5-sonnet, or similar models."
            ) from e
        raise

    # Extract token usage
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        plan_dict = json.loads(content)
        return CodePlan(**plan_dict), input_tokens, output_tokens
    return content, input_tokens, output_tokens


@flyte.trace
async def generate_code(
    model: str,
    conversation: list[dict],
    plan: Optional[CodePlan],
    litellm_params: Optional[dict],
    is_retry: bool = False,
) -> tuple[CodeSolution, int, int]:
    """Generate code with structured output using Pydantic model.

    Args:
        model: LLM model to use
        conversation: Message history
        plan: Optional plan to guide code generation (only used for initial generation)
        litellm_params: LiteLLM parameters
        is_retry: If True, skip plan context (for debugging/fixing code)

    Returns:
        Tuple of (CodeSolution, input_tokens, output_tokens)
    """
    # Add plan context only for initial generation (not retries)
    messages = conversation.copy()
    if plan and not is_retry:
        plan_context = f"""
Plan for implementation:
- Description: {plan.description}
- Approach: {plan.approach}

Follow this plan when generating the code."""

        # Insert plan context before the last user message
        if messages and messages[-1]["role"] == "user":
            messages.insert(-1, {"role": "system", "content": plan_context})
        else:
            messages.append({"role": "system", "content": plan_context})

    # Build params with defaults
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
    }

    # Merge litellm_params (can override anything except response_format)
    params.update(litellm_params or {})

    # Always set response_format last
    params["response_format"] = CodeSolution

    response = await litellm.acompletion(**params)

    # Extract token usage
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        solution_dict = json.loads(content)
        return CodeSolution(**solution_dict), input_tokens, output_tokens
    return content, input_tokens, output_tokens


async def detect_required_packages(
    model: str,
    dependencies: str,
    language: str,
    litellm_params: Optional[dict],
) -> tuple[list[str], int, int]:
    """Use LLM to detect required packages from dependency declarations.

    Returns:
        Tuple of (packages, input_tokens, output_tokens)
    """
    if not dependencies.strip():
        return [], 0, 0

    package_type = PACKAGE_MANAGER_MAP.get(
        language.lower(), "package names for the package manager"
    )

    detection_prompt = f"""Given these {language} dependency declarations:

```{language}
{dependencies}
```

List the {package_type} needed to install these dependencies.
For standard library / built-in modules, don't include them.
Only include third-party packages that need to be installed."""

    # Build params with defaults
    params = {
        "model": model,
        "messages": [{"role": "user", "content": detection_prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    # Merge litellm_params (can override anything except response_format)
    params.update(litellm_params or {})

    # Always set response_format last
    params["response_format"] = _PackageDetectionResponse

    response = await litellm.acompletion(**params)

    # Extract token usage
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        result_dict = json.loads(content)
        packages = result_dict.get("packages", [])
    else:
        packages = content.packages

    return packages, input_tokens, output_tokens


async def _detect_and_track_packages(
    model: str,
    solution: CodeSolution,
    base_pkgs: list[str],
    detected_packages: list[str],
    detected_system_packages: list[str],
    litellm_params: Optional[dict] = None,
) -> tuple[bool, list[str], list[str], int, int]:
    """Detect packages from solution and track them.

    Returns:
        (needs_rebuild, updated_detected_packages, updated_detected_system_packages, input_tokens, output_tokens)
    """
    needs_rebuild = False
    total_input_tokens = 0
    total_output_tokens = 0

    # Detect system packages from LLM
    if (
        solution.system_packages
        and solution.system_packages != detected_system_packages
    ):
        detected_system_packages = solution.system_packages.copy()
        logger.info(f"Detected system packages: {detected_system_packages}")
        needs_rebuild = True

    # Detect language packages from dependencies
    if solution.dependencies.strip():
        new_packages, in_tok, out_tok = await detect_required_packages(
            model, solution.dependencies, solution.language, litellm_params
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if new_packages:
            added_packages = []
            for pkg in new_packages:
                if pkg not in detected_packages and pkg not in base_pkgs:
                    detected_packages.append(pkg)
                    added_packages.append(pkg)

            if added_packages:
                logger.info(
                    f"Detected new {solution.language} packages: {added_packages}"
                )
                needs_rebuild = True

    return (
        needs_rebuild,
        detected_packages,
        detected_system_packages,
        total_input_tokens,
        total_output_tokens,
    )


@flyte.trace
async def generate_tests(
    model: str,
    prompt: str,
    plan: CodePlan,
    solution: CodeSolution,
    constraints: Optional[list[str]] = None,
    schema: Optional[str] = None,
    data_samples: Optional[str] = None,
    inputs: Optional[dict[str, type]] = None,
    outputs: Optional[dict[str, type]] = None,
    litellm_params: Optional[dict] = None,
) -> tuple[str, int, int]:
    """Generate test code to validate the solution.

    Returns:
        Tuple of (test_code, input_tokens, output_tokens)
    """
    full_code = f"{solution.dependencies}\n\n{solution.code}"

    # Get test framework for the language
    language = solution.language.lower()
    test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])
    test_framework = test_framework_info["name"]

    # Build test generation prompt for Python
    import_instruction = (
        "Import functions/classes from solution module (e.g., 'from solution import function_name'). "
        "The code file is named solution.py."
    )
    test_prompt = f"""
You are generating TESTS.

Task:
{prompt}

Plan:
- Description: {plan.description}
- Approach: {plan.approach}"""

    # Add schema if provided
    if schema:
        test_prompt += f"""
Schema:
{schema}

"""

    # Add constraints if provided
    if constraints:
        test_prompt += """
Constraints:
"""
        for i, constraint in enumerate(constraints, 1):
            test_prompt += f"{i}. {constraint}\n"

    # Add data samples if provided
    if data_samples:
        test_prompt += f"""
Data:
Use EXACTLY this format if referenced.
Do NOT invent additional samples.

{data_samples}

"""

    # Add inputs/outputs info
    if inputs:
        test_prompt += f"\n\nCLI Arguments: {list(inputs.keys())}"
    if outputs:
        test_prompt += f"\n\nExpected outputs: {list(outputs.keys())}"

    test_prompt += f"""
Solution code:
```{solution.language}
{full_code}
```"""

    test_prompt += f"""
Test generation instructions:

Generate comprehensive test code ensuring MAXIMUM COVERAGE using {test_framework} to validate this solution.

Requirements:
1. Use {test_framework} framework
2. Test the FULL execution path end-to-end (not just isolated functions)
3. Use provided data AS-IS
4. {import_instruction}
5. Follow {test_framework} best practices
6. Assume /var/outputs exists and is writable

IMPORTANT: Do NOT wrap output in code fences."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 2000,
        "temperature": 0.4,
    }
    params.update(litellm_params or {})
    params["response_format"] = _TestCodeResponse

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        test_dict = json.loads(content)
        test_code = test_dict["test_code"]
    else:
        test_code = content.test_code

    return test_code, input_tokens, output_tokens


async def _build_image_and_task(
    solution: CodeSolution,
    base_pkgs: list[str],
    detected_packages: list[str],
    detected_system_packages: list[str],
    previously_installed_packages: list[str],
    previously_installed_system_packages: list[str],
    additional_commands: list[str],
    container_resources: Optional[flyte.Resources],
    image_name: str,
    current_image: Optional[str],
    image_config: Optional[ImageConfig],
) -> tuple[str, ContainerTask]:
    """Build image with packages and create container task.

    Uses incremental building: if current_image exists, adds only new packages as layers.

    Returns:
        (built_image, container_task)
    """
    # Calculate what's new
    all_packages = base_pkgs + detected_packages
    new_packages = [
        pkg for pkg in all_packages if pkg not in previously_installed_packages
    ]
    new_system_packages = [
        pkg
        for pkg in detected_system_packages
        if pkg not in previously_installed_system_packages
    ]

    if current_image and (new_packages or new_system_packages):
        # Incremental build: start from existing image, add only new packages
        logger.info(
            f"Incrementally updating image '{image_name}': "
            f"adding system={new_system_packages}, language={new_packages}"
        )
        image = flyte.Image.from_base(current_image).clone(name=image_name)

        # Add new system packages
        if new_system_packages:
            image = image.with_apt_packages(*new_system_packages)

        # Add new language packages
        if new_packages:
            if solution.language == "python":
                image = image.with_pip_packages(*new_packages)
    else:
        # First build: start from base image with all packages
        logger.info(
            f"Building image '{image_name}' with packages: "
            f"system={detected_system_packages}, language={all_packages}"
        )
        image = _create_image_spec(
            language=solution.language,
            packages=all_packages,
            system_packages=detected_system_packages,
            additional_commands=additional_commands,
            image_name=image_name,
            image_config=image_config,
        )

    try:
        built_image = await flyte.build.aio(image)
    except Exception as e:
        error_msg = str(e)

        # Check if this is a package not found error
        # Common patterns: "Unable to locate package {name}" or "Package '{name}' has no installation candidate"
        if (
            "Unable to locate package" in error_msg
            or "has no installation candidate" in error_msg
        ):
            # Try to extract package name from error message
            import re

            # Pattern: "Unable to locate package csv" or "Package 'csv' has no..."
            match = re.search(r"(?:Unable to locate package|Package ')(\w+)", error_msg)
            if match:
                bad_package = match.group(1)
                logger.error(
                    f"Image build failed: Invalid system package '{bad_package}'"
                )
                raise InvalidPackageError(bad_package, error_msg) from e

        # Re-raise if not a package error
        logger.error(f"Image build failed: {error_msg}")
        raise

    task = _create_container_task(
        language=solution.language,
        image=built_image,
        container_resources=container_resources,
    )

    # Associate the dynamically created task with the code_gen_environment
    task.parent_env = weakref.ref(code_gen_environment)
    task.parent_env_name = code_gen_environment.name

    return built_image, task


@flyte.trace
async def fix_failing_tests(
    model: str,
    test_code: str,
    diagnosis: ErrorDiagnosis,
    solution: CodeSolution,
    litellm_params: Optional[dict] = None,
) -> tuple[str, int, int]:
    """Fix only the failing tests while keeping passing tests unchanged.

    Args:
        test_code: Complete test file
        diagnosis: Structured diagnosis of test failures
        solution: The solution being tested

    Returns:
        Tuple of (fixed_test_code, input_tokens, output_tokens)
    """
    full_code = f"{solution.dependencies}\n\n{solution.code}"

    # Build diagnosis information from individual failures
    diagnosis_info = []
    for i, failure in enumerate(diagnosis.failures, 1):
        diagnosis_info.append(
            f"""
Test {i}: {failure.test_name}
- Expected: {failure.expected_behavior}
- Actual: {failure.actual_behavior}
- Root cause: {failure.root_cause}
- Suggested fix: {failure.suggested_fix}"""
        )

    diagnosis_section = "\n".join(diagnosis_info)

    fix_prompt = f"""Fix ONLY the failing tests in the test file below.

DIAGNOSIS OF TEST FAILURES:
{diagnosis_section}

SOLUTION CODE (for reference):
```{solution.language}
{full_code}
```

CURRENT TEST FILE:
```{solution.language}
{test_code}
```

YOUR TASK:
1. Review the diagnosis of why tests are failing
2. Fix ONLY the test functions mentioned in the diagnosis
3. Keep ALL other tests EXACTLY as they are (unchanged)
4. Return the COMPLETE test file with only the failing tests fixed

IMPORTANT:
- Do NOT modify passing tests
- Do NOT change imports or setup code unless necessary for the fix
- Do NOT add new tests
- Apply the suggested fixes from the diagnosis
- Return ONLY raw Python code (no code fences or explanations)"""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": fix_prompt}],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    params.update(litellm_params or {})
    params["response_format"] = _TestCodeResponse

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        test_dict = json.loads(content)
        fixed_test_code = test_dict["test_code"]
    else:
        fixed_test_code = content.test_code

    return fixed_test_code, input_tokens, output_tokens


async def _execute_tests(
    solution: CodeSolution, tests: str, eval_task: ContainerTask
) -> tuple[str, str, bool]:
    """Execute tests in container and return results.

    Returns:
        (test_output, test_exit_code, tests_passed)
    """
    language = solution.language.lower()
    file_ext = FILE_EXTENSIONS.get(language, ".txt")

    # Prepare code content
    full_code = f"{solution.dependencies}\n\n{solution.code}\n"

    # Write code and tests to temp files
    code_file = tempfile.NamedTemporaryFile(mode="w", suffix=file_ext, delete=False)
    code_file.write(full_code)
    code_file.close()

    # Write test code
    test_code_to_write = tests
    test_suffix = file_ext

    test_file = tempfile.NamedTemporaryFile(mode="w", suffix=test_suffix, delete=False)
    test_file.write(test_code_to_write)
    test_file.close()

    # Determine input names (must match ContainerTask inputs)
    code_input_name = "solution.py"
    test_input_name = "test_solution.py"

    # Execute tests with Files directly (use **kwargs for dynamic names)
    inputs = {
        code_input_name: await File.from_local(code_file.name),
        test_input_name: await File.from_local(test_file.name),
    }
    test_output, test_exit_code = await eval_task(**inputs)

    # Parse exit code
    tests_passed = test_exit_code.strip() == "0"

    # Cleanup temp files
    os.unlink(code_file.name)
    os.unlink(test_file.name)

    return test_output, test_exit_code, tests_passed


def extract_error_messages_from_pytest(output: str) -> dict[str, str]:
    """Extract the final error message for each failed test from pytest output.

    Args:
        output: Pytest output string

    Returns:
        Dict mapping test name to error message (e.g., "RecursionError: maximum recursion depth exceeded")
    """
    import re

    error_messages = {}
    current_test = None

    # Parse pytest output line by line
    lines = output.split("\n")

    for line in lines:
        # Match test failure header: _____ test_name _____
        test_header_match = re.match(r"^_{5,}\s+(.+?)\s+_{5,}$", line)
        if test_header_match:
            current_test = test_header_match.group(1).strip()
            # Remove parametrize suffix like [530.00-33-3]
            current_test = re.sub(r"\[.*?\]$", "", current_test)
            continue

        # Match error line: starts with "E   " followed by exception
        if current_test and line.startswith("E   "):
            error_line = line[4:].strip()  # Remove "E   " prefix
            # Only capture if it looks like an exception (contains "Error" or "Exception")
            if (
                "Error" in error_line
                or "Exception" in error_line
                or "Failed" in error_line
            ):
                # Extract just the exception type and message, not the full line
                error_messages[current_test] = error_line

    return error_messages


@flyte.trace
async def diagnose_error(
    model: str,
    solution: CodeSolution,
    output: str,
    prompt: str,
    plan: CodePlan,
    litellm_params: Optional[dict],
    test_code: Optional[str] = None,
    data_samples: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    schema: Optional[str] = None,
) -> tuple[ErrorDiagnosis, int, int]:
    """Performs structured analysis to determine if the error is due to:
    - Environment issues (missing packages, dependencies)
    - Code logic issues (bugs, incorrect algorithms)
    - Test code issues (wrong expected values, incorrect test logic)

    Returns:
        Tuple of (ErrorDiagnosis, input_tokens, output_tokens)
    """
    # Extract error messages from pytest output for accurate tracking
    error_messages = extract_error_messages_from_pytest(output)

    full_code = f"{solution.dependencies}\n\n{solution.code}"

    # Build error messages section for prompt
    error_messages_section = ""
    if error_messages:
        error_messages_section = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nEXTRACTED ERROR MESSAGES\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for test_name, error_msg in error_messages.items():
            error_messages_section += f"\n{test_name}: {error_msg}"

    diagnosis_prompt = f"""Diagnose CODE EXECUTION FAILURE. Tests may be wrong - check test bugs first.

TASK: {prompt}

PLAN: {plan.description} | {plan.approach}

SOLUTION CODE:
```{solution.language}
{full_code}
```"""

    if data_samples:
        diagnosis_prompt += f"""

DATA SAMPLES:
```
{data_samples}
```"""

    if constraints:
        diagnosis_prompt += "\n\nCONSTRAINTS:"
        for i, constraint in enumerate(constraints, 1):
            diagnosis_prompt += f"\n{i}. {constraint}"

    if schema:
        diagnosis_prompt += f"""

SCHEMA:
```
{schema}
```"""

    diagnosis_prompt += f"""

TEST CODE:
{test_code}

TEST OUTPUT:
{output}

{error_messages_section}
Diagnose EACH failing test individually:

1. failures - For EACH failing test:
   - test_name: The test function name
   - error_message: Copy from EXTRACTED ERROR MESSAGES above
   - expected_behavior, actual_behavior, root_cause
   - suggested_fix: "Replace `old code` with `new code`" (quote EXACT code, not line numbers)
   - error_type: Choose ONE:
     * "test_error" - Fix MODIFIES test code (assertions/expectations/test setup). Use ONLY if test contradicts data/schema/constraints or invents unverifiable values.
     * "logic" - Fix MODIFIES solution code (DEFAULT for most failures)
     * "environment" - Fix ADDS packages/dependencies

   CRITICAL: If ANY test has test_error, STOP and return ONLY test_error tests (broken tests invalidate logic diagnoses).

2. needs_system_packages, needs_language_packages, needs_additional_commands: Collect from environment errors

REQUIREMENTS:
- Quote exact code in suggested_fix
- /var/outputs is HARDCODED - never make it configurable"""

    # Build params with defaults
    params = {
        "model": model,
        "messages": [{"role": "user", "content": diagnosis_prompt}],
        "max_tokens": 2000,  # Increased for detailed test-by-test analysis
        "temperature": 0.3,
    }

    # Merge litellm_params (can override anything except response_format)
    params.update(litellm_params or {})

    # Always set response_format last
    params["response_format"] = ErrorDiagnosis

    response = await litellm.acompletion(**params)

    # Extract token usage
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        result_dict = json.loads(content)
        return ErrorDiagnosis(**result_dict), input_tokens, output_tokens
    return content, input_tokens, output_tokens


@flyte.trace
async def verify_test_fixes_applied(
    model: str,
    diagnosis: ErrorDiagnosis,
    old_test_code: str,
    new_test_code: str,
    litellm_params: Optional[dict] = None,
) -> tuple[FixVerification, int, int]:
    """Verify that the suggested test fixes from diagnosis are present in new test code.

    Returns:
        Tuple of (FixVerification, input_tokens, output_tokens)
    """
    # Build verification prompt
    fixes_to_check = []
    for i, failure in enumerate(diagnosis.failures, 1):
        fixes_to_check.append(
            f"""Fix {i} (for test: {failure.test_name}):
- Root cause: {failure.root_cause}
- Required fix: {failure.suggested_fix}"""
        )

    fixes_section = "\n\n".join(fixes_to_check)

    verify_prompt = f"""You must verify that ALL the required test fixes below are present in the new test code.

Required fixes:
{fixes_section}

Old test code:
```python
{old_test_code}
```

New test code:
```python
{new_test_code}
```

Check each fix carefully:
1. For each fix, determine if the required change is present in the new test code
2. Compare old vs new test code to see what changed
3. Verify that the failing tests mentioned were actually modified
4. List which fixes are applied and which are missing
5. Set all_fixes_applied to true ONLY if every single fix is present

Be strict - if even one fix is missing or partially applied, set all_fixes_applied to false."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": verify_prompt}],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    params.update(litellm_params or {})
    params["response_format"] = FixVerification

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        result_dict = json.loads(content)
        verification = FixVerification(**result_dict)
    else:
        verification = content

    return verification, input_tokens, output_tokens


@flyte.trace
async def verify_logic_fixes_applied(
    model: str,
    diagnosis: ErrorDiagnosis,
    new_solution: CodeSolution,
    litellm_params: Optional[dict] = None,
) -> tuple[FixVerification, int, int]:
    """Verify that the suggested fixes from diagnosis are present in new code.

    Only verifies "logic" failures - environment and test_error are handled differently.

    Returns:
        Tuple of (FixVerification, input_tokens, output_tokens)
    """
    # Only check logic failures (environment and test_error don't modify solution code)
    logic_failures = [f for f in diagnosis.failures if f.error_type == "logic"]

    if not logic_failures:
        # No logic fixes to verify
        return (
            FixVerification(
                all_fixes_applied=True,
                applied_fixes=[],
                missing_fixes=[],
                explanation="No logic fixes to verify (only environment/test errors)",
            ),
            0,
            0,
        )

    new_code = f"{new_solution.dependencies}\n\n{new_solution.code}"

    # Build verification prompt - only for logic failures
    fixes_to_check = []
    for i, failure in enumerate(logic_failures, 1):
        fixes_to_check.append(
            f"""Fix {i} (for test: {failure.test_name}):
- Root cause: {failure.root_cause}
- Required fix: {failure.suggested_fix}"""
        )

    fixes_section = "\n\n".join(fixes_to_check)

    verify_prompt = f"""You must verify that ALL the required fixes below are present in the new code.

Required fixes:
{fixes_section}

New code to verify:
```{new_solution.language}
{new_code}
```

Check each fix carefully:
1. For each fix, determine if the required change is present in the new code
2. If a fix says "Replace X with Y", verify that X is gone and Y is present
3. List which fixes are applied and which are missing
4. Set all_fixes_applied to true ONLY if every single fix is present

Be strict - if even one fix is missing or partially applied, set all_fixes_applied to false."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": verify_prompt}],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    params.update(litellm_params or {})
    params["response_format"] = FixVerification

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = _extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        result_dict = json.loads(content)
        verification = FixVerification(**result_dict)
    else:
        verification = content

    return verification, input_tokens, output_tokens


async def _diagnose_and_plan_environment_fix(
    model: str,
    solution: CodeSolution,
    code_output: str,
    prompt: str,
    plan: CodePlan,
    detected_packages: list[str],
    detected_system_packages: list[str],
    additional_commands: list[str],
    litellm_params: Optional[dict] = None,
    test_code: Optional[str] = None,
    data_samples: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    schema: Optional[str] = None,
) -> tuple[bool, list[str], list[str], list[str], int, int, ErrorDiagnosis]:
    """Diagnose error and plan environment fix (don't execute yet).

    Returns:
        (is_env_error, updated_detected_packages, updated_detected_system_packages, updated_additional_commands, input_tokens, output_tokens, diagnosis)
    """
    diagnosis, in_tok, out_tok = await diagnose_error(
        model,
        solution,
        code_output,
        prompt,
        plan,
        litellm_params,
        test_code,
        data_samples,
        constraints,
        schema,
    )

    # IMPORTANT: If ANY test errors exist, ONLY keep test_error failures
    # Discard logic/environment failures - they're unreliable when tests are broken
    test_error_failures = [
        f for f in diagnosis.failures if f.error_type == "test_error"
    ]

    if test_error_failures:
        logger.warning(
            f"Found {len(test_error_failures)} test_error failure(s). "
            f"Discarding {len(diagnosis.failures) - len(test_error_failures)} logic/environment failures "
            f"(unreliable when tests are broken)"
        )
        # Replace failures with only test_error failures
        diagnosis.failures = test_error_failures

    # Count error types across all failures (after filtering)
    error_type_counts = {"environment": 0, "test_error": 0, "logic": 0}
    for failure in diagnosis.failures:
        error_type_counts[failure.error_type] += 1

    logger.info(f"Number of failures: {len(diagnosis.failures)}")
    logger.info(
        f"Breakdown: {error_type_counts['environment']} environment, {error_type_counts['test_error']} test_error, {error_type_counts['logic']} logic"
    )

    for i, failure in enumerate(diagnosis.failures, 1):
        logger.info(
            f"Test {i} [{failure.error_type}] - {failure.test_name}",
            extra={"markup": False},
        )
        logger.info(f"Root cause: {failure.root_cause}", extra={"markup": False})
        logger.info(f"Fix: {failure.suggested_fix}", extra={"markup": False})

    # Determine actions based on all groups
    has_environment_errors = error_type_counts["environment"] > 0
    has_test_errors = error_type_counts["test_error"] > 0
    has_logic_errors = error_type_counts["logic"] > 0

    if has_test_errors:
        # Test code has bugs - MUST regenerate tests first
        logger.warning(
            f"{error_type_counts['test_error']} test(s) have test bugs. "
            f"Will regenerate tests first, then handle other errors."
        )
        return (
            "test_error",
            detected_packages,
            detected_system_packages,
            additional_commands,
            in_tok,
            out_tok,
            diagnosis,
        )

    # No test errors - we can fix environment + logic together in one go
    if has_environment_errors and has_logic_errors:
        logger.info(
            f"Will fix {error_type_counts['environment']} environment error(s) "
            f"and patch code for {error_type_counts['logic']} logic error(s) in one iteration"
        )
    elif has_environment_errors:
        logger.info(f"Will fix {error_type_counts['environment']} environment error(s)")
    elif has_logic_errors:
        logger.info(f"Will patch code for {error_type_counts['logic']} logic error(s)")

    # Return flag indicating we should handle both environment and logic
    # False means "not test_error", so we'll proceed to environment + code fix
    return (
        False,
        detected_packages,
        detected_system_packages,
        additional_commands,
        in_tok,
        out_tok,
        diagnosis,
    )


@syncify
async def code_gen_eval(
    name: str,
    prompt: str,
    model: str = "gpt-4.1",
    schema: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    data: Optional[dict[str, any]] = None,
    inputs: Optional[dict[str, type]] = None,
    outputs: Optional[dict[str, type]] = None,
    base_packages: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    max_retries: int = 10,
    max_sample_rows: int = 100,
    container_resources: Optional[flyte.Resources] = None,
    image_config: Optional[ImageConfig] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    litellm_params: Optional[dict] = None,
) -> CodeGenEvalResult:
    """Generate and evaluate code in an isolated container.

    Each call is independent with its own container, packages, and execution environment.

    **IMPORTANT**: Your TaskEnvironment must include `code_gen_environment` as a dependency:

    Example:
        ```python
        from flyteplugins.llm_codegen import code_gen_eval, code_gen_environment

        my_env = flyte.TaskEnvironment(
            name="my_env",
            image=flyte.Image.auto(),
            depends_on=[code_gen_environment],  # Add code_gen_environment
        )

        @my_env.task
        def my_task():
            result = code_gen_eval(prompt="...")
            return result
        ```

    Args:
        name: Unique name for this code generation task (used for flyte.group naming and tracking)
        prompt: The prompt to generate code from
        model: LiteLLM model identifier (default: gpt-4.1). Must support structured outputs (response_format).
        schema: Optional schema definition for EXTERNAL systems (e.g., target database schema, external API schema).
                NOT needed for input data schema - that's automatically extracted from the 'data' parameter.
                Only use this for schemas not present in your data (e.g., database you're writing to, external APIs).
        constraints: Optional list of constraints or requirements
        data: Optional dict of actual data to process (e.g., {"sales": pd.DataFrame(...), "config": File(...)}).
                Supported types: File, pd.DataFrame. DataFrames are auto-converted to Files.
                Schema, statistics, patterns, and samples are AUTOMATICALLY EXTRACTED from this data.
                Use result.run() to execute on this data or as_container_task() for reusable tasks.
        inputs: Optional dict defining CLI arguments (e.g., {"csv_data": File, "threshold": float}).
                Supported types: str, int, float, bool, File, Dir.
                If not provided and `data` is given, types are inferred automatically.
        outputs: Optional dict defining output types for ContainerTask (e.g., {"result": str, "data_file": File}).
                Supported types: str, int, float, bool, datetime, timedelta, File, Dir.
                Generated code will write each output to /var/outputs/{output_name}.
                NOTE: exit_code (int) is automatically added by as_container_task() - no need to include it.
                If None, no outputs are declared (use for side-effect only tasks).
        base_packages: Base packages to install in container
        system_prompt: Optional custom system prompt (structured output requirements appended automatically)
        max_retries: Maximum retries for the entire cycle (default: 20)
        max_sample_rows: Maximum rows to include in extracted data samples (default: 100)
        container_resources: Resources for the container (default: cpu=1, memory="1Gi")
        image_config: Image configuration (registry, registry_secret, python_version)
        api_key: Optional API key as env var name for LLM
        api_base: Optional custom API base URL
        litellm_params: Optional dict of additional LiteLLM parameters (e.g., max_tokens, temperature, top_p, frequency_penalty, timeout).
            Can override defaults for all LLM calls throughout the code generation process.

    Returns:
        CodeGenEvalResult with solution and execution details
    """
    # Language is always Python for now
    language = "python"

    # Validate input types
    if inputs:
        supported_input_types = (str, int, float, bool, File, Dir)
        for input_key, input_type in inputs.items():
            if input_type not in supported_input_types:
                supported_names = [t.__name__ for t in supported_input_types]
                raise ValueError(
                    f"Unsupported input type for '{input_key}': {input_type}. "
                    f"ContainerTask only supports: {', '.join(supported_names)}"
                )

    # Process data parameter: infer types, convert DataFrames, extract context
    original_data_files = None
    extracted_data_context = None

    total_input_tokens = 0
    total_output_tokens = 0

    if data:
        logger.info(f"Processing {len(data)} data inputs...")
        inferred_types = {}
        original_data_files = {}

        for data_key, value in data.items():
            if isinstance(value, File):
                # File: use as-is
                inferred_types[data_key] = File
                original_data_files[data_key] = value
            elif _is_dataframe(value):
                # DataFrame: convert to temp CSV File
                temp_file = Path(tempfile.gettempdir()) / f"{data_key}.csv"
                value.to_csv(temp_file, index=False)
                file_obj = await File.from_local(str(temp_file))

                inferred_types[data_key] = File
                original_data_files[data_key] = file_obj
            else:
                raise ValueError(
                    f"Unsupported data type for '{data_key}': {type(value)}. "
                    f"Only File and pd.DataFrame are supported."
                )

        # Extract comprehensive context from data
        logger.info("Extracting data context (schema, stats, patterns)...")
        extracted_data_context = await extract_data_context(data, max_sample_rows)
        logger.info(f"Extracted context: {len(extracted_data_context)} characters")

        # Use inferred types if inputs not explicitly provided
        if not inputs:
            inputs = inferred_types
            logger.info(f"Inferred input types: {inputs}")

    # Validate output types
    if outputs:
        supported_types = (
            str,
            int,
            float,
            bool,
            datetime.datetime,
            datetime.timedelta,
            File,
            Dir,
        )
        for output_key, output_type in outputs.items():
            if output_type not in supported_types:
                supported_names = [t.__name__ for t in supported_types]
                raise ValueError(
                    f"Unsupported output type for '{output_key}': {output_type}. "
                    f"ContainerTask only supports: {', '.join(supported_names)}"
                )

    logger.info(
        f"Starting code generation: language={language}, model={model}, max_retries={max_retries}"
    )

    # Configure LiteLLM
    if api_key:
        litellm.api_key = os.getenv(api_key)
    if api_base:
        litellm.api_base = api_base

    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0

    # Build prompts and initialize state
    base_prompt_text = system_prompt or DEFAULT_SYSTEM_PROMPT
    final_system_prompt = f"{base_prompt_text}\n{STRUCTURED_OUTPUT_REQUIREMENTS}"

    enhanced_prompt = _build_enhanced_prompt(
        prompt, language, schema, constraints, extracted_data_context, inputs, outputs
    )

    # Base messages that stay constant throughout retries
    base_messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": enhanced_prompt},
    ]

    # Step 0: Generate plan
    logger.info("Generating plan...")
    plan, in_tok, out_tok = await generate_plan(
        model,
        prompt,
        language,
        schema,
        constraints,
        extracted_data_context,
        inputs,
        outputs,
        litellm_params,
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    logger.info(f"Plan created: {plan.description}")
    logger.info(f"Approach: {plan.approach}")

    base_pkgs = base_packages or []

    # Add test framework packages to base packages
    test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])
    for pkg in test_framework_info["packages"]:
        if pkg not in base_pkgs:
            base_pkgs.append(pkg)

    detected_packages: list[str] = []
    detected_system_packages: list[str] = []
    additional_commands: list[str] = []

    # Add test framework system packages
    for pkg in test_framework_info.get("system_packages", []):
        if pkg not in detected_system_packages:
            detected_system_packages.append(pkg)

    current_eval_task = None
    current_image = None
    previously_installed_packages: list[str] = []  # Track what's already in the image
    previously_installed_system_packages: list[str] = []
    last_error = None
    last_error_message = None  # Latest error diagnosis to send to LLM
    last_diagnosis = None  # Keep last diagnosis for verification
    last_result = None
    solution = None
    tests = None
    needs_new_code = True
    needs_new_tests = True  # Only regenerate tests when necessary
    needs_rebuild = True  # Always build on first iteration
    last_packages_snapshot = (
        set(),
        set(),
    )  # (detected_packages, detected_system_packages)

    # Track logic fix attempts for each error signature (for reclassification)
    # Key: (test_name, error_signature) to detect if SAME error repeats
    logic_fix_attempts = {}  # {(test_name, error_sig): attempt_count}
    max_logic_attempts = (
        2  # After this many attempts with SAME error, reclassify as test_error
    )

    # Track test fix attempts for each error signature (for reverse reclassification)
    # Key: (test_name, error_signature) to detect if SAME error repeats after test fixes
    test_fix_attempts = {}  # {(test_name, error_sig): attempt_count}
    max_test_attempts = (
        2  # After this many attempts with SAME error, reclassify as logic
    )

    # Create initial image name from base packages
    all_packages = base_pkgs
    image_spec_for_hash = {
        "language": language,
        "packages": sorted(all_packages),
        "system_packages": sorted([]),
    }
    config_str = json.dumps(image_spec_for_hash, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    image_name = f"code-gen-{language}-{config_hash}"

    for attempt in range(1, max_retries + 1):
        with flyte.group(f"{name}-attempt-{attempt}"):
            logger.info(
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"[ITERATION] Starting attempt {attempt}/{max_retries}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            try:
                # 1. Generate code (only when needed)
                if needs_new_code:
                    # Try up to 3 times to generate code with fixes properly applied
                    max_code_gen_attempts = 3
                    code_gen_success = False

                    for code_attempt in range(1, max_code_gen_attempts + 1):
                        logger.info(
                            f"Generating code (attempt {attempt}, code gen attempt {code_attempt}/{max_code_gen_attempts})..."
                        )
                        # Build messages: base + latest error (if any)
                        messages = base_messages.copy()
                        if last_error_message:
                            # Make prompt progressively more forceful
                            if code_attempt == 1:
                                messages.append(
                                    {"role": "user", "content": last_error_message}
                                )
                            elif code_attempt == 2:
                                forceful_msg = f"""{last_error_message}

CRITICAL: The previous code generation attempt did NOT apply all the required fixes.
You MUST apply EVERY SINGLE fix listed above. Do not skip any fix.
Apply each fix EXACTLY as specified - find the old code and replace it with the new code."""
                                messages.append(
                                    {"role": "user", "content": forceful_msg}
                                )
                            else:
                                very_forceful_msg = f"""{last_error_message}

FINAL ATTEMPT: You have failed to apply the required fixes twice.
This is your last chance. Apply EVERY fix listed above WITHOUT EXCEPTION.
For each fix, you MUST:
1. Find the EXACT old code mentioned
2. Replace it with the EXACT new code mentioned
3. Do NOT change anything else

If you fail to apply all fixes this time, the entire task will fail."""
                                messages.append(
                                    {"role": "user", "content": very_forceful_msg}
                                )

                        solution, in_tok, out_tok = await generate_code(
                            model,
                            messages,
                            plan,
                            litellm_params,
                            is_retry=(attempt > 1),
                        )
                        total_input_tokens += in_tok
                        total_output_tokens += out_tok

                        if solution.language.lower() != language.lower():
                            logger.warning(
                                f"Requested {language} but LLM generated {solution.language}"
                            )

                        # Verify fixes are applied
                        has_logic_failures = last_diagnosis and any(
                            f.error_type == "logic" for f in last_diagnosis.failures
                        )
                        if has_logic_failures:
                            logger.info("Verifying that logic fixes were applied...")
                            verification, in_tok, out_tok = (
                                await verify_logic_fixes_applied(
                                    model, last_diagnosis, solution, litellm_params
                                )
                            )
                            total_input_tokens += in_tok
                            total_output_tokens += out_tok

                            if verification.all_fixes_applied:
                                logger.info(
                                    f"Verification passed: All fixes applied. {verification.explanation}"
                                )
                                code_gen_success = True
                                break
                            else:
                                logger.warning(
                                    f"Verification failed: {verification.explanation}"
                                )
                                logger.warning(f"Applied: {verification.applied_fixes}")
                                logger.warning(f"Missing: {verification.missing_fixes}")

                                if code_attempt < max_code_gen_attempts:
                                    # Update error message with specific missing fixes for next attempt
                                    missing_fixes_msg = "\n\nVERIFICATION FAILED - The following fixes are STILL MISSING:\n"
                                    for i, missing_fix in enumerate(
                                        verification.missing_fixes, 1
                                    ):
                                        missing_fixes_msg += f"\n{i}. {missing_fix}"

                                    missing_fixes_msg += (
                                        f"\n\nYou successfully applied these fixes:\n"
                                    )
                                    for applied_fix in verification.applied_fixes:
                                        missing_fixes_msg += f"- {applied_fix}\n"

                                    missing_fixes_msg += "\nYou MUST now apply the MISSING fixes listed above. Do NOT regenerate the entire solution - just apply the missing fixes to your previous code."

                                    # Append verification feedback to error message
                                    last_error_message = (
                                        last_error_message or ""
                                    ) + missing_fixes_msg

                                    logger.info(
                                        f"Regenerating code with verification feedback (attempt {code_attempt + 1}/{max_code_gen_attempts})..."
                                    )
                                else:
                                    logger.error(
                                        f"Failed to apply all fixes after {max_code_gen_attempts} attempts. Proceeding anyway..."
                                    )
                                    code_gen_success = (
                                        True  # Proceed anyway after max attempts
                                    )
                        else:
                            # No diagnosis to verify against (first attempt or environment error)
                            code_gen_success = True
                            break

                    if not code_gen_success:
                        logger.error(
                            "Failed to generate code with all fixes applied. Skipping this iteration."
                        )
                        continue

                    # 2. Detect and track packages
                    (
                        needs_rebuild,
                        detected_packages,
                        detected_system_packages,
                        in_tok,
                        out_tok,
                    ) = await _detect_and_track_packages(
                        model,
                        solution,
                        base_pkgs,
                        detected_packages,
                        detected_system_packages,
                        litellm_params,
                    )
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok

                    needs_new_code = False
                    # Only generate tests if we don't have them yet
                    if tests is None:
                        needs_new_tests = True

                # 3. Generate tests (only when needed) - single pass with filtering
                if needs_new_tests:
                    logger.info("Generating tests...")

                    tests, in_tok, out_tok = await generate_tests(
                        model,
                        prompt,
                        plan,
                        solution,
                        constraints,
                        schema,
                        extracted_data_context,
                        inputs,
                        outputs,
                        litellm_params,
                    )
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    logger.debug(f"Generated tests: {tests}")

                    needs_new_tests = False

                # Check if packages changed (even without new code)
                current_packages_snapshot = (
                    set(detected_packages),
                    set(detected_system_packages),
                )
                if current_packages_snapshot != last_packages_snapshot:
                    needs_rebuild = True
                    last_packages_snapshot = current_packages_snapshot

                    # Always update image name to match current package set
                    # This ensures image name always reflects exact contents
                    all_packages = base_pkgs + detected_packages
                    image_spec_for_hash = {
                        "language": language,
                        "packages": sorted(all_packages),
                        "system_packages": sorted(detected_system_packages),
                    }
                    config_str = json.dumps(image_spec_for_hash, sort_keys=True)
                    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

                    new_image_name = f"code-gen-{language}-{config_hash}"

                    if new_image_name != image_name:
                        logger.info(
                            f"Image name updated: {image_name} → {new_image_name}"
                        )
                        image_name = new_image_name
                        # Reset current_image so we build from scratch with new name
                        current_image = None

                # 3. Build/rebuild image if needed
                if needs_rebuild or current_eval_task is None:
                    # Retry loop for handling invalid packages
                    max_package_retries = 3
                    package_retry_count = 0

                    while package_retry_count < max_package_retries:
                        try:
                            current_image, current_eval_task = (
                                await _build_image_and_task(
                                    solution,
                                    base_pkgs,
                                    detected_packages,
                                    detected_system_packages,
                                    previously_installed_packages,
                                    previously_installed_system_packages,
                                    additional_commands,
                                    container_resources,
                                    image_name,
                                    current_image,  # Use previous image as base
                                    image_config,
                                )
                            )

                            # Update what's now installed
                            previously_installed_packages = detected_packages.copy()
                            previously_installed_system_packages = (
                                detected_system_packages.copy()
                            )
                            needs_rebuild = False
                            break  # Success, exit retry loop

                        except InvalidPackageError as e:
                            # Invalid package detected - remove it and retry
                            bad_package = e.package_name
                            logger.warning(
                                f"Removing invalid system package '{bad_package}' and retrying build..."
                            )

                            # Remove the bad package from detected_system_packages
                            if bad_package in detected_system_packages:
                                detected_system_packages.remove(bad_package)

                            # Also remove from previously_installed to avoid tracking it
                            if bad_package in previously_installed_system_packages:
                                previously_installed_system_packages.remove(bad_package)

                            logger.info(
                                f"Retrying with system packages: {detected_system_packages}"
                            )

                            # Increment retry count and try again
                            package_retry_count += 1
                            continue  # Retry the build in same iteration

                # 4. Execute tests
                logger.info("Running tests...")
                test_output, test_exit_code, tests_passed = await _execute_tests(
                    solution, tests, current_eval_task
                )
                execution_success = tests_passed

                # 5. Handle success
                if execution_success:
                    logger.info("Tests passed! Solution successful.")
                    logger.info(
                        f"Total tokens: input={total_input_tokens}, output={total_output_tokens}"
                    )
                    # Clear diagnosis on success
                    last_diagnosis = None
                    return CodeGenEvalResult(
                        plan=plan,
                        solution=solution,
                        tests=tests,
                        success=True,
                        output=test_output,
                        exit_code=int(test_exit_code.strip()),
                        error=None,
                        attempts=attempt,
                        conversation_history=base_messages,
                        detected_packages=detected_packages,
                        detected_system_packages=detected_system_packages,
                        image=current_image,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        declared_inputs=inputs,
                        declared_outputs=outputs,
                        data_context=extracted_data_context,
                        original_data=original_data_files,
                    )

                # 6. Handle failure
                if not execution_success:
                    # Check if tests actually executed (vs test file having errors)
                    # If no tests ran, it's likely a test file error (syntax, import, etc.)
                    # Pytest shows "X passed" or "X failed" if tests ran
                    tests_executed = (
                        " passed" in test_output
                        or " failed" in test_output
                        or "collected 0 items" not in test_output
                    ) and "ERROR collecting" not in test_output

                    if not tests_executed:
                        logger.warning(
                            "No tests executed - test file likely has errors. Regenerating tests..."
                        )
                        needs_new_tests = True
                        needs_new_code = False
                        continue

                    # Tests failed - diagnose
                    (
                        is_env_error,
                        detected_packages,
                        detected_system_packages,
                        additional_commands,
                        in_tok,
                        out_tok,
                        diagnosis,
                    ) = await _diagnose_and_plan_environment_fix(
                        model,
                        solution,
                        test_output,
                        prompt,
                        plan,
                        detected_packages,
                        detected_system_packages,
                        additional_commands,
                        litellm_params,
                        tests,
                        extracted_data_context,
                        constraints,
                        schema,
                    )
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok

                    # Store diagnosis for verification in next iteration
                    last_diagnosis = diagnosis

                    # Extract and apply environment fixes from diagnosis
                    if diagnosis.needs_language_packages:
                        added_lang_packages = []
                        for pkg in diagnosis.needs_language_packages:
                            if pkg not in detected_packages and pkg not in base_pkgs:
                                detected_packages.append(pkg)
                                added_lang_packages.append(pkg)
                        if added_lang_packages:
                            logger.info(
                                f"Adding language packages from diagnosis: {added_lang_packages}"
                            )
                            needs_rebuild = True

                    if diagnosis.needs_system_packages:
                        added_sys_packages = []
                        for pkg in diagnosis.needs_system_packages:
                            if pkg not in detected_system_packages:
                                detected_system_packages.append(pkg)
                                added_sys_packages.append(pkg)
                        if added_sys_packages:
                            logger.info(
                                f"Adding system packages from diagnosis: {added_sys_packages}"
                            )
                            needs_rebuild = True

                    if diagnosis.needs_additional_commands:
                        logger.info(
                            f"Adding additional commands from diagnosis: {diagnosis.needs_additional_commands}"
                        )
                        additional_commands.extend(diagnosis.needs_additional_commands)
                        needs_rebuild = True

                    # Check for repeated test_error failures and reclassify as logic
                    # (Test might be correct, code is actually wrong)
                    reclassified_count = 0
                    for failure in diagnosis.failures:
                        if failure.error_type == "test_error":
                            # Create error signature from final error message
                            error_signature = (
                                failure.error_message or failure.actual_behavior
                            )
                            error_key = (failure.test_name, error_signature)

                            # Track attempts for this test with this specific error
                            test_fix_attempts[error_key] = (
                                test_fix_attempts.get(error_key, 0) + 1
                            )

                            logger.info(
                                f"[RECLASSIFY] test_error test='{failure.test_name}', "
                                f"error='{error_signature[:80]}', "
                                f"attempts={test_fix_attempts[error_key]}, "
                                f"max={max_test_attempts}, "
                                f"will_reclassify={test_fix_attempts[error_key] > max_test_attempts}"
                            )

                            if test_fix_attempts[error_key] > max_test_attempts:
                                # Reclassify as logic error
                                original_root_cause = failure.root_cause

                                failure.error_type = "logic"
                                failure.root_cause = (
                                    f"Test failed {max_test_attempts + 1} times with same error after test fixes. "
                                    f"The test expectations are likely correct. The code logic could be wrong. "
                                    f"Original test diagnosis was: {original_root_cause}"
                                )
                                failure.suggested_fix = (
                                    f"Fix the code logic to match test expectations. "
                                    f"Test expects: {failure.expected_behavior}. "
                                    f"Code produces: {failure.actual_behavior}. "
                                    f"Update the code to produce the expected behavior."
                                )

                                # Reset attempts
                                test_fix_attempts.pop(error_key, None)
                                reclassified_count += 1

                                logger.warning(
                                    f"Test '{failure.test_name}' still failing after test fixes. "
                                    f"Reclassified test_error → logic (test is probably correct, code could be wrong)."
                                )

                    if reclassified_count > 0:
                        logger.info(
                            f"Reclassified {reclassified_count} test_error(s) to logic. "
                            f"Will fix code instead of tests."
                        )

                    # Check for repeated logic failures and reclassify as test_error
                    # (LLM might misdiagnose test bugs as logic bugs)
                    reclassified_count = 0
                    for failure in diagnosis.failures:
                        if failure.error_type == "logic":
                            # Create error signature from final error message
                            error_signature = (
                                failure.error_message or failure.actual_behavior
                            )
                            error_key = (failure.test_name, error_signature)

                            # Track attempts for this test with this specific error
                            logic_fix_attempts[error_key] = (
                                logic_fix_attempts.get(error_key, 0) + 1
                            )

                            logger.info(
                                f"[RECLASSIFY] logic test='{failure.test_name}', "
                                f"error='{error_signature[:80]}', "
                                f"attempts={logic_fix_attempts[error_key]}, "
                                f"max={max_logic_attempts}, "
                                f"will_reclassify={logic_fix_attempts[error_key] > max_logic_attempts}"
                            )

                            if logic_fix_attempts[error_key] > max_logic_attempts:
                                # Reclassify as test error
                                original_root_cause = failure.root_cause

                                failure.error_type = "test_error"
                                failure.root_cause = (
                                    f"Test failed {max_logic_attempts + 1} times with same error after logic fixes. "
                                    f"Likely the test itself has wrong expected values, not the code. "
                                    f"Original diagnosis was: {original_root_cause}"
                                )
                                failure.suggested_fix = (
                                    f"Fix the test expectations to match actual correct behavior. "
                                    f"Code produces: {failure.actual_behavior}. "
                                    f"If this is correct, update the test to expect this value instead."
                                )

                                # Reset attempts
                                logic_fix_attempts.pop(error_key, None)
                                reclassified_count += 1

                                logger.warning(
                                    f"Test '{failure.test_name}' still failing after logic fixes. "
                                    f"Reclassified logic → test_error (test has wrong expectations)."
                                )

                    if reclassified_count > 0:
                        logger.info(
                            f"Reclassified {reclassified_count} logic error(s) to test_error. "
                            f"Will fix tests instead of code."
                        )

                        # Recalculate is_env_error after reclassification
                        has_test_errors = any(
                            f.error_type == "test_error" for f in diagnosis.failures
                        )
                        if has_test_errors:
                            # Filter to only test_error failures (same as earlier logic)
                            test_error_failures = [
                                f
                                for f in diagnosis.failures
                                if f.error_type == "test_error"
                            ]
                            diagnosis.failures = test_error_failures
                            is_env_error = "test_error"
                            logger.info(
                                f"After reclassification: {len(test_error_failures)} test_error failure(s). "
                                f"Will fix tests instead of code."
                            )

                    # Check if diagnosis identified bug in test code itself
                    if is_env_error == "test_error":
                        logger.info(
                            "Diagnosis identified bug in test code. Fixing only failed tests..."
                        )

                        # Get all failed test names from diagnosis
                        all_failed_tests = [f.test_name for f in diagnosis.failures]

                        logger.info(f"Failed tests to fix: {all_failed_tests}")

                        # Try up to 3 times to fix tests with fixes properly applied
                        max_test_fix_attempts = 3
                        test_fix_success = False
                        old_tests = tests  # Save current tests for verification

                        for test_fix_attempt in range(1, max_test_fix_attempts + 1):
                            logger.info(
                                f"Fixing failing tests (attempt {test_fix_attempt}/{max_test_fix_attempts})..."
                            )

                            # Ask LLM to fix only the failing tests using the diagnosis
                            tests, in_tok, out_tok = await fix_failing_tests(
                                model,
                                tests,
                                diagnosis,
                                solution,
                                litellm_params,
                            )
                            total_input_tokens += in_tok
                            total_output_tokens += out_tok

                            # Verify test fixes are applied
                            logger.info("Verifying that test fixes were applied...")
                            verification, in_tok, out_tok = (
                                await verify_test_fixes_applied(
                                    model, diagnosis, old_tests, tests, litellm_params
                                )
                            )
                            total_input_tokens += in_tok
                            total_output_tokens += out_tok

                            if verification.all_fixes_applied:
                                logger.info(
                                    f"Verification passed: All test fixes applied. {verification.explanation}"
                                )
                                test_fix_success = True
                                break
                            else:
                                logger.warning(
                                    f"Verification failed: {verification.explanation}"
                                )
                                logger.warning(f"Applied: {verification.applied_fixes}")
                                logger.warning(f"Missing: {verification.missing_fixes}")

                                if test_fix_attempt < max_test_fix_attempts:
                                    # Update diagnosis with specific missing fixes for next attempt
                                    missing_fixes_msg = "\n\nVERIFICATION FAILED - The following test fixes are STILL MISSING:\n"
                                    for i, missing_fix in enumerate(
                                        verification.missing_fixes, 1
                                    ):
                                        missing_fixes_msg += f"\n{i}. {missing_fix}"

                                    missing_fixes_msg += (
                                        f"\n\nYou successfully applied these fixes:\n"
                                    )
                                    for applied_fix in verification.applied_fixes:
                                        missing_fixes_msg += f"- {applied_fix}\n"

                                    missing_fixes_msg += "\nYou MUST now apply the MISSING test fixes listed above."

                                    # Update diagnosis failures with progressively forceful messages
                                    for failure in diagnosis.failures:
                                        if test_fix_attempt == 1:
                                            failure.suggested_fix = f"{failure.suggested_fix}\n\n{missing_fixes_msg}"
                                        elif test_fix_attempt == 2:
                                            failure.suggested_fix = f"{failure.suggested_fix}\n\n{missing_fixes_msg}\n\nCRITICAL: The previous test fix attempt did NOT apply all the required fixes. You MUST apply EVERY SINGLE fix listed above. Do not skip any fix."
                                        else:
                                            failure.suggested_fix = f"{failure.suggested_fix}\n\n{missing_fixes_msg}\n\nFINAL ATTEMPT: You have failed to apply the required test fixes twice. This is your last chance. Apply EVERY fix listed above WITHOUT EXCEPTION."

                                    logger.info(
                                        f"Regenerating test fixes with verification feedback (attempt {test_fix_attempt + 1}/{max_test_fix_attempts})..."
                                    )
                                else:
                                    logger.error(
                                        f"Failed to apply all test fixes after {max_test_fix_attempts} attempts. Proceeding anyway..."
                                    )
                                    test_fix_success = (
                                        True  # Proceed anyway after max attempts
                                    )

                        if not test_fix_success:
                            logger.error(
                                "Failed to fix tests with all fixes applied. Skipping this iteration."
                            )
                            continue

                        needs_new_code = False
                        # Clear diagnosis since we're fixing tests, not code
                        last_diagnosis = None
                        continue

                    # is_env_error is False here, meaning we should handle env and logic

                # Handle logic and/or environment errors
                # Environment errors: packages extracted above, image will rebuild
                # Logic errors: will regenerate code if present
                # The diagnosis is already stored in last_diagnosis and will be used
                # for verification in next iteration to ensure fixes are applied

                # Format failure diagnosis - include all failures (logic + environment)
                # Test errors are already handled above, so we won't reach here if there are test errors
                # Note: Environment failures are included in the message for completeness,
                # but if only env errors exist, we won't regenerate code (packages already added)
                failures_info = []
                logic_count = 0
                env_count = 0

                for i, failure in enumerate(diagnosis.failures, 1):
                    failures_info.append(
                        f"""
Test {i} [{failure.error_type}] - {failure.test_name}
- Expected: {failure.expected_behavior}
- Actual: {failure.actual_behavior}
- Root cause: {failure.root_cause}
- FIX: {failure.suggested_fix}"""
                    )
                    if failure.error_type == "logic":
                        logic_count += 1
                    elif failure.error_type == "environment":
                        env_count += 1

                failures_section = "\n".join(failures_info)

                # Log what we're handling
                if logic_count > 0 and env_count > 0:
                    logger.info(
                        f"Will fix {env_count} environment error(s) and patch code for {logic_count} logic error(s)"
                    )
                elif logic_count > 0:
                    logger.info(f"Will patch code for {logic_count} logic error(s)")
                elif env_count > 0:
                    logger.info(f"Will fix {env_count} environment error(s)")

                # Build error message for code patching
                # (Only used if logic_count > 0, otherwise skipped)
                full_code = f"{solution.dependencies}\n\n{solution.code}"

                error_msg = f"""Tests failed. Apply only the specific fixes below to your code.

Do not regenerate from scratch. PATCH the code by applying ONLY the fixes below.
Do NOT make any other changes - keep everything else exactly as is.

CRITICAL CONSTRAINTS:
1. /var/outputs is a REQUIRED hardcoded path. Never make it configurable or use a variable. Always use the literal string '/var/outputs' in your code.
2. If a part of the code is working correctly, DO NOT change it. Only fix what's broken.
3. Apply each fix by finding the exact code quoted and replacing it - nothing more.
4. Do NOT regenerate the entire code. Just apply the specific patches mentioned below.

Your previous code:
```{solution.language}
{full_code}
```

{failures_section}"""

                # Store error message for next code generation (only if we have logic errors)
                # Diagnosis is stored for verification in last_diagnosis
                if logic_count > 0:
                    logger.info(
                        "Tests failed. Will patch code with fixes (keeping same tests)..."
                    )
                    last_error_message = error_msg
                else:
                    # Only environment errors - no need to store error message for code generation
                    last_error_message = None

                last_result = CodeGenEvalResult(
                    plan=plan,
                    solution=solution,
                    tests=tests,
                    success=False,
                    output=test_output,
                    exit_code=int(test_exit_code.strip()),
                    error=error_msg,
                    attempts=attempt,
                    conversation_history=base_messages,
                    detected_packages=detected_packages,
                    detected_system_packages=detected_system_packages,
                    image=current_image,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    declared_inputs=inputs,
                    declared_outputs=outputs,
                    data_context=extracted_data_context,
                    original_data=original_data_files,
                )

                if attempt == max_retries:
                    return last_result

                # Set flags for next iteration based on error types
                # Only regenerate code if there are logic errors
                # Environment errors only need image rebuild (already set needs_rebuild above)
                if logic_count > 0:
                    # Has logic errors - need to regenerate code
                    needs_new_code = True
                    needs_new_tests = False
                elif env_count > 0:
                    # Only environment errors - just rebuild image
                    logger.info(
                        "Only environment errors - skipping code regeneration, will rebuild image with new packages"
                    )
                    needs_new_code = False
                    needs_new_tests = False

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error during attempt {attempt}: {last_error}")

                error_msg = f"An error occurred: {last_error}"
                last_error_message = error_msg

                last_result = CodeGenEvalResult(
                    plan=plan,
                    solution=solution or CodeSolution(),
                    tests=tests,
                    success=False,
                    output="",
                    exit_code=-1,
                    error=f"Attempt {attempt} failed: {last_error}",
                    attempts=attempt,
                    conversation_history=base_messages,
                    detected_packages=detected_packages,
                    detected_system_packages=detected_system_packages,
                    image=current_image,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    declared_inputs=inputs,
                    declared_outputs=outputs,
                    data_context=extracted_data_context,
                    original_data=original_data_files,
                )

                if attempt == max_retries:
                    return last_result

                # Generate new code after exception (keep tests if we have them)
                needs_new_code = True
                if tests is None:
                    needs_new_tests = True

    return last_result or CodeGenEvalResult(
        plan=plan,
        solution=CodeSolution(),
        tests=tests,
        success=False,
        output="",
        exit_code=-1,
        error=f"All {max_retries} attempts failed: {last_error}",
        attempts=max_retries,
        conversation_history=base_messages,
        detected_packages=detected_packages,
        detected_system_packages=detected_system_packages,
        image=current_image,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        declared_inputs=inputs,
        declared_outputs=outputs,
        data_context=extracted_data_context,
        original_data=original_data_files,
    )
