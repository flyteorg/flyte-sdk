from typing import Any, Literal, Optional

import flyte
from flyte.io import File
from flyte.syncify import syncify
from pydantic import BaseModel, Field, field_validator


class CodePlan(BaseModel):
    """Structured plan for the code solution."""

    description: str = Field(description="Overall description of the solution")
    approach: str = Field(description="High-level approach and algorithm to solve the problem")


class CodeSolution(BaseModel):
    """Structured code solution."""

    language: str = Field(
        default="python",
        description="Programming language",
    )
    code: str = Field(
        default="",
        description="Complete executable code including imports and dependencies",
    )
    system_packages: list[str] = Field(
        default_factory=list,
        description="System packages needed (e.g., gcc, build-essential, curl)",
    )

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, v: str) -> str:
        return v.strip().lower()


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
    detected_system_packages: list[str] = Field(default_factory=list, description="System packages detected by LLM")
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
        description="Input types (user-provided or inferred from samples)",
    )
    declared_outputs: Optional[dict[str, type]] = Field(
        default=None,
        description="Output types declared by user",
    )
    data_context: Optional[str] = Field(
        default=None,
        description="Extracted data context (schema, stats, patterns, samples) used for code generation",
    )
    original_samples: Optional[dict[str, File]] = Field(
        default=None,
        description="Sample data converted to Files (defaults for run()/as_task())",
    )
    generated_schemas: Optional[dict[str, str]] = Field(
        default=None,
        description="Auto-generated Pandera schemas (as Python code strings) for validating data inputs",
    )

    def as_task(
        self,
        name: str = "run_code_on_real_data",
        resources: Optional[flyte.Resources] = None,
        retries: int = 0,
        timeout: Optional[int] = None,
        env_vars: Optional[dict[str, str]] = None,
        secrets: Optional[list] = None,
        cache: str = "auto",
    ):
        """Create a sandbox that runs the generated code in an isolated sandbox.

        The generated code will write outputs to /var/outputs/{output_name} files.
        Returns a callable wrapper that automatically provides the script file.

        Args:
            name: Name for the sandbox
            resources: Optional resources for the task
            retries: Number of retries for the task. Defaults to 0.
            timeout: Timeout in seconds. Defaults to None.
            env_vars: Environment variables to pass to the sandbox.
            secrets: flyte.Secret objects to make available.
            cache: CacheRequest: "auto", "override", or "disable". Defaults to "auto".

        Returns:
            Callable task wrapper with the default inputs baked in. Call with your other declared inputs.
        """
        if not self.success:
            raise ValueError("Cannot create task from failed code generation")

        if not self.image:
            raise ValueError("No image available - code generation did not build an image")

        sandbox = flyte.sandbox.create(
            name=name,
            code=self.solution.code,
            inputs=self.declared_inputs or {},
            outputs=self.declared_outputs or {},
            auto_io=False,
            resources=resources or flyte.Resources(cpu=1, memory="1Gi"),
            retries=retries,
            timeout=timeout,
            env_vars=env_vars,
            secrets=secrets,
            cache=cache,
        )

        image = self.image

        # If we have samples, wrap to inject sample values as defaults
        if self.original_samples:
            sample_defaults = dict(self.original_samples)

            @syncify
            async def task_with_defaults(**kwargs):
                merged = {**sample_defaults, **kwargs}
                return await sandbox.run.aio(image=image, **merged)

            return task_with_defaults

        @syncify
        async def task(**kwargs):
            return await sandbox.run.aio(image=image, **kwargs)

        return task

    async def run(
        self,
        *,
        name: str = "run_code_on_real_data",
        resources: Optional[flyte.Resources] = None,
        retries: int = 0,
        timeout: Optional[int] = None,
        env_vars: Optional[dict[str, str]] = None,
        secrets: Optional[list] = None,
        cache: str = "auto",
        **overrides,
    ) -> Any:
        """Run generated code in an isolated sandbox (one-off execution).

        If samples were provided during generate(), they are used as defaults.
        Override any input by passing it as a keyword argument. If no samples
        exist, all declared inputs must be provided via ``**overrides``.

        Args:
            name: Name for the sandbox
            resources: Optional resources for the task
            retries: Number of retries for the task. Defaults to 0.
            timeout: Timeout in seconds. Defaults to None.
            env_vars: Environment variables to pass to the sandbox.
            secrets: flyte.Secret objects to make available.
            cache: CacheRequest: "auto", "override", or "disable". Defaults to "auto".
            **overrides: Input values. Merged on top of sample defaults (if any).

        Returns:
            Tuple of typed outputs.
        """
        if not self.success:
            raise ValueError("Cannot run failed code generation")

        if not self.image:
            raise ValueError("No image available - code generation did not build an image")

        sandbox = flyte.sandbox.create(
            name=name,
            code=self.solution.code,
            inputs=self.declared_inputs or {},
            outputs=self.declared_outputs or {},
            auto_io=False,
            resources=resources or flyte.Resources(cpu=1, memory="1Gi"),
            retries=retries,
            timeout=timeout,
            env_vars=env_vars,
            secrets=secrets,
            cache=cache,
        )

        run_data = {**(self.original_samples or {}), **overrides}
        return await sandbox.run.aio(image=self.image, **run_data)


# Apply syncify after class definition to avoid Pydantic field detection
CodeGenEvalResult.run = syncify(CodeGenEvalResult.run)


class TestFailure(BaseModel):
    """Individual test failure with diagnosis."""

    test_name: str = Field(description="Name of the failing test")
    error_message: str = Field(
        description="The exact final error message from test output "
        "(e.g., 'RecursionError: maximum recursion depth exceeded')"
    )
    expected_behavior: str = Field(description="What this test expected to happen")
    actual_behavior: str = Field(description="What actually happened when the code ran")
    root_cause: str = Field(description="Why the test failed (quote the exact code that's wrong)")
    suggested_fix: str = Field(description="Specific code changes using format: Replace `current code` with `new code`")
    error_type: Literal["environment", "logic", "test_error"] = Field(
        description="Type of error: 'environment' (missing packages/dependencies), "
        "'logic' (bug in solution code), or 'test_error' (bug in test code)"
    )


class ErrorDiagnosis(BaseModel):
    """Structured diagnosis of execution errors."""

    failures: list[TestFailure] = Field(description="Individual test failures with their diagnoses")
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

    all_fixes_applied: bool = Field(description="True if all suggested fixes are present in the new code")
    applied_fixes: list[str] = Field(
        default_factory=list,
        description="List of fixes that were successfully applied (by test name)",
    )
    missing_fixes: list[str] = Field(
        default_factory=list,
        description="List of fixes that are still missing (by test name)",
    )
    explanation: str = Field(description="Brief explanation of what was checked and what's missing (if anything)")


class TestFunctionPatch(BaseModel):
    """A single fixed test function."""

    test_name: str = Field(description="Name of the test function (e.g. test_basic_analysis)")
    fixed_code: str = Field(description="Complete fixed function body including the def line and decorators")


class TestFixResponse(BaseModel):
    """Response containing only the fixed test functions."""

    patches: list[TestFunctionPatch] = Field(description="List of fixed test functions")


class _PackageReplacementResponse(BaseModel):
    """Response format for suggesting a replacement system package."""

    replacement: Optional[str] = Field(
        default=None,
        description="Correct Debian/Ubuntu apt package name, or null if no system package is needed",
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


class _ConstraintParameters(BaseModel):
    """Parameters for a constraint check. Only the fields relevant to the check_type should be set."""

    value: Optional[float] = Field(
        default=None,
        description="Threshold value for greater_than or less_than checks",
    )
    min: Optional[float] = Field(
        default=None,
        description="Minimum value for between checks",
    )
    max: Optional[float] = Field(
        default=None,
        description="Maximum value for between checks",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for regex checks",
    )
    values: Optional[list[str]] = Field(
        default=None,
        description="Allowed values for isin checks",
    )


class _ConstraintParse(BaseModel):
    """LLM response for parsing a constraint into Pandera check."""

    column_name: str = Field(description="Name of the column this constraint applies to")
    check_type: Literal["greater_than", "less_than", "between", "regex", "isin", "not_null", "none"] = Field(
        description="Type of check to apply"
    )
    parameters: _ConstraintParameters = Field(
        default_factory=_ConstraintParameters,
        description="Parameters for the check. Set only the fields relevant to the check_type.",
    )
    explanation: str = Field(description="Brief explanation of what check will be applied")
