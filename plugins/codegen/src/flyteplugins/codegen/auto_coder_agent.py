import datetime
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import flyte
import litellm
import pandas as pd
from flyte.errors import InvalidPackageError
from flyte.io import File
from flyte.sandbox import ImageConfig
from flyte.syncify import syncify

from flyteplugins.codegen.core.types import CodeGenEvalResult, CodePlan, CodeSolution
from flyteplugins.codegen.data.extraction import extract_data_context, is_dataframe
from flyteplugins.codegen.execution.agent_sdk import code_gen_eval_agent_sdk
from flyteplugins.codegen.execution.docker import build_image, run_tests
from flyteplugins.codegen.generation.llm import (
    detect_and_track_packages,
    diagnose_and_plan_environment_fix,
    fix_failing_tests,
    generate_code,
    generate_plan,
    generate_tests,
    suggest_replacement_package,
    verify_logic_fixes_applied,
    verify_test_fixes_applied,
)
from flyteplugins.codegen.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    STRUCTURED_OUTPUT_REQUIREMENTS,
    TEST_FRAMEWORKS,
    build_enhanced_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class AutoCoderAgent:
    """Agent for single-file Python code generation with automatic testing and iteration.

    Generates a single Python script, builds a sandbox image with the required
    dependencies, runs pytest-based tests, and iterates until tests pass.

    Uses Sandbox internally for isolated code execution.

    Args:
        name: Name for the agent (used in image naming and logging).
        model: LLM model to use (required). Must support structured outputs.
            For LiteLLM mode (default): e.g. "gpt-4.1", "claude-sonnet-4-20250514".
            For Agent SDK mode: a Claude model ("sonnet", "opus", "haiku").
        system_prompt: Optional system prompt to use for LLM. If not provided, a default prompt with structured output requirements is used.
        api_key: Optional environment variable name for LLM API key.
        api_base: Optional base URL for LLM API.
        litellm_params: Optional dict of additional parameters to pass to LiteLLM calls.
        base_packages: Optional list of base packages to install in the sandbox.
        resources: Optional resources for sandbox execution (default: cpu=1, 1Gi).
        image_config: Optional image configuration for sandbox execution.
        max_iterations: Maximum number of generate-test-fix iterations. Defaults to 10.
        max_sample_rows: Optional maximum number of rows to use for sample data. Defaults to 100.
        skip_tests: Optional flag to skip testing. Defaults to False.
        block_network: Allow generated code to access the network inside the sandbox.
            Defaults to False (network disabled for safety).
        sandbox_retries: Number of Flyte task-level retries for each sandbox execution. Defaults to 0.
        timeout: Timeout in seconds for sandboxes. Defaults to None.
        env_vars: Environment variables to pass to sandboxes.
        secrets: flyte.Secret objects to make available to sandboxes.
        cache: CacheRequest for sandboxes: "auto", "override", or "disable". Defaults to "auto".
        use_agent_sdk: Optional flag to use the agent SDK. Defaults to False.
        agent_sdk_max_turns: Maximum agent turns when use_agent_sdk=True. Defaults to 50.

    Example::

        from flyte.sandbox import sandbox_environment
        from flyteplugins.codegen import AutoCoderAgent

        agent = AutoCoderAgent(
            model="gpt-4.1",
            base_packages=["pandas"],
            resources=flyte.Resources(cpu=1, memory="1Gi"),
        )

        env = flyte.TaskEnvironment(
            name="my-env",
            depends_on=[sandbox_environment],
        )

        @env.task
        async def my_task(data_file: File) -> float:
            result = await agent.generate.aio(
                prompt="Process CSV data",
                samples={"csv": data_file},
                outputs={"total": float},
            )
            return await result.run.aio()
    """

    model: str
    name: str = "auto-coder"
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    litellm_params: Optional[dict] = None
    base_packages: Optional[list[str]] = None
    resources: Optional[flyte.Resources] = None
    image_config: Optional[ImageConfig] = None
    max_iterations: int = 10
    max_sample_rows: int = 100
    skip_tests: bool = False
    block_network: bool = True
    sandbox_retries: int = 0
    timeout: Optional[int] = None
    env_vars: Optional[dict[str, str]] = None
    secrets: Optional[list] = None
    cache: str = "auto"
    use_agent_sdk: bool = False
    agent_sdk_max_turns: int = 50

    @syncify
    async def generate(
        self,
        prompt: str,
        schema: Optional[str] = None,
        constraints: Optional[list[str]] = None,
        samples: Optional[dict[str, pd.DataFrame | File]] = None,
        inputs: Optional[dict[str, type]] = None,
        outputs: Optional[dict[str, type]] = None,
    ) -> CodeGenEvalResult:
        """Generate and evaluate code in an isolated sandbox.

        Each call is independent with its own sandbox, packages and execution environment.

        Args:
            prompt: The prompt to generate code from.
            schema: Optional free-form context about data formats, structures or schemas.
                Included verbatim in the LLM prompt. Use for input formats, output schemas,
                database schemas or any structural context the LLM needs to generate code.
            constraints: Optional list of constraints or requirements.
            samples: Optional dict of sample data. Each value is sampled and included in
                the LLM prompt for context, and converted to a File input for the
                sandbox. Values are used as defaults at runtime — override them
                when calling ``result.run()`` or ``result.as_task()``.
                Supported types: File, pd.DataFrame.
            inputs: Optional dict declaring non-sample CLI argument types
                (e.g., ``{"threshold": float, "mode": str}``).
                Sample entries are automatically added as File inputs — don't redeclare them here.
                Supported types: str, int, float, bool, File.
            outputs: Optional dict defining output types (e.g., ``{"result": str, "report": File}``).
                Supported types: str, int, float, bool, datetime, timedelta, File.

        Returns:
            CodeGenEvalResult with solution and execution details.
        """
        language = "python"

        # Input validation
        if inputs:
            supported_input_types = (str, int, float, bool, File)
            for input_key, input_type in inputs.items():
                if input_type not in supported_input_types:
                    supported_names = [t.__name__ for t in supported_input_types]
                    raise ValueError(
                        f"Unsupported input type for '{input_key}': {input_type}. "
                        f"Sandbox only supports: {', '.join(supported_names)}"
                    )

        # Data processing
        sample_files = None
        extracted_data_context = None
        data_schemas = {}
        schema_input_tokens = 0
        schema_output_tokens = 0

        if samples:
            logger.info(f"Processing {len(samples)} sample inputs...")
            inferred_types = {}
            sample_files = {}

            for data_key, value in samples.items():
                if isinstance(value, File):
                    inferred_types[data_key] = File
                    sample_files[data_key] = value
                elif is_dataframe(value):
                    temp_file = Path(tempfile.gettempdir()) / f"{data_key}.csv"
                    value.to_csv(temp_file, index=False)
                    file_obj = await File.from_local(str(temp_file))
                    inferred_types[data_key] = File
                    sample_files[data_key] = file_obj
                else:
                    raise ValueError(
                        f"Unsupported sample type for '{data_key}': {type(value)}. Supported: File, pd.DataFrame."
                    )

            logger.info("Extracting data context (schema, stats, patterns) and inferring Pandera schemas...")
            (
                extracted_data_context,
                data_schemas,
                schema_input_tokens,
                schema_output_tokens,
            ) = await extract_data_context(
                samples,
                self.max_sample_rows,
                constraints=constraints,
                model=self.model,
                litellm_params=self.litellm_params,
            )
            if data_schemas:
                logger.info(f"Inferred Pandera schemas for: {list(data_schemas.keys())}")

            if not inputs:
                inputs = inferred_types
                logger.info(f"Inferred input types: {inputs}")
            else:
                # Merge data-inferred types into user-provided inputs
                for key, typ in inferred_types.items():
                    if key not in inputs:
                        inputs[key] = typ
                logger.info(f"Merged input types: {inputs}")

        schemas_as_code = data_schemas or {}

        # Output validation
        if outputs:
            supported_types = (
                str,
                int,
                float,
                bool,
                datetime.datetime,
                datetime.timedelta,
                File,
            )
            for output_key, output_type in outputs.items():
                if output_type not in supported_types:
                    supported_names = [t.__name__ for t in supported_types]
                    raise ValueError(
                        f"Unsupported output type for '{output_key}': {output_type}. "
                        f"Sandbox only supports: {', '.join(supported_names)}"
                    )

        # Agent SDK routing
        if self.use_agent_sdk:
            if self.skip_tests:
                logger.warning(
                    "skip_tests is not supported with Agent SDK mode. The agent autonomously decides when to test."
                )
            logger.info("Using Claude Agent SDK approach")
            return await code_gen_eval_agent_sdk(
                name=self.name,
                model=self.model,
                prompt=prompt,
                schema=schema,
                constraints=constraints,
                inputs=inputs,
                outputs=outputs,
                original_samples=sample_files,
                data_context=extracted_data_context,
                generated_schemas=schemas_as_code or None,
                base_packages=self.base_packages,
                resources=self.resources,
                image_config=self.image_config,
                block_network=self.block_network,
                retries=self.sandbox_retries,
                timeout=self.timeout,
                env_vars=self.env_vars,
                secrets=self.secrets,
                cache=self.cache,
                max_turns=self.agent_sdk_max_turns,
            )

        logger.info(
            f"Starting code generation: language={language}, model={self.model}, max_iterations={self.max_iterations}"
        )

        # LiteLLM setup
        if self.api_key:
            litellm.api_key = os.getenv(self.api_key)
        if self.api_base:
            litellm.api_base = self.api_base

        # Build prompts
        base_prompt_text = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        final_system_prompt = f"{base_prompt_text}\n{STRUCTURED_OUTPUT_REQUIREMENTS}"
        enhanced_prompt = build_enhanced_prompt(
            prompt,
            language,
            schema,
            constraints,
            extracted_data_context,
            inputs,
            outputs,
        )
        base_messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": enhanced_prompt},
        ]

        # Generate plan
        logger.info("Generating plan...")
        plan, in_tok, out_tok = await generate_plan(
            self.model,
            prompt,
            language,
            schema,
            constraints,
            extracted_data_context,
            inputs,
            outputs,
            self.litellm_params,
        )
        logger.info(f"Plan created: {plan.description}")
        logger.info(f"Approach: {plan.approach}")

        # Prepare base packages
        base_pkgs = list(self.base_packages or [])
        if not self.skip_tests:
            test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])
            for pkg in test_framework_info["packages"]:
                if pkg not in base_pkgs:
                    base_pkgs.append(pkg)

        # Run iteration loop
        session = _CodeGenSession(
            agent=self,
            language=language,
            prompt=prompt,
            schema=schema,
            constraints=constraints,
            inputs=inputs,
            outputs=outputs,
            base_packages=base_pkgs,
            extracted_data_context=extracted_data_context,
            sample_files=sample_files,
            schemas_as_code=schemas_as_code,
            base_messages=base_messages,
            plan=plan,
            skip_tests=self.skip_tests,
            block_network=self.block_network,
            initial_input_tokens=(schema_input_tokens + in_tok) if samples else in_tok,
            initial_output_tokens=((schema_output_tokens + out_tok) if samples else out_tok),
        )
        return await session.run()


class _CodeGenSession:
    """Internal: manages mutable state for a single LiteLLM code generation run.

    Encapsulates the retry loop, code/test generation, image building,
    error diagnosis and reclassification logic.
    """

    def __init__(
        self,
        *,
        agent: AutoCoderAgent,
        language: str,
        prompt: str,
        schema: Optional[str],
        constraints: Optional[list[str]],
        inputs: Optional[dict[str, type]],
        outputs: Optional[dict[str, type]],
        base_packages: list[str],
        extracted_data_context: Optional[str],
        sample_files: Optional[dict[str, File]],
        schemas_as_code: dict[str, str],
        base_messages: list[dict[str, str]],
        plan: CodePlan,
        skip_tests: bool,
        block_network: bool,
        initial_input_tokens: int,
        initial_output_tokens: int,
    ):
        # Agent reference (immutable config)
        self.agent = agent
        self.name = agent.name
        self.model = agent.model
        self.max_iterations = 1 if skip_tests else agent.max_iterations
        self.skip_tests = skip_tests
        self.block_network = block_network
        self.resources = agent.resources
        self.sandbox_retries = agent.sandbox_retries
        self.timeout = agent.timeout
        self.env_vars = agent.env_vars
        self.secrets = agent.secrets
        self.cache = agent.cache
        self.image_config = agent.image_config
        self.litellm_params = agent.litellm_params

        # Per-call config (immutable)
        self.language = language
        self.prompt = prompt
        self.schema = schema
        self.constraints = constraints
        self.inputs = inputs
        self.outputs = outputs
        self.extracted_data_context = extracted_data_context
        self.sample_files = sample_files
        self.schemas_as_code = schemas_as_code
        self.base_messages = base_messages
        self.plan = plan

        # Package state
        self.base_pkgs = base_packages
        self.detected_packages: list[str] = []
        self.detected_system_packages: list[str] = []
        self.additional_commands: list[str] = []
        self.previously_installed_packages: list[str] = []
        self.previously_installed_system_packages: list[str] = []

        # Image state
        self.current_image: Optional[str] = None
        self.image_name = self._compute_image_name(self.base_pkgs, [])

        # Generation state
        self.solution: Optional[CodeSolution] = None
        self.tests: Optional[str] = None
        self.needs_new_code = True
        self.needs_new_tests = True
        self.needs_rebuild = True
        self.last_packages_snapshot: tuple[set, set] = (set(), set())

        # Error tracking
        self.last_error: Optional[str] = None
        self.last_error_message: Optional[str] = None
        self.last_diagnosis = None
        self.last_result: Optional[CodeGenEvalResult] = None

        # Reclassification tracking
        self.logic_fix_attempts: dict[tuple, int] = {}
        self.test_fix_attempts: dict[tuple, int] = {}
        self.max_logic_attempts = 1
        self.max_test_attempts = 1

        # Token tracking
        self.total_input_tokens = initial_input_tokens
        self.total_output_tokens = initial_output_tokens

        # Add test framework system packages
        if not self.skip_tests:
            test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])
            for pkg in test_framework_info.get("system_packages", []):
                if pkg not in self.detected_system_packages:
                    self.detected_system_packages.append(pkg)

    def _compute_image_name(self, packages: list[str], system_packages: list[str]) -> str:
        spec = {
            "language": self.language,
            "packages": sorted(packages),
            "system_packages": sorted(system_packages),
        }
        config_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        return f"auto-coder-agent-{self.language}-{config_hash}"

    def _track_tokens(self, in_tok: int, out_tok: int):
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

    def _make_result(
        self,
        *,
        success: bool,
        test_output: str,
        exit_code: int,
        attempt: int,
        error: Optional[str] = None,
    ) -> CodeGenEvalResult:
        return CodeGenEvalResult(
            plan=self.plan,
            solution=self.solution or CodeSolution(),
            tests=self.tests,
            success=success,
            output=test_output,
            exit_code=exit_code,
            error=error,
            attempts=attempt,
            conversation_history=self.base_messages,
            detected_packages=self.detected_packages,
            detected_system_packages=self.detected_system_packages,
            image=self.current_image,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            declared_inputs=self.inputs,
            declared_outputs=self.outputs,
            data_context=self.extracted_data_context,
            original_samples=self.sample_files,
            generated_schemas=self.schemas_as_code or None,
        )

    async def run(self) -> CodeGenEvalResult:
        """Execute the full retry loop."""
        for attempt in range(1, self.max_iterations + 1):
            logger.info(
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"[ITERATION] Starting attempt {attempt}/{self.max_iterations}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            try:
                result = await self._attempt(attempt)
                if result is not None:
                    return result
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Error during attempt {attempt}: {self.last_error}")
                self.last_error_message = f"An error occurred: {self.last_error}"
                self.last_result = self._make_result(
                    success=False,
                    test_output="",
                    exit_code=-1,
                    attempt=attempt,
                    error=f"Attempt {attempt} failed: {self.last_error}",
                )
                if attempt == self.max_iterations:
                    return self.last_result
                self.needs_new_code = True
                if self.tests is None:
                    self.needs_new_tests = True

        return self.last_result or self._make_result(
            success=False,
            test_output="",
            exit_code=-1,
            attempt=self.max_iterations,
            error=f"All {self.max_iterations} attempts failed: {self.last_error}",
        )

    async def _attempt(self, attempt: int) -> Optional[CodeGenEvalResult]:
        """Run a single iteration. Returns result on success, None to continue."""
        # 1. Generate code (only when needed)
        if self.needs_new_code:
            logger.info("Generating code...")
            if not await self._generate_code(attempt):
                return None  # Skip to next iteration

            # Detect and track packages
            (
                self.needs_rebuild,
                self.detected_packages,
                self.detected_system_packages,
                in_tok,
                out_tok,
            ) = await detect_and_track_packages(
                self.model,
                self.solution,
                self.base_pkgs,
                self.detected_packages,
                self.detected_system_packages,
                self.litellm_params,
            )
            self._track_tokens(in_tok, out_tok)
            self.needs_new_code = False
            if self.tests is None:
                self.needs_new_tests = True

        # Short-circuit: skip tests if requested
        if self.skip_tests:
            logger.info("skip_tests=True: skipping test generation and execution.")
            # Still build the image (needed for as_task() and run())
            self._update_image_name_if_needed()
            if self.needs_rebuild or self.current_image is None:
                await self._build_image()
            return self._make_result(
                success=True,
                test_output="",
                exit_code=0,
                attempt=attempt,
            )

        # 2. Generate tests (only when needed)
        if self.needs_new_tests:
            logger.info("Generating tests...")
            self.tests, in_tok, out_tok = await generate_tests(
                self.model,
                self.prompt,
                self.plan,
                self.solution,
                self.constraints,
                self.schema,
                self.extracted_data_context,
                self.inputs,
                self.outputs,
                self.litellm_params,
            )
            self._track_tokens(in_tok, out_tok)
            self.needs_new_tests = False

        # 3. Update image name if packages changed
        self._update_image_name_if_needed()

        # 4. Build/rebuild image if needed
        if self.needs_rebuild or self.current_image is None:
            await self._build_image()

        # 5. Execute tests
        logger.info("Running tests...")
        run_tests_output = await run_tests.aio(
            code=self.solution.code,
            tests=self.tests,
            image=self.current_image,
            name=self.name,
            resources=self.resources,
            block_network=self.block_network,
            retries=self.sandbox_retries,
            timeout=self.timeout,
            env_vars=self.env_vars,
            secrets=self.secrets,
            cache=self.cache,
            _attempt=attempt,
        )

        tests_passed, test_output, test_exit_code = (
            run_tests_output.tests_passed,
            run_tests_output.output,
            run_tests_output.exit_code,
        )

        # 6. Handle success
        if tests_passed:
            logger.info("Tests passed! Solution successful.")
            logger.info(f"Total tokens: input={self.total_input_tokens}, output={self.total_output_tokens}")
            self.last_diagnosis = None
            return self._make_result(
                success=True,
                test_output=test_output,
                exit_code=int(test_exit_code.strip()),
                attempt=attempt,
            )

        # 7. Handle failure
        return await self._handle_failure(test_output, test_exit_code, attempt)

    async def _generate_code(self, attempt: int) -> bool:
        """Generate code with up to 3 verification attempts. Returns True on success."""
        max_attempts = 3

        for code_attempt in range(1, max_attempts + 1):
            logger.info(f"Generating code (attempt {attempt}, code gen {code_attempt}/{max_attempts})...")

            # Build messages with progressively forceful error context
            messages = self.base_messages.copy()
            if self.last_error_message:
                if code_attempt == 1:
                    messages.append({"role": "user", "content": self.last_error_message})
                elif code_attempt == 2:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{self.last_error_message}\n\n"
                                "CRITICAL: The previous code generation attempt did NOT apply all the required fixes.\n"
                                "You MUST apply EVERY SINGLE fix listed above. Do not skip any fix.\n"
                                "Apply each fix EXACTLY as specified - find the old code and replace it with the new code."
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{self.last_error_message}\n\n"
                                "FINAL ATTEMPT: You have failed to apply the required fixes twice.\n"
                                "This is your last chance. Apply EVERY fix listed above WITHOUT EXCEPTION.\n"
                                "For each fix, you MUST:\n"
                                "1. Find the EXACT old code mentioned\n"
                                "2. Replace it with the EXACT new code mentioned\n"
                                "3. Do NOT change anything else\n\n"
                                "If you fail to apply all fixes this time, the entire task will fail."
                            ),
                        }
                    )

            self.solution, in_tok, out_tok = await generate_code(
                self.model,
                messages,
                self.plan,
                self.litellm_params,
                is_retry=(attempt > 1),
            )
            self._track_tokens(in_tok, out_tok)

            if self.solution.language.lower() != self.language.lower():
                logger.warning(f"Requested {self.language} but LLM generated {self.solution.language}")

            # Verify fixes are applied (if we have a diagnosis with logic failures)
            has_logic_failures = self.last_diagnosis and any(
                f.error_type == "logic" for f in self.last_diagnosis.failures
            )
            if not has_logic_failures:
                return True

            logger.info("Verifying that logic fixes were applied...")
            verification, in_tok, out_tok = await verify_logic_fixes_applied(
                self.model,
                self.last_diagnosis,
                self.solution,
                self.litellm_params,
            )
            self._track_tokens(in_tok, out_tok)

            if verification.all_fixes_applied:
                logger.info(f"Verification passed: All fixes applied. {verification.explanation}")
                return True

            logger.warning(f"Verification failed: {verification.explanation}")
            logger.warning(f"Applied: {verification.applied_fixes}")
            logger.warning(f"Missing: {verification.missing_fixes}")

            if code_attempt < max_attempts:
                # Append verification feedback for next attempt
                missing_msg = "\n\nVERIFICATION FAILED - The following fixes are STILL MISSING:\n"
                for i, fix in enumerate(verification.missing_fixes, 1):
                    missing_msg += f"\n{i}. {fix}"
                missing_msg += "\n\nYou successfully applied these fixes:\n"
                for fix in verification.applied_fixes:
                    missing_msg += f"- {fix}\n"
                missing_msg += "\nYou MUST now apply the MISSING fixes listed above. Do NOT regenerate the entire solution - just apply the missing fixes to your previous code."
                self.last_error_message = (self.last_error_message or "") + missing_msg
            else:
                logger.error(f"Failed to apply all fixes after {max_attempts} attempts. Proceeding anyway...")
                return True  # Proceed anyway

        logger.error("Failed to generate code with all fixes applied. Skipping this iteration.")
        return False

    def _update_image_name_if_needed(self):
        """Check if packages changed and update image name."""
        current_snapshot = (
            set(self.detected_packages),
            set(self.detected_system_packages),
        )
        if current_snapshot == self.last_packages_snapshot:
            return

        self.needs_rebuild = True
        self.last_packages_snapshot = current_snapshot

        all_packages = self.base_pkgs + self.detected_packages
        new_name = self._compute_image_name(all_packages, self.detected_system_packages)
        if new_name != self.image_name:
            logger.info(f"Image name updated: {self.image_name} -> {new_name}")
            self.image_name = new_name
            self.current_image = None

    async def _build_image(self):
        """Build/rebuild image with package retry loop."""
        max_retries = 3
        for _ in range(max_retries):
            try:
                self.current_image = await build_image(
                    self.solution.language,
                    self.base_pkgs,
                    self.detected_packages,
                    self.detected_system_packages,
                    self.previously_installed_packages,
                    self.previously_installed_system_packages,
                    self.additional_commands,
                    self.image_name,
                    self.current_image,
                    self.image_config,
                )
                self.previously_installed_packages = self.detected_packages.copy()
                self.previously_installed_system_packages = self.detected_system_packages.copy()
                self.needs_rebuild = False
                return
            except InvalidPackageError as e:
                bad_package = e.package_name
                logger.warning(f"Invalid system package '{bad_package}', asking LLM for replacement...")
                if bad_package in self.detected_system_packages:
                    self.detected_system_packages.remove(bad_package)
                if bad_package in self.previously_installed_system_packages:
                    self.previously_installed_system_packages.remove(bad_package)

                # Ask LLM for the correct package name
                solution_code = self.solution.code
                replacement, in_tok, out_tok = await suggest_replacement_package(
                    self.model,
                    bad_package,
                    e.original_error,
                    solution_code,
                    self.litellm_params,
                )
                self.total_input_tokens += in_tok
                self.total_output_tokens += out_tok

                if replacement and replacement not in self.detected_system_packages:
                    self.detected_system_packages.append(replacement)
                    logger.info(f"Replacing '{bad_package}' with '{replacement}'")

                logger.info(f"Retrying with system packages: {self.detected_system_packages}")

    async def _handle_failure(
        self,
        test_output: str,
        test_exit_code: str,
        attempt: int,
    ) -> Optional[CodeGenEvalResult]:
        """Handle test failure: diagnose, reclassify, fix tests or code. Returns None to continue."""
        # Check if tests actually executed
        tests_executed = (
            " passed" in test_output or " failed" in test_output or "collected 0 items" not in test_output
        ) and "ERROR collecting" not in test_output

        if not tests_executed:
            logger.warning("No tests executed - test file likely has errors. Regenerating tests...")
            self.needs_new_tests = True
            self.needs_new_code = False
            return None

        # Diagnose failures
        (
            primary_error_type,
            self.detected_packages,
            self.detected_system_packages,
            self.additional_commands,
            in_tok,
            out_tok,
            diagnosis,
        ) = await diagnose_and_plan_environment_fix(
            self.model,
            self.solution,
            test_output,
            self.prompt,
            self.plan,
            self.detected_packages,
            self.detected_system_packages,
            self.additional_commands,
            self.litellm_params,
            self.tests,
            self.extracted_data_context,
            self.constraints,
            self.schema,
        )
        self._track_tokens(in_tok, out_tok)
        self.last_diagnosis = diagnosis

        # Apply environment fixes from diagnosis
        self._apply_environment_fixes(diagnosis)

        # Reclassify repeated errors
        primary_error_type = self._reclassify_errors(diagnosis, primary_error_type)

        # Handle test errors: fix tests
        if primary_error_type == "test_error":
            return await self._handle_test_errors(diagnosis, attempt)

        # Handle logic/environment errors: build patch message
        return self._handle_logic_env_errors(diagnosis, test_output, test_exit_code, attempt)

    def _apply_environment_fixes(self, diagnosis):
        """Extract and apply environment fixes from diagnosis."""
        if diagnosis.needs_language_packages:
            added = [
                p
                for p in diagnosis.needs_language_packages
                if p not in self.detected_packages and p not in self.base_pkgs
            ]
            if added:
                self.detected_packages.extend(added)
                logger.info(f"Adding language packages from diagnosis: {added}")
                self.needs_rebuild = True

        if diagnosis.needs_system_packages:
            added = [p for p in diagnosis.needs_system_packages if p not in self.detected_system_packages]
            if added:
                self.detected_system_packages.extend(added)
                logger.info(f"Adding system packages from diagnosis: {added}")
                self.needs_rebuild = True

        if diagnosis.needs_additional_commands:
            logger.info(f"Adding additional commands from diagnosis: {diagnosis.needs_additional_commands}")
            self.additional_commands.extend(diagnosis.needs_additional_commands)
            self.needs_rebuild = True

    def _reclassify_errors(self, diagnosis, primary_error_type: str) -> str:
        """Reclassify repeated errors (test_error <-> logic). Returns updated primary_error_type."""
        # test_error -> logic (test might be correct, code is wrong)
        reclassified = 0
        for failure in diagnosis.failures:
            if failure.error_type != "test_error":
                continue
            sig = failure.error_message or failure.actual_behavior
            key = (failure.test_name, sig)
            self.test_fix_attempts[key] = self.test_fix_attempts.get(key, 0) + 1

            if self.test_fix_attempts[key] > self.max_test_attempts:
                original = failure.root_cause
                failure.error_type = "logic"
                failure.root_cause = (
                    f"Test failed {self.max_test_attempts + 1} times with same error after test fixes. "
                    f"The test expectations are likely correct. The code logic could be wrong. "
                    f"Original test diagnosis was: {original}"
                )
                failure.suggested_fix = (
                    f"Fix the code logic to match test expectations. "
                    f"Test expects: {failure.expected_behavior}. "
                    f"Code produces: {failure.actual_behavior}. "
                    f"Update the code to produce the expected behavior."
                )
                self.test_fix_attempts.pop(key, None)
                reclassified += 1
                logger.warning(f"Reclassified test_error -> logic for '{failure.test_name}'")

        if reclassified:
            logger.info(f"Reclassified {reclassified} test_error(s) to logic.")

        # logic -> test_error (LLM might misdiagnose test bugs as logic bugs)
        reclassified = 0
        for failure in diagnosis.failures:
            if failure.error_type != "logic":
                continue
            sig = failure.error_message or failure.actual_behavior
            key = (failure.test_name, sig)
            self.logic_fix_attempts[key] = self.logic_fix_attempts.get(key, 0) + 1

            if self.logic_fix_attempts[key] > self.max_logic_attempts:
                original = failure.root_cause
                failure.error_type = "test_error"
                failure.root_cause = (
                    f"Test failed {self.max_logic_attempts + 1} times with same error after logic fixes. "
                    f"Likely the test itself has wrong expected values, not the code. "
                    f"Original diagnosis was: {original}"
                )
                failure.suggested_fix = (
                    f"Fix the test expectations to match actual correct behavior. "
                    f"Code produces: {failure.actual_behavior}. "
                    f"If this is correct, update the test to expect this value instead."
                )
                self.logic_fix_attempts.pop(key, None)
                reclassified += 1
                logger.warning(f"Reclassified logic -> test_error for '{failure.test_name}'")

        if reclassified:
            logger.info(f"Reclassified {reclassified} logic error(s) to test_error.")
            has_test_errors = any(f.error_type == "test_error" for f in diagnosis.failures)
            if has_test_errors:
                diagnosis.failures = [f for f in diagnosis.failures if f.error_type == "test_error"]
                primary_error_type = "test_error"
                logger.info(f"After reclassification: {len(diagnosis.failures)} test_error failure(s).")

        return primary_error_type

    async def _handle_test_errors(self, diagnosis, attempt: int) -> Optional[CodeGenEvalResult]:
        """Fix failing tests with up to 3 verification attempts. Returns None to continue."""
        logger.info("Diagnosis identified bug in test code. Fixing only failed tests...")
        logger.info(f"Failed tests to fix: {[f.test_name for f in diagnosis.failures]}")

        max_attempts = 3

        for fix_attempt in range(1, max_attempts + 1):
            logger.info(f"Fixing failing tests (attempt {fix_attempt}/{max_attempts})...")

            self.tests, patches, in_tok, out_tok = await fix_failing_tests(
                self.model,
                self.tests,
                diagnosis,
                self.solution,
                self.litellm_params,
            )
            self._track_tokens(in_tok, out_tok)

            logger.info("Verifying that test fixes were applied...")
            verification, in_tok, out_tok = await verify_test_fixes_applied(
                self.model,
                diagnosis,
                patches,
                self.litellm_params,
            )
            self._track_tokens(in_tok, out_tok)

            if verification.all_fixes_applied:
                logger.info(f"Verification passed: All test fixes applied. {verification.explanation}")
                self.needs_new_code = False
                self.last_diagnosis = None
                return None  # Continue to next iteration with fixed tests

            logger.warning(f"Verification failed: {verification.explanation}")

            if fix_attempt < max_attempts:
                missing_msg = "\n\nVERIFICATION FAILED - The following test fixes are STILL MISSING:\n"
                for i, fix in enumerate(verification.missing_fixes, 1):
                    missing_msg += f"\n{i}. {fix}"
                missing_msg += "\n\nYou successfully applied these fixes:\n"
                for fix in verification.applied_fixes:
                    missing_msg += f"- {fix}\n"
                missing_msg += "\nYou MUST now apply the MISSING test fixes listed above."

                forceful = ""
                if fix_attempt == 2:
                    forceful = "\n\nCRITICAL: The previous test fix attempt did NOT apply all the required fixes. You MUST apply EVERY SINGLE fix listed above. Do not skip any fix."
                elif fix_attempt >= 3:
                    forceful = "\n\nFINAL ATTEMPT: You have failed to apply the required test fixes twice. This is your last chance. Apply EVERY fix listed above WITHOUT EXCEPTION."

                for failure in diagnosis.failures:
                    failure.suggested_fix = f"{failure.suggested_fix}\n\n{missing_msg}{forceful}"
            else:
                logger.error(f"Failed to apply all test fixes after {max_attempts} attempts. Proceeding anyway...")
                self.needs_new_code = False
                self.last_diagnosis = None
                return None

        return None

    def _handle_logic_env_errors(
        self,
        diagnosis,
        test_output: str,
        test_exit_code: str,
        attempt: int,
    ) -> Optional[CodeGenEvalResult]:
        """Handle logic and/or environment errors. Returns None to continue, result if max retries."""
        failures_info = []
        logic_count = 0
        env_count = 0

        for i, failure in enumerate(diagnosis.failures, 1):
            failures_info.append(
                f"\nTest {i} [{failure.error_type}] - {failure.test_name}\n"
                f"- Expected: {failure.expected_behavior}\n"
                f"- Actual: {failure.actual_behavior}\n"
                f"- Root cause: {failure.root_cause}\n"
                f"- FIX: {failure.suggested_fix}"
            )
            if failure.error_type == "logic":
                logic_count += 1
            elif failure.error_type == "environment":
                env_count += 1

        if logic_count > 0 and env_count > 0:
            logger.info(f"Will fix {env_count} environment error(s) and patch code for {logic_count} logic error(s)")
        elif logic_count > 0:
            logger.info(f"Will patch code for {logic_count} logic error(s)")
        elif env_count > 0:
            logger.info(f"Will fix {env_count} environment error(s)")

        full_code = self.solution.code
        error_msg = (
            "Tests failed. Apply only the specific fixes below to your code.\n\n"
            "Do not regenerate from scratch. PATCH the code by applying ONLY the fixes below.\n"
            "Do NOT make any other changes - keep everything else exactly as is.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. /var/outputs is a PRE-EXISTING directory. NEVER delete, recreate, or modify it. "
            "NEVER use shutil.rmtree or os.makedirs on /var/outputs. Only write files into it using: "
            "open('/var/outputs/<name>', 'w').write(str(value)). Always use the literal path '/var/outputs' "
            "-- never make it configurable or store it in a variable.\n"
            "2. If a part of the code is working correctly, DO NOT change it. Only fix what's broken.\n"
            "3. Apply each fix by finding the exact code quoted and replacing it - nothing more.\n"
            "4. Do NOT regenerate the entire code. Just apply the specific patches mentioned below.\n\n"
            f"Your previous code:\n```{self.solution.language}\n{full_code}\n```\n\n" + "\n".join(failures_info)
        )

        if logic_count > 0:
            logger.info("Tests failed. Will patch code with fixes (keeping same tests)...")
            self.last_error_message = error_msg
        else:
            self.last_error_message = None

        self.last_result = self._make_result(
            success=False,
            test_output=test_output,
            exit_code=int(test_exit_code.strip()),
            attempt=attempt,
            error=error_msg,
        )

        if attempt == self.max_iterations:
            return self.last_result

        # Set flags for next iteration
        if logic_count > 0:
            self.needs_new_code = True
            self.needs_new_tests = False
        elif env_count > 0:
            logger.info("Only environment errors - skipping code regeneration, will rebuild image with new packages")
            self.needs_new_code = False
            self.needs_new_tests = False

        return None
