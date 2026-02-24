import hashlib
import logging
import shlex
from pathlib import Path
from typing import Optional

import flyte
from flyte.errors import InvalidPackageError
from flyte.io import File
from flyte.sandbox import ImageConfig

from flyteplugins.codegen.core.types import CodeGenEvalResult, CodePlan, CodeSolution
from flyteplugins.codegen.execution.docker import build_image, run_tests
from flyteplugins.codegen.generation.prompts import build_enhanced_prompt

logger = logging.getLogger(__name__)


SAFE_PREFIXES = {
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "wc",
    "mkdir",
    "touch",
    "rm",
    "mv",
    "cp",
    "echo",
    "sed",
    "awk",
    "find",
}


def _classify_bash_command(cmd: str) -> str:
    try:
        tokens = shlex.split(cmd)
    except Exception:
        return "deny"

    if not tokens:
        return "deny"

    prog = tokens[0]

    if prog == "pytest" or "pytest" in tokens:
        return "pytest"

    # Allow safe workspace ops
    if prog in SAFE_PREFIXES:
        return "allow"

    # Deny others
    return "deny"


async def code_gen_eval_agent_sdk(
    name: str,
    model: str,
    prompt: str,
    schema: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    inputs: Optional[dict[str, type]] = None,
    outputs: Optional[dict[str, type]] = None,
    original_samples: Optional[dict[str, File]] = None,
    data_context: Optional[str] = None,
    generated_schemas: Optional[dict[str, str]] = None,
    base_packages: Optional[list[str]] = None,
    resources: Optional[flyte.Resources] = None,
    image_config: Optional[ImageConfig] = None,
    block_network: bool = True,
    retries: int = 0,
    timeout: Optional[int] = None,
    env_vars: Optional[dict[str, str]] = None,
    secrets: Optional[list] = None,
    cache: str = "auto",
    max_turns: int = 50,
    language: str = "python",
) -> CodeGenEvalResult:
    """Generate single-file Python code using Claude Agent SDK.

    Runs an autonomous Claude agent that generates a single Python script,
    writes tests, builds sandbox images, and iterates until tests pass.

    Args:
        name: Unique name for this task. Used for workspace isolation, sandbox image names, etc.
        model: Claude model to use (e.g. "sonnet", "opus", "haiku").
        prompt: Task description
        schema: Optional external schema definition (e.g., target database schema)
        constraints: Optional constraints
        inputs: Optional input types
        outputs: Optional output types
        original_samples: Optional sample data files (defaults for result.run()/as_task())
        data_context: Optional extracted data context string
        generated_schemas: Optional Pandera schemas as Python code strings
        base_packages: Optional base packages to always include
        resources: Optional resources for sandbox execution
        image_config: Optional image configuration (registry, python_version, etc.)
        block_network: Allow generated code to access the network inside the sandbox
        retries: Number of retries for sandbox execution (agent iterations)
        timeout: Timeout for sandbox execution in seconds
        env_vars: Optional environment variables to set in the sandbox
        secrets: Optional secrets to make available in the sandbox
        cache: Caching behavior for sandbox execution ("auto", "override", "disable")
        max_turns: Maximum number of agent turns before stopping (default: 50)
        language: Programming language for code generation (default: "python")

    Returns:
        CodeGenEvalResult with generated solution
    """
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

    logger.info(f"Starting Agent SDK code generation for task: {name}")

    _task_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
    workspace = Path(f"/tmp/codegen-{_task_hash}")
    workspace.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    tool_calls: list[str] = []

    # Mutable dicts for closure mutation
    _exec_state = {"test_count": 0}  # incremented only when sandbox actually runs
    _sandbox_state = {
        "packages": [],
        "system_packages": [],
        "image": None,
    }

    base_pkgs = (base_packages or []) + ["pytest"]

    # Convert inputs/outputs to type-name dicts for prompt building
    inputs_for_prompt = (
        {k: t.__name__ if hasattr(t, "__name__") else str(t) for k, t in inputs.items()} if inputs else None
    )
    outputs_for_prompt = (
        {k: t.__name__ if hasattr(t, "__name__") else str(t) for k, t in outputs.items()} if outputs else None
    )

    def _build_tool_detail(tool_name: str, raw_input: dict) -> str:
        if tool_name == "Bash":
            return raw_input.get("command", "")
        elif tool_name in ("Write", "Read", "Edit"):
            return raw_input.get("file_path", "")
        return ""

    def _read_package_file(filename: str) -> list[str]:
        """Read a package file (one package per line) from the output directory."""
        path = workspace / filename
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text().split("\n") if line.strip()]

    async def _run_tests_in_sandbox() -> tuple[str, int]:
        """Build image from packages.txt/system_packages.txt and run tests in sandbox."""
        packages = _read_package_file("packages.txt")
        system_packages = _read_package_file("system_packages.txt")

        # Read solution and tests from agent output
        solution_content = (workspace / "solution.py").read_text()
        tests_content = (workspace / "tests.py").read_text()

        detected = [p for p in packages if p not in base_pkgs]
        all_packages = base_pkgs + detected

        built_image = await build_image(
            language=language,
            base_pkgs=base_pkgs,
            detected_packages=detected,
            detected_system_packages=system_packages,
            previously_installed_packages=[],
            previously_installed_system_packages=[],
            additional_commands=[],
            image_name=f"sandbox-{name}",
            current_image=None,
            image_config=image_config,
        )

        _exec_state["test_count"] += 1
        run_tests_output = await run_tests.aio(
            code=solution_content,
            tests=tests_content,
            image=built_image,
            name=f"sandbox-{name}",
            resources=resources,
            block_network=block_network,
            retries=retries,
            timeout=timeout,
            env_vars=env_vars,
            secrets=secrets,
            cache=cache,
            _attempt=_exec_state["test_count"],
        )

        test_exit_code, test_output = (
            run_tests_output.exit_code,
            run_tests_output.output,
        )

        exit_code = int(test_exit_code.strip()) if test_exit_code.strip() else -1

        # Update sandbox state
        _sandbox_state["image"] = built_image
        _sandbox_state["packages"] = all_packages
        _sandbox_state["system_packages"] = system_packages

        # Checkpoint image reference so it can be restored on retry/cache-hit paths
        # where _run_tests_in_sandbox is never called (e.g. continue: False).
        try:
            img_ref_file = workspace / ".image_ref"
            img_ref_file.write_text(str(built_image))
            await File.from_local(
                str(img_ref_file),
                remote_destination=File.named_remote(f"{_task_hash}-image_ref").path,
            )
        except Exception as e:
            logger.warning(f"Failed to checkpoint image ref: {e}")

        return test_output, exit_code

    _CHECKPOINT_FILES = (
        "solution.py",
        "tests.py",
        "packages.txt",
        "system_packages.txt",
    )

    async def on_user_prompt_submit(
        input_data: Optional[dict],
        tool_use_id: Optional[str],
        context: Optional[dict],
    ) -> dict:
        """Restore workspace from named-remote checkpoints before the agent starts.

        File.named_remote() produces the same deterministic remote path for a given
        name within a task execution, so uploads from attempt N are visible on attempt
        N+1. On first run nothing has been uploaded yet, so download fails silently and
        agent starts fresh. On retry, workspace is restored, agent resumes from
        the last known state instead of regenerating from scratch.
        """
        restored: list[str] = []
        for filename in _CHECKPOINT_FILES:
            logger.info(f"Checking for checkpoint file in remote: {_task_hash}-{filename}")
            remote = File.named_remote(f"{_task_hash}-{filename}")
            logger.info(f"Path: {remote.path}, exists: {await remote.exists()}")

            if await remote.exists():
                logger.info(f"Restoring {filename} from checkpoint remote storage")
                await remote.download(str(workspace / filename))
                restored.append(filename)

        if "solution.py" not in restored:
            return {}

        exit_code_val = -1
        remote_exit = File.named_remote(f"{_task_hash}-exit_code")
        if await remote_exit.exists():
            await remote_exit.download(str(workspace / "exit_code"))
            exit_code_val = int((workspace / "exit_code").read_text().strip())

        remote_img = File.named_remote(f"{_task_hash}-image_ref")
        if await remote_img.exists():
            img_ref_file = workspace / ".image_ref"
            await remote_img.download(str(img_ref_file))
            _sandbox_state["image"] = img_ref_file.read_text().strip()

        remote_result = File.named_remote(f"{_task_hash}-result")
        if await remote_result.exists():
            await remote_result.download(str(workspace / "result"))

        if exit_code_val == 0:
            logger.info("Tests already passed from prior run. Skipping agent execution and returning cached results.")
            return {
                "continue": False,
                "stopReason": "Tests already passed from prior run.",
            }

        existing = [f for f in ("solution.py", "tests.py", "packages.txt") if f in restored]
        ctx = f"Existing workspace files: {', '.join(existing)}."

        logger.info(f"Restored workspace from checkpoints: {ctx}")

        if exit_code_val != -1:
            ctx += f" Last test exit_code: {exit_code_val}."
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": ctx,
            }
        }

    async def on_pre_tool_use(
        input_data: Optional[dict[str, str | object]],
        tool_use_id: Optional[str],
        context: Optional[dict],
    ) -> dict:
        """PreToolUse: checkpoint writes to named remote; run sandbox directly."""
        if not input_data:
            return {}

        tool_name = input_data.get("tool_name", "unknown")
        raw_input = input_data.get("tool_input") or {}
        detail = _build_tool_detail(tool_name, raw_input)
        cmd = raw_input.get("command", "")
        action = _classify_bash_command(cmd) if cmd else "allow"

        tool_calls.append(f"{tool_name}: {detail}" if detail else tool_name)

        if action == "deny":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "This command is not allowed in the sandbox.",
                }
            }

        # If a checkpoint already exists in named remote for this file (written by a
        # previous attempt's PostToolUse), restore it to workspace and deny the write.
        # PostToolUse handles checkpointing after each successful write.
        if tool_name == "Write":
            file_path = raw_input.get("file_path", "")
            filename = Path(file_path).name if file_path else ""
            if filename in _CHECKPOINT_FILES:
                remote = File.named_remote(f"{_task_hash}-{filename}")
                if await remote.exists():
                    await remote.download(file_path)
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                "File already exists from a previous attempt. Use the existing file content."
                            ),
                        }
                    }

        # Sandbox runs as a Flyte container task, cached by Flyte's task cache.
        if tool_name == "Bash" and action == "pytest":
            has_required_files = (workspace / "solution.py").exists() and (workspace / "tests.py").exists()
            if has_required_files:
                try:
                    test_output, exit_code = await _run_tests_in_sandbox()

                    logger.info(f"Sandbox test execution: exit_code={exit_code}")
                    (workspace / "result").write_text(test_output)
                    (workspace / "exit_code").write_text(str(exit_code))

                    try:
                        await File.from_local(
                            str(workspace / "exit_code"),
                            remote_destination=File.named_remote(f"{_task_hash}-exit_code").path,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to checkpoint exit_code: {e}")

                    try:
                        await File.from_local(
                            str(workspace / "result"),
                            remote_destination=File.named_remote(f"{_task_hash}-result").path,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to checkpoint result: {e}")

                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Tests executed in isolated sandbox (exit_code={exit_code}). "
                                f"Results written to {workspace}/result. "
                                f"Read it for the test output and ignore any warnings."
                            ),
                        }
                    }
                except InvalidPackageError as e:
                    logger.warning(f"Invalid system package: {e.package_name}")
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Image build failed: system package '{e.package_name}' does not exist "
                                f"in apt repositories. Remove it from {workspace}/system_packages.txt and try again."
                            ),
                        }
                    }
                except Exception as e:
                    logger.warning(f"Sandbox execution failed: {e}")
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "Test execution failed in the sandbox.",
                        }
                    }

        return {}

    @flyte.trace
    async def trace_post_tool_use(tool_name: str, detail: str, file_path: str) -> dict:
        """Checkpoint Write/Edit results to named remote.

        Upload is inside the trace so sequence-based replay on retries dedups re-uploads:
        same file at the same sequence position results in cache hit, so body will be skipped.
        New writes/edits are at new positions, cache miss, body runs, and uploads.
        """
        filename = Path(file_path).name if file_path else ""
        local = workspace / filename
        if tool_name in ("Write", "Edit") and filename in _CHECKPOINT_FILES and local.exists():
            try:
                remote_path = File.named_remote(f"{_task_hash}-{filename}").path
                await File.from_local(
                    str(local),
                    remote_destination=remote_path,
                )
                logger.info(f"Checkpointed {filename} to named remote {remote_path} after {tool_name} tool use.")
            except Exception as e:
                logger.warning(f"Failed to checkpoint {filename}: {e}")
        return {}

    async def on_post_tool_use(
        input_data: Optional[dict[str, str | object]],
        tool_use_id: Optional[str],
        context: Optional[dict],
    ) -> dict:
        """PostToolUse: checkpoint Write/Edit results via trace."""
        if not input_data:
            return {}

        tool_name = input_data.get("tool_name", "unknown")
        raw_input = input_data.get("tool_input") or {}
        detail = _build_tool_detail(tool_name, raw_input)
        file_path = raw_input.get("file_path", "") if tool_name in ("Write", "Edit") else ""

        await trace_post_tool_use(tool_name, detail, file_path)
        return {}

    @flyte.trace
    async def trace_post_tool_use_failure(
        tool_name: str,
        error: str,
        is_interrupt: bool,
    ) -> dict[str, str | bool]:
        return {}

    async def on_post_tool_use_failure(
        input_data: Optional[dict[str, str | bool | object]],
        tool_use_id: Optional[str],
        context: Optional[dict],
    ) -> dict:
        """PostToolUseFailure: record tool errors."""
        await trace_post_tool_use_failure(
            tool_name=str(input_data.get("tool_name", "")),
            error=str(input_data.get("error", "")),
            is_interrupt=bool(input_data.get("is_interrupt", False)),
        )
        return {}

    @flyte.trace
    async def trace_stop(
        tool_calls_count: int,
        tool_calls_summary: str,
        test_execution_count: int,
    ) -> dict[str, int | str]:
        return {}

    async def on_stop(
        input_data: Optional[dict[str, str | bool]],
        tool_use_id: Optional[str],
        context: Optional[dict],
    ) -> dict:
        """Stop: checkpoint workspace files and record a summary of the agent run."""
        await trace_stop(
            tool_calls_count=len(tool_calls),
            tool_calls_summary=", ".join(tool_calls[-20:]),  # last 20 to stay bounded
            test_execution_count=_exec_state["test_count"],
        )
        return {}

    # Build the task description from user prompt + schema + constraints + data
    base_prompt = build_enhanced_prompt(
        prompt,
        language,
        schema or None,
        constraints,
        data_context or None,
        inputs_for_prompt,
        outputs_for_prompt,
    )

    # System prompt: role + workspace rules + test workflow
    system_prompt = f"""
You are an expert {language.capitalize()} code generation agent.
Your job is to write a working {language.capitalize()} solution, comprehensive tests, and iterate until all tests pass.

There are two separate environments you must understand:

## 1. AGENT WORKSPACE (where you write files)

Path: {workspace}
This is your working directory. Write all your files here:
- {workspace}/solution.py — your complete solution code
- {workspace}/tests.py — pytest-based tests
- {workspace}/packages.txt — pip dependencies (one per line, no stdlib)
- {workspace}/system_packages.txt — apt dependencies (only if needed for native libs)

## 2. SANDBOX RUNTIME (where tests and solution.py EXECUTE)

Tests and the solution run inside an isolated sandbox. The sandbox has these paths:
- /var/inputs/solution.py — your solution code is placed here automatically.
- /var/inputs/ — READ-ONLY. Input data files are also mounted here (e.g. /var/inputs/csv_data).
- /var/outputs/ — Write output files here. PRE-CREATED. NEVER delete or recreate it.
  Write outputs like: open('/var/outputs/<name>', 'w').write(str(value))

CRITICAL: {workspace} does not exist inside the sandbox. Never reference {workspace} in solution.py or tests.py.
The solution code lives at /var/inputs/solution.py inside the sandbox.

## SOLUTION RULES
- Include all imports at the top.
- Must be a runnable script with an `if __name__ == '__main__':` block.
- Use argparse with optional (--prefixed) arguments for all inputs.
  Example: parser.add_argument('--csv_data', required=True). Do not use positional arguments.
- Read input files from paths passed via argparse (at runtime these will be /var/inputs/<name>).
- Write all outputs to /var/outputs/<name>. Always use the literal path '/var/outputs'.
- Do not run solution.py directly. Only validate via pytest.

## TEST RULES
- Use pytest. Import from solution module: `from solution import ...`
- Test the full execution path end-to-end.
- If you need to run solution.py as a subprocess, use: `{language} /var/inputs/solution.py --arg value`
- Tests run in the sandbox: create test input files under /var/inputs/, run the solution, then verify /var/outputs/.

## WORKFLOW
1. Write {workspace}/solution.py (code references /var/inputs and /var/outputs)
2. Write {workspace}/tests.py (tests verify /var/outputs after running solution)
3. Write {workspace}/packages.txt (and system_packages.txt if needed)
4. Run: pytest {workspace}/tests.py -v --tb=short
   Tests run in an isolated sandbox with your packages installed.
   If the command is denied, read {workspace}/result and {workspace}/exit_code for output.
5. If tests fail: read {workspace}/result, fix code/packages, re-run. Repeat until exit code is 0."""

    # User query: the actual task + data context
    user_query = f"""Generate a {language.capitalize()} solution for the following task:

{base_prompt}"""

    if generated_schemas:
        user_query += "\n\nDATA SCHEMAS (Pandera validation schemas for your input data):\n"
        for schema_name, schema_code in generated_schemas.items():
            user_query += f"\n--- {schema_name} ---\n```python\n{schema_code}\n```\n"

    user_query += "\n\nStart by creating the solution code, then tests, then packages.txt, then run the tests."

    logger.info("Running Agent SDK...")

    async def restrict_to_workspace(
        tool_name: str,
        input_data: dict,
        context: object,
    ) -> object:
        """Deny file operations outside the workspace directory."""
        from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

        if tool_name in ("Write", "Edit", "Read"):
            file_path = input_data.get("file_path", "")
            if file_path:
                try:
                    Path(file_path).resolve().relative_to(workspace.resolve())  # noqa: ASYNC240
                except ValueError:
                    return PermissionResultDeny(
                        message=(
                            f"Access outside workspace is not allowed: {file_path}. "
                            f"Only paths under {workspace} are permitted."
                        )
                    )
        return PermissionResultAllow(updated_input=input_data)

    try:
        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            allowed_tools=["Bash", "Read", "Write", "Edit"],
            cwd=str(workspace),
            permission_mode="acceptEdits",
            max_turns=max_turns,
            can_use_tool=restrict_to_workspace,
            hooks={
                "UserPromptSubmit": [HookMatcher(hooks=[on_user_prompt_submit])],
                "PreToolUse": [HookMatcher(hooks=[on_pre_tool_use])],
                "PostToolUse": [HookMatcher(hooks=[on_post_tool_use])],
                "PostToolUseFailure": [HookMatcher(hooks=[on_post_tool_use_failure])],
                "Stop": [HookMatcher(hooks=[on_stop])],
            },
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_query)

            async for message in client.receive_response():
                # Log agent messages for debugging
                if hasattr(message, "type"):
                    logger.debug(f"Agent message: type={message.type}")
                if hasattr(message, "content"):
                    content = str(message.content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    logger.debug(f"Agent content: {content}")

        # Log what files the agent created
        existing_files = list(workspace.iterdir()) if workspace.exists() else []  # noqa: ASYNC240
        logger.info(f"Agent output directory contents: {[f.name for f in existing_files]}")

        # Read outputs
        solution_file = workspace / "solution.py"
        tests_file = workspace / "tests.py"
        result_file = workspace / "result"
        exit_code_file = workspace / "exit_code"

        # Agent must create solution.py at minimum
        if not solution_file.exists():
            raise RuntimeError(
                f"Agent did not create solution.py. "
                f"Files in output dir: {[f.name for f in existing_files]}. "
                f"Tool calls made: {len(tool_calls)} ({', '.join(tool_calls[:10])})"
            )

        # Read agent-created files (with defaults for optional ones)
        solution_content = solution_file.read_text()
        tests = tests_file.read_text() if tests_file.exists() else ""
        detected_packages = _read_package_file("packages.txt")
        detected_system_packages = _read_package_file("system_packages.txt")

        # Test-execution files are created by _run_tests_in_sandbox, not the agent
        test_output = result_file.read_text() if result_file.exists() else ""
        exit_code_text = exit_code_file.read_text().strip() if exit_code_file.exists() else ""
        exit_code = int(exit_code_text) if exit_code_text else -1

        success = exit_code == 0

        logger.info(f"Agent SDK completed: success={success}, exit_code={exit_code}")
        logger.info(f"Tool calls: {len(tool_calls)} total")

        plan = CodePlan(
            description="Agent SDK autonomous generation",
            approach="Agent explored, generated, tested, and fixed autonomously",
        )

        return CodeGenEvalResult(
            plan=plan,
            solution=CodeSolution(
                language=language,
                code=solution_content,
                system_packages=detected_system_packages,
            ),
            tests=tests,
            success=success,
            output=test_output,
            exit_code=exit_code,
            error=test_output if not success else None,
            attempts=1,
            conversation_history=[],
            detected_packages=detected_packages,
            detected_system_packages=detected_system_packages,
            image=_sandbox_state["image"] or None,
            total_input_tokens=0,
            total_output_tokens=0,
            declared_inputs=inputs,
            declared_outputs=outputs,
            data_context=data_context,
            original_samples=original_samples,
            generated_schemas=generated_schemas,
        )

    except Exception as e:
        logger.error(f"Agent SDK generation failed: {e}")
        raise
