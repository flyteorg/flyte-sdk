import ast
import json
import logging
import re
import sys
from typing import Optional

import flyte
import flyte.errors
import litellm

from flyteplugins.codegen.core.types import (
    CodePlan,
    CodeSolution,
    ErrorDiagnosis,
    FixVerification,
    TestFixResponse,
    TestFunctionPatch,
    _PackageDetectionResponse,
    _PackageReplacementResponse,
    _TestCodeResponse,
)
from flyteplugins.codegen.data.schema import extract_token_usage
from flyteplugins.codegen.generation.prompts import (
    PACKAGE_MANAGER_MAP,
    TEST_FRAMEWORKS,
    build_enhanced_prompt,
)

logger = logging.getLogger(__name__)

_PYTHON_STDLIB = sys.stdlib_module_names


def strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM output."""
    stripped = code.strip()
    # Match ```python ... ``` or ``` ... ```
    match = re.match(r"^```(?:\w+)?\s*\n(.*?)```\s*$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def filter_stdlib(packages: list[str]) -> list[str]:
    """Remove Python standard library modules from a package list."""
    return [p for p in packages if p.split(".")[0].lower() not in _PYTHON_STDLIB]


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
    base_prompt = build_enhanced_prompt(prompt, language, schema, constraints, data_samples, inputs, outputs)

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
        if "UnsupportedParamsError" in type(e).__name__ or "does not support parameters" in str(e):
            raise flyte.errors.RuntimeUserError(
                f"Model '{model}' does not support structured outputs (response_format parameter). "
                f"Please use a model that supports structured outputs like: gpt-4.1, "
                f"claude-3-5-sonnet, or similar models."
            ) from e
        raise

    # Extract token usage
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            plan_dict = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON, using fallback")
            return (
                CodePlan(description=content[:500], approach=""),
                input_tokens,
                output_tokens,
            )
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
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            solution_dict = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse code solution JSON, extracting code from raw response")
            return (
                CodeSolution(code=strip_code_fences(content)),
                input_tokens,
                output_tokens,
            )
        solution = CodeSolution(**solution_dict)
    else:
        solution = content

    # Strip code fences the model may have included
    solution.code = strip_code_fences(solution.code)
    return solution, input_tokens, output_tokens


@flyte.trace
async def detect_required_packages(
    model: str,
    code: str,
    language: str,
    litellm_params: Optional[dict],
) -> tuple[list[str], int, int]:
    """Use LLM to detect required packages from code.

    Returns:
        Tuple of (packages, input_tokens, output_tokens)
    """
    if not code.strip():
        return [], 0, 0

    package_type = PACKAGE_MANAGER_MAP.get(language.lower(), "package names for the package manager")

    detection_prompt = f"""Given this {language} code:

```{language}
{code}
```

List the {package_type} needed to install the dependencies used in this code.
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
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            result_dict = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse package detection JSON, returning empty list")
            return [], input_tokens, output_tokens
        packages = result_dict.get("packages", [])
    else:
        packages = content.packages

    return filter_stdlib(packages), input_tokens, output_tokens


async def suggest_replacement_package(
    model: str,
    bad_package: str,
    original_error: str,
    solution_code: str,
    litellm_params: Optional[dict] = None,
) -> tuple[Optional[str], int, int]:
    """Ask the LLM for the correct Debian package name to replace a bad one.

    Returns:
        Tuple of (replacement_package_name or None, input_tokens, output_tokens)
    """
    prompt = f"""The Debian/Ubuntu apt package "{bad_package}" does not exist.

Error: {original_error}

The code that needs this package:
```python
{solution_code[:1000]}
```

What is the correct Debian/Ubuntu apt package name that should be used instead?
Set replacement to the correct package name or null if no system package is needed."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    params.update(litellm_params or {})
    params["response_format"] = _PackageReplacementResponse

    try:
        response = await litellm.acompletion(**params)
        input_tokens, output_tokens = extract_token_usage(response)

        content = response.choices[0].message.content
        if isinstance(content, str):
            try:
                result = json.loads(content)
                replacement = result.get("replacement")
            except json.JSONDecodeError:
                return None, input_tokens, output_tokens
        else:
            replacement = content.replacement

        if replacement:
            logger.info(f"LLM suggested replacing '{bad_package}' with '{replacement}'")
        return replacement, input_tokens, output_tokens
    except Exception as e:
        logger.warning(f"Failed to get package replacement suggestion: {e}")
        return None, 0, 0


async def detect_and_track_packages(
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
    if solution.system_packages and solution.system_packages != detected_system_packages:
        detected_system_packages = solution.system_packages.copy()
        logger.info(f"Detected system packages: {detected_system_packages}")
        needs_rebuild = True

    # Detect language packages from code
    if solution.code.strip():
        new_packages, in_tok, out_tok = await detect_required_packages(
            model, solution.code, solution.language, litellm_params
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
                logger.info(f"Detected new {solution.language} packages: {added_packages}")
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
    """
    Generate test code to validate the solution.

    Returns:
        (test_code, input_tokens, output_tokens)
    """

    def _validate_python(code: str) -> None:
        """Raise if code is invalid or likely truncated."""
        if not code.strip():
            raise RuntimeError("Generated empty test code")

        try:
            compile(code, "test_code", "exec")
        except SyntaxError as e:
            raise RuntimeError(f"Generated test code is invalid or truncated: {e}") from e

    full_code = solution.code
    language = solution.language.lower()
    test_framework_info = TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])
    test_framework = test_framework_info["name"]

    import_instruction = (
        "Import functions/classes from solution module (e.g., "
        "'from solution import function_name'). "
        "The code file is named solution.py and is located at "
        "/var/inputs/solution.py."
    )

    test_prompt = f"""
You are generating TESTS.

Task:
{prompt}

Plan:
- Description: {plan.description}
- Approach: {plan.approach}
"""

    if schema:
        test_prompt += f"\nSchema:\n{schema}\n"

    if constraints:
        test_prompt += "\nConstraints:\n"
        for i, c in enumerate(constraints, 1):
            test_prompt += f"{i}. {c}\n"

    if data_samples:
        test_prompt += f"""
Data:
Use EXACTLY this format if referenced.
Do NOT invent additional samples.

{data_samples}
"""

    if inputs:
        test_prompt += f"\nCLI Arguments: {list(inputs.keys())}"
    if outputs:
        test_prompt += f"\nExpected outputs: {list(outputs.keys())}"

    test_prompt += f"""
Solution code:
```{solution.language}
{full_code}
```"""

    test_prompt += f"""
Test generation instructions:

Generate comprehensive but concise tests that maximize coverage through multiple complementary cases using {test_framework}, not a single large test.

Requirements:
1. Use {test_framework} framework
2. Test the FULL execution path end-to-end (not just isolated functions)
3. Use provided data AS-IS
4. {import_instruction}
5. Follow {test_framework} best practices
6. /var/outputs is a pre-existing directory — NEVER delete or recreate it.
   The solution writes output files there (one file per output),
   and tests should READ from /var/outputs to verify correctness

IMPORTANT: Do NOT wrap output in code fences. Return ONLY valid Python test code."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 5000,
        "temperature": 0.4,
    }
    params.update(litellm_params or {})

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content

    if not isinstance(content, str):
        content = str(content)

    test_code = strip_code_fences(content)

    try:
        _validate_python(test_code)
        return test_code, input_tokens, output_tokens
    except RuntimeError:
        logger.warning("Plain text test generation invalid, retrying with structured output")

    # Structured JSON fallback (optional)
    params_structured = dict(params)
    params_structured["response_format"] = _TestCodeResponse

    response = await litellm.acompletion(**params_structured)
    input_tokens2, output_tokens2 = extract_token_usage(response)

    content = response.choices[0].message.content

    if isinstance(content, str):
        try:
            data = json.loads(content)
            test_code = data["test_code"]
        except Exception as e:
            raise RuntimeError("Structured output invalid; cannot recover test code") from e
    else:
        test_code = content.test_code

    test_code = strip_code_fences(test_code)
    _validate_python(test_code)

    return (
        test_code,
        input_tokens + input_tokens2,
        output_tokens + output_tokens2,
    )


def _strip_parametrize_suffix(name: str) -> str:
    """Strip pytest parametrize suffix like [0-True-10-3] from a test name."""
    return re.sub(r"\[.*?\]$", "", name)


def apply_test_patches(original_code: str, patches: list[TestFunctionPatch]) -> str:
    """Apply test function patches to the original test file using AST-based replacement.

    Parses the original file to find function boundaries by name, then replaces
    the source lines for each patched function. Preserves imports, fixtures,
    and all non-patched code exactly.

    Args:
        original_code: The full original test file source.
        patches: List of patches, each containing a function name and its fixed source.

    Returns:
        The patched test file as a string.
    """
    tree = ast.parse(original_code)
    lines = original_code.splitlines(keepends=True)

    # Ensure last line has a newline for consistent joining
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    # Build map of function_name -> (start_line_0indexed, end_line_0indexed_exclusive)
    # We need to handle decorated functions: start from the first decorator line
    func_map: dict[str, tuple[int, int]] = {}
    all_top_level = list(ast.iter_child_nodes(tree))

    for idx, node in enumerate(all_top_level):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Start line: use decorator_list if present, otherwise the def line
            if node.decorator_list:
                start = node.decorator_list[0].lineno - 1  # 0-indexed
            else:
                start = node.lineno - 1

            # End line: next top-level node's start, or end of file
            if idx + 1 < len(all_top_level):
                next_node = all_top_level[idx + 1]
                if hasattr(next_node, "decorator_list") and next_node.decorator_list:
                    end = next_node.decorator_list[0].lineno - 1
                else:
                    end = next_node.lineno - 1
            else:
                end = len(lines)

            # Trim trailing blank lines from this function's range
            while end > start and lines[end - 1].strip() == "":
                end -= 1

            func_map[node.name] = (start, end)

    # Apply patches in reverse order (so line numbers stay valid)
    # Normalize patch names by stripping parametrize suffixes
    # If LLM returns multiple patches for the same base function, keep the last one
    patches_by_name: dict[str, TestFunctionPatch] = {}
    for p in patches:
        base_name = _strip_parametrize_suffix(p.test_name)
        if base_name in patches_by_name:
            logger.warning(
                f"Multiple patches for '{base_name}' (from '{p.test_name}'). "
                f"Keeping last patch — ensure it includes ALL parametrize fixes."
            )
        patches_by_name[base_name] = p
    replacements = []
    for name, (start, end) in func_map.items():
        if name in patches_by_name:
            replacements.append((start, end, patches_by_name[name].fixed_code))

    # Sort by start line descending so we can splice without shifting
    replacements.sort(key=lambda r: r[0], reverse=True)

    for start, end, fixed_code in replacements:
        # Ensure fixed code ends with newline
        fixed = fixed_code.rstrip("\n") + "\n"
        # Replace the lines
        lines[start:end] = [fixed]

    result = "".join(lines)

    # Validate the patched code compiles
    try:
        compile(result, "patched_test", "exec")
    except SyntaxError as e:
        logger.warning(f"Patched test code has syntax error: {e}. Returning as-is.")

    return result


@flyte.trace
async def fix_failing_tests(
    model: str,
    test_code: str,
    diagnosis: ErrorDiagnosis,
    solution: CodeSolution,
    litellm_params: Optional[dict] = None,
) -> tuple[str, list[TestFunctionPatch], int, int]:
    """Fix only the failing tests by returning patches for individual functions.

    Instead of asking the LLM to reproduce the entire test file, asks for only
    the fixed test functions and splices them into the original file.

    Args:
        test_code: Complete test file
        diagnosis: Structured diagnosis of test failures
        solution: The solution being tested

    Returns:
        Tuple of (fixed_test_code, patches, input_tokens, output_tokens)
    """
    full_code = solution.code

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

    # Extract only the failing test functions to include in the prompt
    # Strip parametrize suffixes so we match the actual function names in the AST
    failing_names = {_strip_parametrize_suffix(f.test_name) for f in diagnosis.failures}
    try:
        tree = ast.parse(test_code)
        test_lines = test_code.splitlines(keepends=True)
        all_nodes = list(ast.iter_child_nodes(tree))
        failing_snippets = []
        for idx, node in enumerate(all_nodes):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in failing_names:
                    if node.decorator_list:
                        start = node.decorator_list[0].lineno - 1
                    else:
                        start = node.lineno - 1
                    if idx + 1 < len(all_nodes):
                        next_node = all_nodes[idx + 1]
                        if hasattr(next_node, "decorator_list") and next_node.decorator_list:
                            end = next_node.decorator_list[0].lineno - 1
                        else:
                            end = next_node.lineno - 1
                    else:
                        end = len(test_lines)
                    while end > start and test_lines[end - 1].strip() == "":
                        end -= 1
                    failing_snippets.append("".join(test_lines[start:end]))
        failing_code_section = "\n\n".join(failing_snippets)
    except SyntaxError:
        # If we can't parse, fall back to sending the whole test file
        failing_code_section = test_code

    fix_prompt = f"""You are performing MINIMAL PATCH REPAIR on pytest test functions.

YOUR PRIMARY TASK: Apply the suggested fix for each failing test EXACTLY as described below.

═══════════════════════════════════════
DIAGNOSED FAILURES AND REQUIRED FIXES
═══════════════════════════════════════
{diagnosis_section}

For each failure above, the "Suggested fix" tells you EXACTLY what code change to make.
You MUST apply that specific change — do not interpret, simplify, or find an alternative approach.

═══════════════════════════════════════
FAILING TEST FUNCTIONS (ORIGINAL CODE)
═══════════════════════════════════════
```{solution.language}
{failing_code_section}
```

SOLUTION CODE (for reference only — do NOT modify):
The solution is saved as /var/inputs/solution.py. Tests import from it via `from solution import ...`.
```{solution.language}
{full_code}
```

RULES:
1. For each suggested fix, locate the exact code it refers to and apply the substitution.
2. Preserve ALL existing assertions, decorators, and test structure.
3. Add any necessary imports at the top of the function body if needed.
4. DO NOT rewrite, simplify, or restructure the test.
5. DO NOT change anything that the diagnosis does not mention.

OUTPUT FORMAT:
Return ONLY patches for functions that require modification:
- test_name: BASE function name (no [param] suffix)
- fixed_code: COMPLETE function including decorators"""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": fix_prompt}],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    params.update(litellm_params or {})
    params["response_format"] = TestFixResponse

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            result_dict = json.loads(content)
            patches = [TestFunctionPatch(**p) for p in result_dict["patches"]]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Failed to parse test fix patches JSON: {exc}")
            # Fallback: treat as a single patch with the raw content
            patches = []
            raw = strip_code_fences(content)
            for failure in diagnosis.failures:
                patches.append(TestFunctionPatch(test_name=failure.test_name, fixed_code=raw))
    else:
        patches = content.patches

    # Strip code fences from each patch's fixed_code
    for patch in patches:
        patch.fixed_code = strip_code_fences(patch.fixed_code)

    # Apply patches to original test code
    fixed_test_code = apply_test_patches(test_code, patches)

    return fixed_test_code, patches, input_tokens, output_tokens


def extract_error_messages_from_pytest(output: str) -> dict[str, str]:
    """Extract the final error message for each failed test from pytest output.

    Args:
        output: Pytest output string

    Returns:
        Dict mapping test name to error message (e.g., "RecursionError: maximum recursion depth exceeded")
    """
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
            if "Error" in error_line or "Exception" in error_line or "Failed" in error_line:
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

    full_code = solution.code

    # Build error messages section for prompt
    error_messages_section = ""
    if error_messages:
        error_messages_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "EXTRACTED ERROR MESSAGES\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        for test_name, error_msg in error_messages.items():
            error_messages_section += f"\n{test_name}: {error_msg}"

    diagnosis_prompt = f"""
You are a STRICT FAILURE ANALYSIS SYSTEM.

Your goal is to determine whether failures are caused by:
A) Solution code bugs
B) Incorrect tests (tests generated by LLM)
C) Environment/setup issues

- Tests are NOT authoritative.
- Many tests contain hallucinated expected values.
- Only blame solution code if you can PROVE a violation of the specification.

TASK: {prompt}

PLAN: {plan.description} | {plan.approach}

SOLUTION CODE (saved as /var/inputs/solution.py, tests import via `from solution import ...`):
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

MANDATORY DECISION PROCEDURE:

For EACH failing test, follow these steps IN ORDER:

STEP 0 — Execution evidence
Determine whether the solution produced observable effects.
If none are observed, consider invocation or execution issues.

STEP 1 — Environment failure
If the error indicates missing packages, import errors, file system issues,
or system configuration problems -> classify as "environment".

STEP 2 — Test contradicts spec/data/schema
Classify as "test_error" if ANY of the following are true:

- Expected value is not derivable from inputs, spec, or schema
- Test invents constants not present in problem description
- Test assumes behavior not specified
- Test contradicts constraints or data samples
- Test fails before solution logic executes
- Assertion checks wrong field/format/order
- Multiple valid outputs exist but test expects one specific value

STEP 3 — Uncertain root cause
If you CANNOT prove the solution is wrong -> classify as "test_error".

STEP 4 — Proven logic error
Classify as "logic" if you can demonstrate:

- Exact specification requirement violated
- Specific incorrect algorithm or implementation
- Output contradicts constraints despite valid test

Logic errors REQUIRE proof from the specification. Absence of proof -> NOT a logic error.

OUTPUT FORMAT: Return JSON with:

1. failures: list of objects:
    - test_name
    - error_message (copy exactly)
    - expected_behavior
    - actual_behavior
    - root_cause (must cite evidence)
    - suggested_fix ("Replace `old` with `new`")
    - error_type: "test_error" | "logic" | "environment"
2. Collect from environment errors: needs_system_packages, needs_language_packages and needs_additional_commands

CRITICAL RULE:
- If ANY test is classified as "test_error", RETURN ONLY test_error failures. Do NOT include logic diagnoses.
- /var/outputs is a pre-existing directory and exists for the entire run. NEVER delete/recreate it.
Only write files into it. Never make it configurable.
"""

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
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            result_dict = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse diagnosis JSON, returning empty diagnosis")
            return ErrorDiagnosis(failures=[]), input_tokens, output_tokens
        return ErrorDiagnosis(**result_dict), input_tokens, output_tokens
    return content, input_tokens, output_tokens


@flyte.trace
async def verify_test_fixes_applied(
    model: str,
    diagnosis: ErrorDiagnosis,
    patches: list[TestFunctionPatch],
    litellm_params: Optional[dict] = None,
) -> tuple[FixVerification, int, int]:
    """Verify that the suggested test fixes from diagnosis are present in the patches.

    First checks structurally that every failing function has a matching patch
    (accounting for parametrize suffixes). Then sends diagnosis + patches to LLM
    to verify fix content.

    Returns:
        Tuple of (FixVerification, input_tokens, output_tokens)
    """
    # Structural check: every failing test function must have a matching patch
    patch_base_names = {_strip_parametrize_suffix(p.test_name) for p in patches}
    failing_base_names = {_strip_parametrize_suffix(f.test_name) for f in diagnosis.failures}
    missing_functions = failing_base_names - patch_base_names
    if missing_functions:
        return (
            FixVerification(
                all_fixes_applied=False,
                applied_fixes=[n for n in failing_base_names if n in patch_base_names],
                missing_fixes=list(missing_functions),
                explanation=f"No patches returned for functions: {', '.join(missing_functions)}",
            ),
            0,
            0,
        )

    # Build verification prompt with only diagnosis + patches (no full files)
    fixes_to_check = []
    for i, failure in enumerate(diagnosis.failures, 1):
        fixes_to_check.append(
            f"""Fix {i} (for test: {failure.test_name}):
- Error: {failure.error_message}
- Root cause: {failure.root_cause}
- Required fix: {failure.suggested_fix}
- Verification: The old code/pattern must be REMOVED and the new code/pattern must be PRESENT"""
        )

    fixes_section = "\n\n".join(fixes_to_check)

    patches_section = []
    for patch in patches:
        patches_section.append(f"### {patch.test_name}\n```python\n{patch.fixed_code}\n```")
    patches_text = "\n\n".join(patches_section) if patches_section else "(no patches returned)"

    verify_prompt = f"""You are a CODE DIFF REVIEWER. Your job is to verify that specific code changes were actually made.

For each required fix below, check whether the EXACT code change described in "Required fix" is present in the patched function. Do NOT accept alternative approaches that "address the same issue" — the specific code transformation must be visible.

═══════════════════════════════════════
REQUIRED FIXES
═══════════════════════════════════════
{fixes_section}

═══════════════════════════════════════
PATCHED TEST FUNCTIONS
═══════════════════════════════════════
{patches_text}

VERIFICATION RULES:
1. For each required fix, find the patched function for that test.
2. Check if the OLD code/pattern mentioned in the fix is GONE from the patch.
3. Check if the NEW code/pattern mentioned in the fix is PRESENT in the patch.
4. If the fix says "Replace X with Y", then X must NOT appear and Y MUST appear.
5. A fix is NOT applied if the patch uses a different approach to solve the same problem.
6. Set all_fixes_applied to true ONLY if EVERY fix passes checks 2, 3, and 4."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": verify_prompt}],
        "max_tokens": 1000,
        "temperature": 0.1,
    }
    params.update(litellm_params or {})
    params["response_format"] = FixVerification

    response = await litellm.acompletion(**params)
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            result_dict = json.loads(content)
            verification = FixVerification(**result_dict)
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to parse test fix verification, assuming fixes not applied")
            verification = FixVerification(
                all_fixes_applied=False,
                applied_fixes=[],
                missing_fixes=["parse_error"],
                explanation="Failed to parse verification response",
            )
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

    new_code = new_solution.code

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
    input_tokens, output_tokens = extract_token_usage(response)

    content = response.choices[0].message.content
    if isinstance(content, str):
        try:
            result_dict = json.loads(content)
            verification = FixVerification(**result_dict)
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to parse logic fix verification, assuming fixes not applied")
            verification = FixVerification(
                all_fixes_applied=False,
                applied_fixes=[],
                missing_fixes=["parse_error"],
                explanation="Failed to parse verification response",
            )
    else:
        verification = content

    return verification, input_tokens, output_tokens


async def diagnose_and_plan_environment_fix(
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
) -> tuple[bool | str, list[str], list[str], list[str], int, int, ErrorDiagnosis]:
    """Diagnose error and plan environment fix (don't execute yet).

    Returns:
        Tuple of (primary_error_type, updated_detected_packages, updated_detected_system_packages,
                  updated_additional_commands, input_tokens, output_tokens, diagnosis)

        where primary_error_type is either:
        - "test_error" (str): Test code has bugs, must fix tests first
        - False (bool): Environment and/or logic errors, handle together
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

    # Important: If any test errors exist, only keep test_error failures
    # Discard logic/environment failures - they're unreliable when tests are broken
    test_error_failures = [f for f in diagnosis.failures if f.error_type == "test_error"]

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
        # Test code has bugs - must regenerate tests first
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
