"""Prompt building and constants for LLM code generation."""

from typing import Optional

# Language-specific file extensions
FILE_EXTENSIONS = {"python": ".py"}

# Package manager mapping
PACKAGE_MANAGER_MAP = {"python": "pip package names (excluding standard library)"}

# Test framework configurations
TEST_FRAMEWORKS = {
    "python": {
        "name": "pytest",
        "packages": ["pytest"],
        "system_packages": [],
        "command": "python -m pytest",
    }
}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a coding assistant that generates high-quality code in {language}."""

# Structured output requirements
STRUCTURED_OUTPUT_REQUIREMENTS = """
IMPORTANT: You must structure your response with:
1. description: Brief explanation of what the code does
2. language: The programming language used
3. code: Complete executable code including all import statements and dependencies at the top
4. system_packages: List of system packages needed (e.g., ["gcc", "build-essential", "curl"]). Leave empty if none needed.

EXECUTION ENVIRONMENT:
- /var/inputs and /var/outputs directories are PRE-CREATED by the runtime. NEVER delete, recreate, or modify them.
  NEVER use shutil.rmtree, os.rmdir, os.remove on /var/inputs or /var/outputs.
  NEVER call os.makedirs('/var/outputs') or os.makedirs('/var/inputs') — they already exist.
- /var/inputs is READ-ONLY. Never write to /var/inputs.
- Write each declared output as a SEPARATE FILE under /var/outputs/: open('/var/outputs/<name>', 'w').write(str(value))
- Always use the literal path '/var/outputs' — never make it configurable or store it in a variable.
- Output files MUST be written before the script exits. Do NOT just print() values — you MUST write them to files.

Ensure all code is complete, executable, and follows best practices for the chosen language."""


def build_enhanced_prompt(
    prompt: str,
    language: str,
    schema: Optional[str],
    constraints: Optional[list[str]],
    data_samples: Optional[str],
    inputs: Optional[dict[str, type]],
    outputs: Optional[dict[str, type]],
) -> str:
    """Build enhanced prompt with language, schema, constraints, data samples, inputs, and outputs.

    Args:
        prompt: User's prompt
        language: Programming language
        schema: Optional schema definition
        constraints: Optional list of constraints
        data_samples: Optional data samples context
        inputs: Optional input types
        outputs: Optional output types

    Returns:
        Enhanced prompt string
    """
    enhanced_prompt = f"Language: {language}\n\n{prompt}"

    if schema:
        enhanced_prompt += f"\n\nSchema:\n```\n{schema}\n```"

    # Always add script requirement first, then user constraints
    script_constraint = (
        "REQUIRED: Your code will be saved as solution.py and imported by tests via "
        "`from solution import ...`. Define ALL functions and classes at MODULE LEVEL "
        "(not inside if __name__ == '__main__'). "
        "Include an if __name__ == '__main__': block that parses command line arguments "
        "using argparse and calls your functions. "
    )

    # Add CLI argument requirement based on declared inputs
    if inputs:
        # Build argument list from declared inputs
        args_list = []
        for name, param_type in inputs.items():
            type_name = param_type.__name__ if hasattr(param_type, "__name__") else str(param_type)
            # Clarify that File inputs are received as string paths
            if "File" in type_name:
                args_list.append(f"--{name} (str): path to {type_name.lower()}")
            else:
                args_list.append(f"--{name} ({type_name})")
        args_spec = ", ".join(args_list)
        script_constraint += f"Accept these command line arguments: {args_spec}. "

        # Add explicit instruction for File handling
        has_file_inputs = any("File" in str(t) for t in inputs.values())
        if has_file_inputs:
            script_constraint += (
                "File arguments are string paths - use them directly with open() or other file operations."
            )
    elif data_samples:
        script_constraint += "Accept appropriate command line arguments to process the data samples."
    else:
        script_constraint += "Include appropriate command line arguments if needed."

    all_constraints = [script_constraint]

    # Add output requirement based on declared outputs
    if outputs:
        output_parts = []
        for name, output_type in outputs.items():
            type_name = output_type.__name__ if hasattr(output_type, "__name__") else str(output_type)
            if "File" in type_name:
                output_parts.append(f"- {name}: write the output file directly to /var/outputs/{name}")
            else:
                output_parts.append(f"- {name} ({type_name}): write the value to /var/outputs/{name}")
        output_list = "\n".join(output_parts)
        output_constraint = f"""OUTPUT REQUIREMENTS — you MUST write each output as a file under /var/outputs/:
{output_list}
Use this exact pattern for each output:
  with open('/var/outputs/<name>', 'w') as f:
      f.write(str(value))
/var/outputs/ already exists. NEVER delete, recreate, or modify the directory itself. Only write files into it.
Outputs MUST be written before the script exits — do NOT just print() values."""
        all_constraints.append(output_constraint)

    if constraints:
        all_constraints.extend(constraints)

    enhanced_prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in all_constraints)

    if data_samples:
        enhanced_prompt += f"\n\nData samples:\n```\n{data_samples}\n```"

    return enhanced_prompt
