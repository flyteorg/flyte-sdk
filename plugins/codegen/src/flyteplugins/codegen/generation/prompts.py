"""Prompt building and constants for LLM code generation."""

from typing import Optional

from flyte.io import Dir, File

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
4. system_packages: List of system packages needed
   (e.g., ["gcc", "build-essential", "curl"]). Leave empty if none needed.

EXECUTION ENVIRONMENT:
- /var/inputs and /var/outputs directories are PRE-CREATED by the runtime. NEVER delete, recreate, or modify them.
  NEVER use shutil.rmtree, os.rmdir, os.remove on /var/inputs or /var/outputs.
  NEVER call os.makedirs('/var/outputs') or os.makedirs('/var/inputs') — they already exist.
- /var/inputs is READ-ONLY. Never write to /var/inputs.
- Materialize each declared output under /var/outputs/.
  For scalar outputs: open('/var/outputs/<name>', 'w').write(str(value))
  For File outputs: write the file directly to /var/outputs/<name>
  For Dir outputs: create the directory directly at /var/outputs/<name>
- Always use the literal path '/var/outputs' — never make it configurable or store it in a variable.
- Output files MUST be written before the script exits. Do NOT just print() values — you MUST write them to files.

Ensure all code is complete, executable, and follows best practices for the chosen language."""


def build_enhanced_prompt(
    prompt: str,
    language: str,
    schema: Optional[str],
    constraints: Optional[list[str]],
    data_context: Optional[str],
    inputs: Optional[dict[str, type]],
    outputs: Optional[dict[str, type]],
) -> str:
    """Build enhanced prompt with language, schema, constraints, data context, inputs, and outputs.

    Args:
        prompt: User's prompt
        language: Programming language
        schema: Optional schema definition
        constraints: Optional list of constraints
        data_context: Optional extracted data context (stats, patterns, schemas)
        inputs: Optional input types
        outputs: Optional output types

    Returns:
        Enhanced prompt string
    """
    enhanced_prompt = f"Language: {language}\n\n{prompt}"

    if schema:
        enhanced_prompt += f"\n\nSchema:\n``\n{schema}\n``"

    def _is_path_input_type(param_type: type) -> bool:
        return param_type in (File, Dir) or any(name in str(param_type) for name in ("File", "Dir"))

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
            # Clarify that File/Dir inputs are received as string paths
            if _is_path_input_type(param_type):
                args_list.append(f"--{name} (str): path to {type_name.lower()}")
            else:
                args_list.append(f"--{name} ({type_name})")
        args_spec = ", ".join(args_list)
        script_constraint += f"Accept these command line arguments: {args_spec}. "

        # Add explicit instruction for File/Dir handling
        has_path_inputs = any(_is_path_input_type(t) for t in inputs.values())
        if has_path_inputs:
            script_constraint += (
                "File and Dir arguments are string paths - use them directly with open(), pathlib, or other file operations."
            )
    elif data_context:
        script_constraint += "Accept appropriate command line arguments to process the data samples."
    else:
        script_constraint += "Include appropriate command line arguments if needed."

    all_constraints = [script_constraint]

    # Add output requirement based on declared outputs
    if outputs:
        output_parts = []
        for name, output_type in outputs.items():
            type_name = output_type.__name__ if hasattr(output_type, "__name__") else str(output_type)
            if output_type is Dir or "Dir" in type_name:
                output_parts.append(f"- {name}: create the output directory directly at /var/outputs/{name}")
            elif _is_path_input_type(output_type):
                output_parts.append(f"- {name}: write the output file directly to /var/outputs/{name}")
            else:
                output_parts.append(f"- {name} ({type_name}): write the value to /var/outputs/{name}")
        output_list = "\n".join(output_parts)
        output_constraint = f"""OUTPUT REQUIREMENTS — you MUST materialize each output under /var/outputs/:
{output_list}
Use this exact pattern for scalar outputs:
  with open('/var/outputs/<name>', 'w') as f:
      f.write(str(value))
For File outputs, write the file directly to /var/outputs/<name>.
For Dir outputs, create the directory directly at /var/outputs/<name>.
/var/outputs/ already exists. NEVER delete, recreate, or modify the directory itself. Only write files into it.
Outputs MUST be written before the script exits — do NOT just print() values."""
        all_constraints.append(output_constraint)

    if constraints:
        all_constraints.extend(constraints)

    enhanced_prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in all_constraints)

    if data_context:
        enhanced_prompt += f"\n\nData context:\n``\n{data_context}\n``"

    return enhanced_prompt
