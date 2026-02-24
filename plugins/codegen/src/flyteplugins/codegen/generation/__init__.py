"""LLM-based code generation functionality."""

from flyteplugins.codegen.generation.llm import (
    detect_and_track_packages,
    diagnose_and_plan_environment_fix,
    diagnose_error,
    extract_error_messages_from_pytest,
    fix_failing_tests,
    generate_code,
    generate_plan,
    generate_tests,
    verify_logic_fixes_applied,
    verify_test_fixes_applied,
)
from flyteplugins.codegen.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    FILE_EXTENSIONS,
    PACKAGE_MANAGER_MAP,
    STRUCTURED_OUTPUT_REQUIREMENTS,
    TEST_FRAMEWORKS,
    build_enhanced_prompt,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FILE_EXTENSIONS",
    "PACKAGE_MANAGER_MAP",
    "STRUCTURED_OUTPUT_REQUIREMENTS",
    "TEST_FRAMEWORKS",
    "build_enhanced_prompt",
    "detect_and_track_packages",
    "diagnose_and_plan_environment_fix",
    "diagnose_error",
    "extract_error_messages_from_pytest",
    "fix_failing_tests",
    "generate_code",
    "generate_plan",
    "generate_tests",
    "verify_logic_fixes_applied",
    "verify_test_fixes_applied",
]
