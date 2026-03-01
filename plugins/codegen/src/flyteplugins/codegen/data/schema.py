import json
import logging
from typing import Optional

import litellm
import pandas as pd
import pandera.pandas as pa

from flyteplugins.codegen.core.types import _ConstraintParse

logger = logging.getLogger(__name__)


def schema_to_script(schema: pa.DataFrameSchema) -> str:
    """Convert a Pandera schema to a script string, falling back to repr if black is not installed."""
    try:
        return schema.to_script()
    except ImportError:
        return repr(schema)


def infer_conservative_schema(df: pd.DataFrame) -> pa.DataFrameSchema:
    """Infer Pandera schema conservatively - types only, no value constraints.

    Args:
        df: DataFrame to infer schema from

    Returns:
        DataFrameSchema with only dtype and nullability checks
    """
    # Normalize nullable extension dtypes that Pandera doesn't recognize
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
        elif isinstance(dtype, pd.Int8Dtype | pd.Int16Dtype | pd.Int32Dtype | pd.Int64Dtype):
            df[col] = df[col].astype("int64" if not df[col].isna().any() else "float64")
        elif isinstance(dtype, pd.UInt8Dtype | pd.UInt16Dtype | pd.UInt32Dtype | pd.UInt64Dtype):
            df[col] = df[col].astype("uint64" if not df[col].isna().any() else "float64")
        elif isinstance(dtype, pd.Float32Dtype | pd.Float64Dtype):
            df[col] = df[col].astype("float64")
        elif isinstance(dtype, pd.BooleanDtype):
            df[col] = df[col].astype("bool" if not df[col].isna().any() else object)

    # Use Pandera's built-in inference
    base_schema = pa.infer_schema(df)

    # Remove all value-based checks (keep only type checks)
    relaxed_columns = {}

    for col_name, col_schema in base_schema.columns.items():
        # Keep dtype and nullable, remove all checks
        relaxed_columns[col_name] = pa.Column(
            dtype=col_schema.dtype,
            nullable=col_schema.nullable,
            checks=None,  # Remove all inferred checks
        )

    return pa.DataFrameSchema(
        columns=relaxed_columns,
        strict=False,  # Allow additional columns
    )


def extract_token_usage(response) -> tuple[int, int]:
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


async def parse_constraint_with_llm(
    constraint: str,
    data_name: str,
    schema: pa.DataFrameSchema,
    model: str,
    litellm_params: Optional[dict] = None,
) -> tuple[Optional[_ConstraintParse], int, int]:
    """Use LLM to parse natural language constraint into Pandera check.

    Args:
        constraint: Natural language constraint (e.g., "quantity must be positive")
        data_name: Name of the data this constraint applies to
        schema: Inferred schema for column name validation
        model: LLM model to use
        litellm_params: Optional LiteLLM parameters

    Returns:
        Tuple of (ConstraintParse or None, input_tokens, output_tokens)
    """
    column_names = list(schema.columns.keys())

    parse_prompt = f"""Parse this data constraint into a structured check.

Data name: {data_name}
Available columns: {", ".join(column_names)}

Constraint: "{constraint}"

Determine:
1. Which column does this apply to? (must be from available columns)
2. What type of check is this?
   - 'greater_than': column > value (for "positive", "must be at least X")
   - 'less_than': column < value
   - 'between': min <= column <= max
   - 'regex': column matches pattern (for format constraints like "YYYY-MM-DD")
   - 'isin': column value in list (only if specific values listed)
   - 'not_null': column cannot be null
   - 'none': constraint doesn't apply to data validation
3. Parameters needed for the check

Examples:
- "quantity must be positive" →
  column_name: "quantity", check_type: "greater_than",
  parameters: {{"value": 0}}, explanation: "quantity must be greater than 0"
- "price between 0 and 1000" →
  column_name: "price", check_type: "between",
  parameters: {{"min": 0, "max": 1000}},
  explanation: "price must be between 0 and 1000"
- "date in YYYY-MM-DD format" →
  column_name: "date", check_type: "regex",
  parameters: {{"pattern": "\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}}"}},
  explanation: "date must match YYYY-MM-DD format"
- "product must be Widget A or B" →
  column_name: "product", check_type: "isin",
  parameters: {{"values": ["Widget A", "Widget B"]}},
  explanation: "product must be one of the allowed values"

If constraint is unclear or doesn't apply to a specific column, use check_type: 'none'."""

    params = {
        "model": model,
        "messages": [{"role": "user", "content": parse_prompt}],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    params.update(litellm_params or {})
    params["response_format"] = _ConstraintParse

    try:
        response = await litellm.acompletion(**params)
        input_tokens, output_tokens = extract_token_usage(response)

        content = response.choices[0].message.content
        if isinstance(content, str):
            parse_dict = json.loads(content)
            parsed = _ConstraintParse(**parse_dict)
        else:
            parsed = content

        # Validate column exists
        if parsed.check_type != "none" and parsed.column_name not in column_names:
            logger.warning(f"Constraint '{constraint}' references unknown column '{parsed.column_name}'. Skipping.")
            return None, input_tokens, output_tokens

        return parsed, input_tokens, output_tokens

    except Exception as e:
        logger.warning(f"Failed to parse constraint '{constraint}': {e}")
        return None, 0, 0


def apply_parsed_constraint(
    schema: pa.DataFrameSchema,
    parsed: _ConstraintParse,
) -> pa.DataFrameSchema:
    """Apply a parsed constraint to the schema.

    Args:
        schema: DataFrameSchema to update
        parsed: Parsed constraint

    Returns:
        Updated schema
    """
    if parsed.check_type == "none":
        return schema

    col_name = parsed.column_name
    params = parsed.parameters

    # Build Pandera check based on type
    check = None

    if parsed.check_type == "greater_than":
        check = pa.Check.gt(params.value if params.value is not None else 0)

    elif parsed.check_type == "less_than":
        check = pa.Check.lt(params.value if params.value is not None else 0)

    elif parsed.check_type == "between":
        min_val = params.min if params.min is not None else 0
        max_val = params.max if params.max is not None else 100
        check = pa.Check.in_range(min_val, max_val)

    elif parsed.check_type == "regex":
        pattern = params.pattern if params.pattern is not None else ".*"
        check = pa.Check.str_matches(pattern)

    elif parsed.check_type == "isin":
        values = params.values or []
        if values:
            check = pa.Check.isin(values)

    elif parsed.check_type == "not_null":
        # Update nullable flag instead of adding check
        schema = schema.update_column(col_name, nullable=False)
        return schema

    if check:
        # Add check to column
        existing_checks = schema.columns[col_name].checks or []
        if not isinstance(existing_checks, list):
            existing_checks = [existing_checks]

        schema = schema.update_column(col_name, checks=[*existing_checks, check])

        logger.info(f"Applied constraint to '{col_name}': {parsed.explanation}")

    return schema


async def apply_user_constraints(
    schema: pa.DataFrameSchema,
    constraints: list[str],
    data_name: str,
    model: str,
    litellm_params: Optional[dict] = None,
) -> tuple[pa.DataFrameSchema, int, int]:
    """Apply user-specified constraints to schema using LLM parsing.

    Args:
        schema: Base schema (types only)
        constraints: List of natural language constraints
        data_name: Name of the data
        model: LLM model for parsing
        litellm_params: Optional LiteLLM parameters

    Returns:
        Tuple of (enhanced_schema, total_input_tokens, total_output_tokens)
    """
    enhanced_schema = schema
    total_input_tokens = 0
    total_output_tokens = 0

    for constraint in constraints:
        # Use LLM to parse constraint
        parsed, in_tok, out_tok = await parse_constraint_with_llm(constraint, data_name, schema, model, litellm_params)

        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if parsed:
            # Apply to schema
            enhanced_schema = apply_parsed_constraint(enhanced_schema, parsed)

    return enhanced_schema, total_input_tokens, total_output_tokens
