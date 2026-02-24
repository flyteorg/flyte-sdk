import logging
from pathlib import Path
from typing import Optional

import flyte
import pandas as pd
import pandera.pandas as pa
from flyte.io import File

from flyteplugins.codegen.data.schema import (
    apply_user_constraints,
    infer_conservative_schema,
    schema_to_script,
)

logger = logging.getLogger(__name__)


def is_dataframe(obj) -> bool:
    """Check if object is a pandas DataFrame.

    Args:
        obj: Object to check

    Returns:
        True if object is a DataFrame
    """
    try:
        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


async def extract_dataframe_context(
    df, name: str, max_sample_rows: int = 5, schema: Optional[pa.DataFrameSchema] = None
) -> str:
    """Extract comprehensive context from DataFrame.

    Args:
        df: pandas DataFrame
        name: Name of the data input
        max_sample_rows: Number of sample rows to include
        schema: Optional Pandera schema to include in context

    Returns:
        Formatted string with all extracted context
    """
    context_parts = []

    # 1. Structural Context
    context_parts.append(f"## Data: {name}")
    context_parts.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Include Pandera schema if provided (use Pandera's built-in formatter)
    if schema:
        context_parts.append(f"\nPandera Schema for {name} (use for validation):")
        context_parts.append("```python")
        context_parts.append(schema_to_script(schema))
        context_parts.append("```")

    # 2. Statistical Context
    context_parts.append("\nStatistical Summary:")

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
                context_parts.append(f"    {col}: {unique_count} unique values. Top 5: {{{top_str}}}")
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
    context_parts.append("\nData Patterns:")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        context_parts.append(f"  - {dup_count:,} duplicate rows ({dup_count / len(df) * 100:.1f}%)")

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


async def extract_file_context(file: File, name: str, max_sample_rows: int = 5) -> str:
    """Extract context from non-tabular files (text, binary, unknown formats).

    This is a fallback for files that can't be loaded as DataFrames.
    Structured files (CSV, Parquet, JSON, Excel) are handled by extract_data_context()
    with Pandera schema inference.

    Args:
        file: File to extract context from
        name: Name of the data input
        max_sample_rows: Number of sample rows to include

    Returns:
        Formatted string with all extracted context
    """
    local_path = await file.download()
    file_ext = Path(local_path).suffix.lower()

    # Try to read as text file
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[:max_sample_rows]

        context_parts = [
            f"## Data: {name}",
            f"Type: Text file ({file_ext})",
            f"Lines: {len(lines)}",
            f"\nFirst {max_sample_rows} lines:",
            "".join(lines),
        ]
        return "\n".join(context_parts)

    except Exception:
        # Binary or unreadable file
        file_size = Path(local_path).stat().st_size
        context_parts = [
            f"## Data: {name}",
            f"Type: Binary/Unknown ({file_ext})",
            f"Size: {file_size:,} bytes",
            "\n(Unable to extract text preview)",
        ]
        return "\n".join(context_parts)


@flyte.trace
async def extract_data_context(
    data: dict[str, pd.DataFrame | File],
    max_sample_rows: int = 5,
    constraints: Optional[list[str]] = None,
    model: Optional[str] = None,
    litellm_params: Optional[dict] = None,
) -> tuple[str, dict[str, str], int, int]:
    """Extract comprehensive context from data inputs with Pandera schema inference.

    Extracts:
    1. Structural context (schema, types, shape)
    2. Statistical context (distributions, ranges)
    3. Behavioral context (patterns, invariants)
    4. Operational context (scale, nulls)
    5. Representative samples
    6. Pandera schemas (inference + user constraints), returned as Python code strings

    Args:
        data: Dict of data inputs (File or DataFrame)
        max_sample_rows: Number of sample rows to include
        constraints: Optional list of user constraints to apply to schemas
        model: LLM model for constraint parsing (required if constraints provided)
        litellm_params: Optional LiteLLM parameters

    Returns:
        Tuple of (context_string, schemas_as_code_dict, total_input_tokens, total_output_tokens)
    """
    context_parts = []
    schemas: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for name, value in data.items():
        df = None

        if isinstance(value, File):
            # Load file as DataFrame for schema inference
            local_path = await value.download()
            file_ext = Path(local_path).suffix.lower()

            try:
                if file_ext in [".csv", ".tsv"]:
                    delimiter = "\t" if file_ext == ".tsv" else ","
                    df = pd.read_csv(local_path, delimiter=delimiter, nrows=10000)
                elif file_ext in [".parquet", ".pq"]:
                    df = pd.read_parquet(local_path)
                    if len(df) > 10000:
                        df = df.sample(n=10000)
                elif file_ext == ".json":
                    try:
                        df = pd.read_json(local_path, lines=True, nrows=10000)
                    except:
                        df = pd.read_json(local_path)
                elif file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(local_path, nrows=10000)
                else:
                    # Non-tabular file (e.g., .log, .txt) — extract text context
                    context = await extract_file_context(value, name, max_sample_rows)
                    context_parts.append(context)
                    continue
            except Exception as e:
                logger.warning(f"Failed to load {name} as DataFrame for schema inference: {e}")
                # Fall back to non-schema extraction
                context = await extract_file_context(value, name, max_sample_rows)
                context_parts.append(context)
                continue

        elif is_dataframe(value):
            df = value
        else:
            context_parts.append(f"## Data: {name}\nType: {type(value)}\n(Unsupported type)")
            continue

        if df is not None:
            # Infer Pandera schema
            schema = infer_conservative_schema(df)

            # Apply user constraints if provided
            if constraints and model:
                schema, in_tok, out_tok = await apply_user_constraints(schema, constraints, name, model, litellm_params)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

            # Convert to code string for serialization
            schemas[name] = schema_to_script(schema)

            # Extract context with schema
            context = await extract_dataframe_context(df, name, max_sample_rows, schema)
            context_parts.append(context)

    context_str = "\n\n" + "=" * 80 + "\n\n".join(context_parts)
    return context_str, schemas, total_input_tokens, total_output_tokens
