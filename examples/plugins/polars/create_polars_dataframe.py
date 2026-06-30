"""Script to create sample Polars DataFrames and LazyFrames for testing.

This script creates:
1. A single parquet file containing a Polars DataFrame
2. A LazyFrame-compatible parquet file

Run this script before using polars_dataframe_inputs.py with `flyte run`.
"""

import polars as pl


def create_dataframe() -> pl.DataFrame:
    """Create a sample Polars DataFrame."""
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "age": [25, 30, 35, 40, 28, 33],
            "category": ["A", "B", "A", "C", "B", "C"],
            "salary": [55000.0, 75000.0, 72000.0, 50000.0, 68000.0, 70000.0],
            "active": [True, False, True, True, True, False],
        }
    )


def create_lazyframe() -> pl.LazyFrame:
    """Create a sample Polars LazyFrame."""
    return pl.LazyFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "age": [25, 30, 35, 40, 28, 33],
            "category": ["A", "B", "A", "C", "B", "C"],
            "salary": [55000.0, 75000.0, 72000.0, 50000.0, 68000.0, 70000.0],
            "active": [True, False, True, True, True, False],
        }
    )


if __name__ == "__main__":
    import pathlib

    parent = pathlib.Path(__file__).parent

    # Create a single parquet file
    df = create_dataframe()
    fp = parent / "dataframe.parquet"
    df.write_parquet(fp)
    print(f"Polars DataFrame saved to {fp}")

    # Create a lazyframe-specific parquet (same format, different name for clarity)
    lf = create_lazyframe()
    lf_fp = parent / "lazyframe.parquet"
    lf.collect().write_parquet(lf_fp)
    print(f"Polars LazyFrame data saved to {lf_fp}")
