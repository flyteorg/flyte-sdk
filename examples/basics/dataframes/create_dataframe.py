import pandas as pd


def create_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "category": ["A", "B", "A", "C"],
            "active": [True, False, True, True],
        }
    )


if __name__ == "__main__":
    import pathlib

    df = create_dataframe()
    fp = pathlib.Path(__file__).parent / "dataframe.parquet"
    df.to_parquet(fp)
    print(f"DataFrame saved to {fp}")

    # Write the dataframe to a directory of Parquet files, partitioned by 'category'
    partitioned_dir = pathlib.Path(__file__).parent / "dataframe_partitioned"
    df.to_parquet(partitioned_dir, partition_cols=["category"])
    print(f"Partitioned DataFrame (by 'category') saved to {partitioned_dir}")
