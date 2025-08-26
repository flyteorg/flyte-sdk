"""
Basic DataFrame Usage Examples

This example demonstrates how flyte.DataFrame is a wrapper around raw dataframes.

Key principles:
- Working with raw dataframes (pd.DataFrame) = automatic upload/download
- Working with flyte.DataFrame = I/O only when you explicitly request it
- At task completion: all dataframes are uploaded to remote storage
"""

import pandas as pd

import flyte
from flyte.io import DataFrame

# Create task environment with required dependencies
img = flyte.Image.from_debian_base()
img = img.with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "dataframe_usage",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

# Simple sample data
SAMPLE_DATA = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "SF", "LA"]
}


@env.task
async def create_raw_dataframe() -> pd.DataFrame:
    """
    Task returns RAW dataframe (pd.DataFrame).
    UPLOAD happens at the end of the task, the Flyte Type Engine will do the upload from memory to blob store.
    """
    return pd.DataFrame(SAMPLE_DATA)


@env.task
async def create_wrapper_dataframe() -> DataFrame:
    """
    Task returns WRAPPER dataframe (flyte.DataFrame).
    This creates real remote data first, then returns a wrapper reference.
    As a user, you would rarely do this, this is just demonstrating how to handle existing remote data.
    """
    from flyte._context import internal_ctx
    import flyte.storage as storage
    import tempfile
    import os
    
    # Create a different dataset
    other_data = pd.DataFrame({
        "product": ["Widget A", "Widget B", "Widget C"],
        "price": [10.99, 15.50, 8.25],
        "stock": [100, 50, 75]
    })
    
    # Get a random remote path for upload
    remote_path = internal_ctx().raw_data.get_random_remote_path("products.parquet")
    
    # Save to temp file and upload manually using storage layer
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        other_data.to_parquet(tmp_file.name)
        uploaded_uri = await storage.put(tmp_file.name, remote_path)
        os.unlink(tmp_file.name)
    
    # Now create a wrapper reference to the uploaded data
    return DataFrame.from_existing_remote(uploaded_uri, format="parquet")


@env.task
async def inspect_wrapper_metadata(df: DataFrame) -> dict:
    """
    Working with flyte.DataFrame wrapper - NO I/O happens.
    Just accessing metadata from the wrapper.
    """
    return {
        "uri": df.uri,
        "format": df.format,
    }


@env.task
async def wrapper_to_raw(df: DataFrame) -> pd.DataFrame:
    """
    Input: flyte.DataFrame wrapper -> Output: raw pd.DataFrame
    DOWNLOAD happens: df.open().all() downloads remote data to pandas
    UPLOAD happens: returning pandas triggers automatic upload
    """
    # DOWNLOAD: wrapper -> raw pandas (I/O occurs)
    pandas_df = await df.open(pd.DataFrame).all()
    print(f"Downloaded pandas dataframe:\n{pandas_df}", flush=True)
    
    # Process the raw dataframe
    pandas_df["processed"] = True
    print(f"With processed column:\n{pandas_df}", flush=True)
    
    # UPLOAD: raw pandas is automatically uploaded at task completion
    return pandas_df


@env.task  
async def wrapper_passthrough(df: DataFrame) -> DataFrame:
    """
    Input: flyte.DataFrame wrapper -> Output: flyte.DataFrame wrapper
    If you happen to not perform any operations on the DataFrame, no I/O happens.
    """
    print(f"No I/O - just passing wrapper reference: {df.uri}", flush=True)
    return df


@env.task
async def raw_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw pd.DataFrame -> Output: raw pd.DataFrame  
    DOWNLOAD happens: input pandas is downloaded from remote storage
    UPLOAD happens: output pandas is uploaded at task completion
    """
    # Input df is already downloaded raw pandas
    df["doubled_age"] = df["age"] * 2
    print(f"Pandas dataframe with doubled age:\n{df}", flush=True)
    
    # Output will be uploaded automatically
    return df


@env.task
async def main_workflow() -> dict:
    """
    Demonstrates when I/O happens vs when it doesn't.
    """
    # Pattern 1: Create raw dataframe (UPLOAD occurs)
    raw_df = await create_raw_dataframe()
    print(f"Created raw dataframe:\n{raw_df}", flush=True)

    # Pattern 2: Create wrapper dataframe (NO I/O)  
    wrapped_df_from_existing_data = await create_wrapper_dataframe()
    print(f"Created wrapper dataframe pointing to: {wrapped_df_from_existing_data.}", flush=True)
    
    # Pattern 3: Wrapper metadata access (NO I/O)
    metadata = await inspect_wrapper_metadata(wrapped_df_from_existing_data)
    print(f"DataFrame wrapper metadata: {metadata}", flush=True)
    
    # Pattern 4: Wrapper -> Raw (DOWNLOAD + UPLOAD)
    processed_raw = await wrapper_to_raw(wrapped_df_from_existing_data)
    
    # Pattern 5: Wrapper -> Wrapper (NO I/O)
    same_wrapper = await wrapper_passthrough(wrapped_df_from_existing_data)
    
    # Pattern 6: Raw -> Raw (DOWNLOAD + UPLOAD)  
    doubled_raw = await raw_to_raw(raw_df)

    return {
        "raw_df_type": str(type(raw_df)),
        "wrapper_df_uri": wrapped_df_from_existing_data.uri,
        "metadata": metadata,
        "processed_type": str(type(processed_raw)),
        "passthrough_uri": same_wrapper.uri,
        "doubled_type": str(type(doubled_raw)),
        "message": "All dataframes uploaded at task completion!"
    }


if __name__ == "__main__":
    # Use local execution mode
    run = flyte.with_runcontext(mode="local").run(main_workflow)
    
    print("DataFrame wrapper vs raw I/O patterns demonstrated!")
    print("Key takeaway: flyte.DataFrame = wrapper (I/O on demand)")
    print("             pd.DataFrame = raw (automatic I/O)")
    print("Results:", run.outputs())