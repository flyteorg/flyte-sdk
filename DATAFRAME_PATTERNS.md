# DataFrame Pydantic Patterns

The DataFrame class has been successfully converted from a dataclass to a Pydantic model for consistency with File and Dir. Here are the recommended usage patterns:

## ✅ **Recommended: Alternative Constructors**

### **DataFrame.create() - General Purpose**
```python
from flyte.io import DataFrame
import pandas as pd

# Create with all parameters
df_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
df = DataFrame.create(
    val=df_data,
    uri="s3://bucket/file.parquet", 
    file_format="parquet"
)
```

### **DataFrame.from_val() - When you have data**
```python
# Create from dataframe value
df_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
df = DataFrame.from_val(
    val=df_data,
    uri="s3://bucket/file.parquet"
)
```

### **DataFrame.from_uri() - When you have a URI**
```python
# Create from URI only
df = DataFrame.from_uri(
    uri="s3://bucket/file.parquet",
    file_format="parquet"
)
```

## ⚠️ **Legacy: Still Supported for Backward Compatibility**

```python
# These still work but are not recommended for new code
df = DataFrame(val=df_data, uri="s3://bucket/file.parquet")
df = DataFrame(dataframe=df_data, uri="/path")  # Legacy parameter
```

## **Key Benefits of New Patterns**

1. **Proper Pydantic**: Uses private attributes (`PrivateAttr`) for internal state
2. **Clean Serialization**: Only public fields (`uri`, `file_format`) are serialized
3. **Type Safety**: Better integration with type checkers and IDEs
4. **Consistent**: Matches patterns used by File and Dir classes
5. **Future-Proof**: Built on modern Pydantic v2 best practices

## **Private Attributes**

The following attributes are now properly private and excluded from serialization:
- `_val`: The actual dataframe value
- `_metadata`: Structured dataset metadata 
- `_literal_sd`: Literal structured dataset
- `_dataframe_type`: Dataframe type for opening
- `_already_uploaded`: Upload status flag

## **Migration Guide**

- **Existing code**: No changes needed - all existing patterns still work
- **New code**: Use alternative constructors (`DataFrame.create()`, `DataFrame.from_val()`, etc.)
- **Libraries**: Can gradually migrate to new patterns when convenient