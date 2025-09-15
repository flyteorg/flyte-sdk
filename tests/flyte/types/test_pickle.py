from unittest.mock import AsyncMock, patch

import cloudpickle
import pytest
from flyteidl.core import literals_pb2, types_pb2

from flyte.types._pickle import DEFAULT_PICKLE_BYTES_LIMIT, FlytePickle, FlytePickleTransformer


class TestFlytePickleSizeOptimization:
    """Test the size-based optimization for pickle handling."""

    @pytest.fixture
    def transformer(self):
        return FlytePickleTransformer()

    @pytest.fixture
    def small_object(self):
        """Create a small object that should be stored inline."""
        return {"small": "data", "number": 42}

    @pytest.fixture
    def large_object(self):
        """Create a large object that should be stored as blob."""
        # Create an object larger than DEFAULT_PICKLE_BYTES_LIMIT
        large_data = "x" * (DEFAULT_PICKLE_BYTES_LIMIT + 1000)
        return {"large": large_data, "data": list(range(1000))}

    @pytest.mark.asyncio
    async def test_small_object_stored_as_binary(self, transformer, small_object):
        """Test that small objects are stored as binary inline."""
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        literal = await transformer.to_literal(small_object, type(small_object), expected_lt)

        # Should be stored as binary, not blob
        assert literal.scalar.binary is not None
        assert not literal.scalar.HasField("blob")  # No blob field for small objects

        # Verify the binary data can be unpickled
        unpickled = cloudpickle.loads(literal.scalar.binary.value)
        assert unpickled == small_object

    @pytest.mark.asyncio
    async def test_large_object_stored_as_blob(self, transformer, large_object):
        """Test that large objects are stored as blob files."""
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        # Mock FlytePickle.to_pickle to avoid actual file operations
        with patch.object(FlytePickle, "to_pickle", new_callable=AsyncMock) as mock_to_pickle:
            mock_to_pickle.return_value = "s3://test-bucket/test-path"

            # Mock sys.getsizeof to return a large size
            with patch("sys.getsizeof", return_value=DEFAULT_PICKLE_BYTES_LIMIT + 1000):
                literal = await transformer.to_literal(large_object, type(large_object), expected_lt)

        # Should be stored as blob, not binary
        assert literal.scalar.blob is not None
        assert literal.scalar.blob.uri == "s3://test-bucket/test-path"
        # Binary should not be set for large objects
        assert not literal.scalar.HasField("binary")

        # Verify to_pickle was called with the large object
        mock_to_pickle.assert_called_once_with(large_object)

    @pytest.mark.asyncio
    async def test_size_threshold_exactly_at_limit(self, transformer):
        """Test behavior when object size is exactly at the limit."""
        test_object = {"test": "data"}
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        # Mock sys.getsizeof to return exactly the limit
        with patch("sys.getsizeof", return_value=DEFAULT_PICKLE_BYTES_LIMIT):
            literal = await transformer.to_literal(test_object, type(test_object), expected_lt)

        # Should be stored as binary (not greater than limit)
        assert literal.scalar.binary is not None
        assert not literal.scalar.HasField("blob")  # No blob field for small objects

    @pytest.mark.asyncio
    async def test_to_python_value_from_binary(self, transformer):
        """Test converting binary literal back to Python value."""
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        binary_data = cloudpickle.dumps(test_data)

        literal = literals_pb2.Literal(scalar=literals_pb2.Scalar(binary=literals_pb2.Binary(value=binary_data)))

        result = await transformer.to_python_value(literal, dict)
        assert result == test_data

    @pytest.mark.asyncio
    async def test_to_python_value_from_blob(self, transformer):
        """Test converting blob literal back to Python value."""
        test_data = {"test": "large_data"}
        uri = "s3://test-bucket/test-file"

        literal = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                blob=literals_pb2.Blob(
                    uri=uri,
                    metadata=literals_pb2.BlobMetadata(
                        type=types_pb2.BlobType(
                            format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                            dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
                        )
                    ),
                )
            )
        )

        # Mock FlytePickle.from_pickle
        with patch.object(FlytePickle, "from_pickle", new_callable=AsyncMock) as mock_from_pickle:
            mock_from_pickle.return_value = test_data

            result = await transformer.to_python_value(literal, dict)

        assert result == test_data
        mock_from_pickle.assert_called_once_with(uri)

    @pytest.mark.asyncio
    async def test_to_python_value_invalid_literal(self, transformer):
        """Test error handling for invalid literal formats."""
        # Literal with neither blob URI nor binary data
        literal = literals_pb2.Literal(scalar=literals_pb2.Scalar())

        with pytest.raises(ValueError, match=r"Cannot convert[\s\S]*to"):
            await transformer.to_python_value(literal, dict)

    @pytest.mark.asyncio
    async def test_none_value_raises_assertion_error(self, transformer):
        """Test that None values raise AssertionError."""
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        with pytest.raises(AssertionError, match="Cannot pickle None Value"):
            await transformer.to_literal(None, type(None), expected_lt)

    def test_default_pickle_bytes_limit_constant(self):
        """Test that the default limit constant is set correctly."""
        assert DEFAULT_PICKLE_BYTES_LIMIT == 2**10 * 10  # 10KB
        assert DEFAULT_PICKLE_BYTES_LIMIT == 10240

    @pytest.mark.asyncio
    async def test_sys_getsizeof_called_correctly(self, transformer):
        """Test that sys.getsizeof is called with the correct object."""
        test_object = {"test": "data"}
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        with patch("sys.getsizeof") as mock_getsizeof:
            mock_getsizeof.return_value = 100  # Small size

            await transformer.to_literal(test_object, type(test_object), expected_lt)

            # Verify sys.getsizeof was called with our test object
            mock_getsizeof.assert_called_once_with(test_object)

    @pytest.mark.asyncio
    async def test_roundtrip_small_object(self, transformer, small_object):
        """Test complete roundtrip for small objects (binary storage)."""
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        # Mock sys.getsizeof to ensure small object storage
        with patch("sys.getsizeof", return_value=DEFAULT_PICKLE_BYTES_LIMIT - 100):
            # Convert to literal
            literal = await transformer.to_literal(small_object, type(small_object), expected_lt)

            # Convert back to Python value
            result = await transformer.to_python_value(literal, type(small_object))

        assert result == small_object

    @pytest.mark.asyncio
    async def test_roundtrip_large_object(self, transformer, large_object):
        """Test complete roundtrip for large objects (blob storage)."""
        expected_lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
                dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
            )
        )

        # Mock both to_pickle and from_pickle for blob storage
        with (
            patch.object(FlytePickle, "to_pickle", new_callable=AsyncMock) as mock_to_pickle,
            patch.object(FlytePickle, "from_pickle", new_callable=AsyncMock) as mock_from_pickle,
            patch("sys.getsizeof", return_value=DEFAULT_PICKLE_BYTES_LIMIT + 1000),
        ):
            mock_to_pickle.return_value = "s3://test-bucket/test-path"
            mock_from_pickle.return_value = large_object

            # Convert to literal
            literal = await transformer.to_literal(large_object, type(large_object), expected_lt)

            # Convert back to Python value
            result = await transformer.to_python_value(literal, type(large_object))

        assert result == large_object
        mock_to_pickle.assert_called_once_with(large_object)
        mock_from_pickle.assert_called_once_with("s3://test-bucket/test-path")
