from unittest.mock import MagicMock, patch

import pytest
from flyteidl2.core import literals_pb2, types_pb2

pd = pytest.importorskip("pandas")
snowflake_connector = pytest.importorskip("snowflake.connector")


SNOWFLAKE_URI_WRITE = "snowflake://testuser/testaccount/testwarehouse/testdb/testschema/testtable"
SNOWFLAKE_URI_READ = "snowflake://testuser/testaccount/testwarehouse/testdb/testschema/query-abc-123"


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})


class TestGetPrivateKey:
    def test_converts_pem_to_der(self):
        """Test that _get_private_key calls cryptography correctly."""
        mock_private_key = MagicMock()
        mock_der_bytes = b"der-encoded-key"
        mock_private_key.private_bytes.return_value = mock_der_bytes

        with (
            patch(
                "cryptography.hazmat.primitives.serialization.load_pem_private_key",
                return_value=mock_private_key,
            ) as mock_load,
            patch("cryptography.hazmat.backends.default_backend"),
        ):
            from flyteplugins.snowflake.dataframe import _get_private_key

            result = _get_private_key("-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----")

            assert result == mock_der_bytes
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args
            assert call_kwargs[1]["password"] is None

    def test_with_passphrase(self):
        """Test that passphrase is passed through when provided."""
        mock_private_key = MagicMock()
        mock_private_key.private_bytes.return_value = b"der-encoded-key"

        with (
            patch(
                "cryptography.hazmat.primitives.serialization.load_pem_private_key",
                return_value=mock_private_key,
            ) as mock_load,
            patch("cryptography.hazmat.backends.default_backend"),
        ):
            from flyteplugins.snowflake.dataframe import _get_private_key

            _get_private_key("fake-key-content", private_key_passphrase="my-passphrase")

            call_kwargs = mock_load.call_args
            assert call_kwargs[1]["password"] == b"my-passphrase"

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped from key content."""
        mock_private_key = MagicMock()
        mock_private_key.private_bytes.return_value = b"der-encoded-key"

        with (
            patch(
                "cryptography.hazmat.primitives.serialization.load_pem_private_key",
                return_value=mock_private_key,
            ) as mock_load,
            patch("cryptography.hazmat.backends.default_backend"),
        ):
            from flyteplugins.snowflake.dataframe import _get_private_key

            _get_private_key("  \n  fake-key-content  \n  ")

            call_args = mock_load.call_args[0]
            assert call_args[0] == b"fake-key-content"


class TestGetConnection:
    def test_connection_with_private_key(self, monkeypatch):
        """Test that a connection is created with private key when env var is set."""
        monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY", "fake-key-content")
        monkeypatch.delenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", raising=False)

        import flyteplugins.snowflake.dataframe as sf_module

        with (
            patch.object(sf_module, "_get_private_key", return_value=b"der-key") as mock_gpk,
            patch.object(snowflake_connector, "connect") as mock_connect,
        ):
            mock_connect.return_value = MagicMock()
            sf_module._get_connection("user", "account", "db", "schema", "warehouse")

            mock_gpk.assert_called_once_with("fake-key-content", None)
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["private_key"] == b"der-key"
            assert call_kwargs["user"] == "user"
            assert call_kwargs["account"] == "account"

    def test_connection_with_private_key_and_passphrase(self, monkeypatch):
        """Test connection with both private key and passphrase env vars."""
        monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY", "fake-key-content")
        monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "my-pass")

        import flyteplugins.snowflake.dataframe as sf_module

        with (
            patch.object(sf_module, "_get_private_key", return_value=b"der-key") as mock_gpk,
            patch.object(snowflake_connector, "connect") as mock_connect,
        ):
            mock_connect.return_value = MagicMock()
            sf_module._get_connection("user", "account", "db", "schema", "warehouse")

            mock_gpk.assert_called_once_with("fake-key-content", "my-pass")

    def test_connection_without_private_key(self, monkeypatch):
        """Test that a connection is created without private key when env var is not set."""
        monkeypatch.delenv("SNOWFLAKE_PRIVATE_KEY", raising=False)

        import flyteplugins.snowflake.dataframe as sf_module

        with patch.object(snowflake_connector, "connect") as mock_connect:
            mock_connect.return_value = MagicMock()
            sf_module._get_connection("user", "account", "db", "schema", "warehouse")

            call_kwargs = mock_connect.call_args[1]
            assert "private_key" not in call_kwargs
            assert call_kwargs["user"] == "user"
            assert call_kwargs["account"] == "account"
            assert call_kwargs["database"] == "db"
            assert call_kwargs["schema"] == "schema"
            assert call_kwargs["warehouse"] == "warehouse"


class TestWriteToSf:
    def test_writes_dataframe(self, sample_dataframe):
        """Test that _write_to_sf parses the URI and writes the DataFrame."""
        from flyte.io._dataframe.dataframe import DataFrame

        import flyteplugins.snowflake.dataframe as sf_module

        mock_conn = MagicMock()

        with (
            patch.object(sf_module, "_get_connection", return_value=mock_conn) as mock_get_conn,
            patch("snowflake.connector.pandas_tools.write_pandas") as mock_wp,
        ):
            df_wrapper = DataFrame.from_df(val=sample_dataframe, uri=SNOWFLAKE_URI_WRITE)
            sf_module._write_to_sf(df_wrapper)

            mock_get_conn.assert_called_once_with("testuser", "testaccount", "testdb", "testschema", "testwarehouse")
            mock_wp.assert_called_once_with(mock_conn, sample_dataframe, "testtable")

    def test_raises_on_missing_uri(self):
        """Test that _write_to_sf raises when URI is not set."""
        from flyte.io._dataframe.dataframe import DataFrame

        from flyteplugins.snowflake.dataframe import _write_to_sf

        df_wrapper = DataFrame.from_df(val=pd.DataFrame())
        with pytest.raises(ValueError, match=r"dataframe\.uri cannot be None"):
            _write_to_sf(df_wrapper)


class TestReadFromSf:
    def test_reads_dataframe(self, sample_dataframe):
        """Test that _read_from_sf parses the URI and fetches the DataFrame."""
        import flyteplugins.snowflake.dataframe as sf_module

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all.return_value = sample_dataframe
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(sf_module, "_get_connection", return_value=mock_conn) as mock_get_conn:
            flyte_value = literals_pb2.StructuredDataset(uri=SNOWFLAKE_URI_READ)
            metadata = literals_pb2.StructuredDatasetMetadata()

            result = sf_module._read_from_sf(flyte_value, metadata)

            pd.testing.assert_frame_equal(result, sample_dataframe)
            mock_get_conn.assert_called_once_with("testuser", "testaccount", "testdb", "testschema", "testwarehouse")
            mock_cursor.get_results_from_sfqid.assert_called_once_with("query-abc-123")

    def test_raises_on_empty_uri(self):
        """Test that _read_from_sf raises when URI is empty."""
        from flyteplugins.snowflake.dataframe import _read_from_sf

        flyte_value = literals_pb2.StructuredDataset(uri="")
        metadata = literals_pb2.StructuredDatasetMetadata()

        with pytest.raises(ValueError, match=r"flyte_value\.uri cannot be empty"):
            _read_from_sf(flyte_value, metadata)


class TestPandasToSnowflakeEncoder:
    @pytest.mark.asyncio
    async def test_encode(self, sample_dataframe):
        """Test that the encoder writes to Snowflake and returns correct literal."""
        from flyte.io._dataframe.dataframe import DataFrame

        import flyteplugins.snowflake.dataframe as sf_module

        encoder = sf_module.PandasToSnowflakeEncodingHandlers()
        assert encoder.protocol == "snowflake"
        assert encoder.python_type == pd.DataFrame

        mock_conn = MagicMock()

        with (
            patch.object(sf_module, "_get_connection", return_value=mock_conn),
            patch("snowflake.connector.pandas_tools.write_pandas"),
        ):
            df_wrapper = DataFrame.from_df(val=sample_dataframe, uri=SNOWFLAKE_URI_WRITE)
            sd_type = types_pb2.StructuredDatasetType()

            result = await encoder.encode(df_wrapper, sd_type)

            assert isinstance(result, literals_pb2.StructuredDataset)
            assert result.uri == SNOWFLAKE_URI_WRITE
            assert result.metadata.structured_dataset_type == sd_type


class TestSnowflakeToPandasDecoder:
    @pytest.mark.asyncio
    async def test_decode(self, sample_dataframe):
        """Test that the decoder reads from Snowflake and returns a pandas DataFrame."""
        import flyteplugins.snowflake.dataframe as sf_module

        decoder = sf_module.SnowflakeToPandasDecodingHandler()
        assert decoder.protocol == "snowflake"
        assert decoder.python_type == pd.DataFrame

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all.return_value = sample_dataframe
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(sf_module, "_get_connection", return_value=mock_conn):
            flyte_value = literals_pb2.StructuredDataset(uri=SNOWFLAKE_URI_READ)
            metadata = literals_pb2.StructuredDatasetMetadata()

            result = await decoder.decode(flyte_value, metadata)

            pd.testing.assert_frame_equal(result, sample_dataframe)
            mock_cursor.get_results_from_sfqid.assert_called_once_with("query-abc-123")
