import datetime

from flyte.storage._config import ABFS, GCS, S3, Storage


class TestStorage:
    def test_get_fsspec_kwargs_base(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs()
        assert result == {}
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_base_with_anonymous(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs(anonymous=True)
        assert result == {}
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_base_with_kwargs(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs(test_param="value")
        assert result == {}
        assert "anonymous" not in result


class TestS3Config:
    def test_get_fsspec_kwargs_default(self):
        s3 = S3()
        result = s3.get_fsspec_kwargs()

        assert "config" not in result
        assert "client_options" in result
        assert result["client_options"]["timeout"] == "99999s"
        assert result["client_options"]["allow_http"] is True
        assert "retry_config" in result
        assert result["retry_config"]["max_retries"] == 3
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_credentials(self):
        s3 = S3(access_key_id="test-key", secret_access_key="test-secret", endpoint="http://test-endpoint")
        result = s3.get_fsspec_kwargs()

        assert "config" in result
        assert result["config"]["access_key_id"] == "test-key"
        assert result["config"]["secret_access_key"] == "test-secret"
        assert result["config"]["endpoint"] == "http://test-endpoint"
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_anonymous(self):
        s3 = S3(access_key_id="test-key", secret_access_key="test-secret")
        result = s3.get_fsspec_kwargs(anonymous=True)

        assert "config" in result
        # The skip_signature key should exist in the config dictionary
        assert result["config"].get("skip_signature") is True, result["config"]
        assert result["config"]["access_key_id"] == "test-key"
        assert result["config"]["secret_access_key"] == "test-secret"
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_override_credentials(self):
        s3 = S3(access_key_id="default-key", secret_access_key="default-secret", endpoint="default-endpoint")
        result = s3.get_fsspec_kwargs(
            access_key_id="override-key", secret_access_key="override-secret", endpoint_url="override-endpoint"
        )

        assert "config" in result
        assert result["config"]["access_key_id"] == "override-key"
        assert result["config"]["secret_access_key"] == "override-secret"
        assert result["config"]["endpoint"] == "override-endpoint"
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_retries_backoff_override(self):
        custom_backoff = datetime.timedelta(seconds=10)
        s3 = S3(retries=3, backoff=datetime.timedelta(seconds=5))
        result = s3.get_fsspec_kwargs(retries=5, backoff=custom_backoff)

        assert result["retry_config"]["max_retries"] == 5
        assert result["retry_config"]["backoff"]["init_backoff"] == custom_backoff
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_addressing_style_virtual(self):
        s3 = S3(addressing_style="virtual")
        result = s3.get_fsspec_kwargs()

        assert "config" in result
        assert result["config"]["virtual_hosted_style_request"] is True
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_addressing_style_path(self):
        s3 = S3(addressing_style="path")
        result = s3.get_fsspec_kwargs()

        assert "config" in result
        assert result["config"]["virtual_hosted_style_request"] is False
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_no_addressing_style(self):
        s3 = S3()
        result = s3.get_fsspec_kwargs()

        assert "config" not in result or "virtual_hosted_style_request" not in result.get("config", {})


class TestGCSConfig:
    def test_get_fsspec_kwargs_default(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs()
        assert result == {}
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_anonymous(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs(anonymous=True)
        assert result == {}
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_custom_params(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs(token="test-token", project="test-project")
        assert result == {"token": "test-token", "project": "test-project"}
        assert "anonymous" not in result


class TestABFSConfig:
    def test_get_fsspec_kwargs_default(self):
        abfs = ABFS()
        result = abfs.get_fsspec_kwargs()

        assert "config" not in result
        assert "client_options" in result
        assert result["client_options"]["timeout"] == "99999s"
        assert result["client_options"]["allow_http"] is True
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_credentials(self):
        abfs = ABFS(
            account_name="test-account",
            account_key="test-key",
            tenant_id="test-tenant",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )
        result = abfs.get_fsspec_kwargs()

        assert "config" in result
        assert result["config"]["account_name"] == "test-account"
        assert result["config"]["account_key"] == "test-key"
        assert result["config"]["tenant_id"] == "test-tenant"
        assert result["config"]["client_id"] == "test-client-id"
        assert result["config"]["client_secret"] == "test-client-secret"
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_anonymous(self):
        abfs = ABFS(account_name="test-account", account_key="test-key")
        result = abfs.get_fsspec_kwargs(anonymous=True)

        assert "config" in result
        # The skip_signature key should exist in the config dictionary
        assert result["config"].get("skip_signature") is True
        assert result["config"]["account_name"] == "test-account"
        assert result["config"]["account_key"] == "test-key"
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_override_credentials(self):
        abfs = ABFS(account_name="default-account", account_key="default-key")
        result = abfs.get_fsspec_kwargs(account_name="override-account", account_key="override-key")

        assert "config" in result
        assert result["config"]["account_name"] == "override-account"
        assert result["config"]["account_key"] == "override-key"
        assert "anonymous" not in result
