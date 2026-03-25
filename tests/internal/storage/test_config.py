import datetime

from flyte.storage._config import ABFS, GCS, S3, Storage


class TestStorage:
    def test_get_fsspec_kwargs_base(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs()

        assert "client_options" in result
        assert result["client_options"]["timeout"] == "99999s"
        assert result["client_options"]["allow_http"] is True
        assert "retry_config" in result
        assert result["retry_config"]["max_retries"] == 3
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_base_with_anonymous(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs(anonymous=True)

        assert "client_options" in result
        assert "retry_config" in result
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_base_with_kwargs(self):
        storage = Storage()
        result = storage.get_fsspec_kwargs(test_param="value")

        assert result["test_param"] == "value"
        assert "client_options" in result
        assert "retry_config" in result
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

    def test_get_fsspec_kwargs_caller_overrides_not_clobbered(self):
        s3 = S3(access_key_id="my-key", secret_access_key="my-secret", endpoint="https://s3.us-west-2.amazonaws.com")
        custom_retry = {"max_retries": 10, "retry_timeout": datetime.timedelta(minutes=10)}
        custom_client_options = {"timeout": "30s", "allow_http": False}
        result = s3.get_fsspec_kwargs(
            retry_config=custom_retry,
            client_options=custom_client_options,
        )

        # Caller-provided retry_config and client_options must not be overwritten by defaults
        assert result["retry_config"] is custom_retry
        assert result["retry_config"]["max_retries"] == 10
        assert result["client_options"] is custom_client_options
        assert result["client_options"]["timeout"] == "30s"
        assert result["client_options"]["allow_http"] is False
        # S3-specific config should still be populated
        assert result["config"]["access_key_id"] == "my-key"
        assert result["config"]["secret_access_key"] == "my-secret"
        assert result["config"]["endpoint"] == "https://s3.us-west-2.amazonaws.com"

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

    def test_get_fsspec_kwargs_with_profile_uses_credential_provider(self, monkeypatch):
        s3 = S3(endpoint="http://test-endpoint", addressing_style="virtual")
        monkeypatch.setenv("AWS_PROFILE", "dev-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")

        def _fake_provider(self, aws_profile, aws_config_file, region):
            assert aws_profile == "dev-profile"
            assert aws_config_file == "/tmp/config"
            assert region is None
            return "provider"

        monkeypatch.setattr(S3, "_build_s3_credential_provider_from_config_file", _fake_provider)
        result = s3.get_fsspec_kwargs()

        assert result["credential_provider"] == "provider"
        cfg = result.get("config", {})
        assert "access_key_id" not in cfg
        assert "secret_access_key" not in cfg
        assert cfg.get("endpoint") == "http://test-endpoint"
        assert cfg.get("virtual_hosted_style_request") is True

    def test_get_fsspec_kwargs_profile_not_used_when_static_credentials_present(self, monkeypatch):
        s3 = S3(access_key_id="test-key", secret_access_key="test-secret")
        monkeypatch.setenv("AWS_PROFILE", "dev-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")
        monkeypatch.setattr(
            S3,
            "_build_s3_credential_provider_from_config_file",
            lambda self, aws_profile, aws_config_file, region: (_ for _ in ()).throw(
                AssertionError("provider should not be called")
            ),
        )
        result = s3.get_fsspec_kwargs()

        assert result["config"]["access_key_id"] == "test-key"
        assert result["config"]["secret_access_key"] == "test-secret"
        assert "credential_provider" not in result

    def test_get_fsspec_kwargs_profile_from_env(self, monkeypatch):
        s3 = S3()
        monkeypatch.setenv("AWS_PROFILE", "default-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")

        def _fake_provider(self, aws_profile, aws_config_file, region):
            assert aws_profile == "default-profile"
            assert aws_config_file == "/tmp/config"
            assert region is None
            return "provider"

        monkeypatch.setattr(S3, "_build_s3_credential_provider_from_config_file", _fake_provider)
        result = s3.get_fsspec_kwargs()

        assert result["credential_provider"] == "provider"

    def test_get_fsspec_kwargs_anonymous_does_not_use_profile_provider(self, monkeypatch):
        s3 = S3()
        monkeypatch.setenv("AWS_PROFILE", "dev-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")
        monkeypatch.setattr(
            S3,
            "_build_s3_credential_provider_from_config_file",
            lambda self, aws_profile, aws_config_file, region: (_ for _ in ()).throw(
                AssertionError("provider should not be called for anonymous")
            ),
        )
        result = s3.get_fsspec_kwargs(anonymous=True)

        assert result["config"]["skip_signature"] is True
        assert "credential_provider" not in result

    def test_get_fsspec_kwargs_profile_provider_receives_region(self, monkeypatch):
        s3 = S3(region="us-west-2")
        monkeypatch.setenv("AWS_PROFILE", "dev-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")

        def _fake_provider(self, aws_profile, aws_config_file, region):
            assert aws_profile == "dev-profile"
            assert aws_config_file == "/tmp/config"
            assert region == "us-west-2"
            return "provider"

        monkeypatch.setattr(S3, "_build_s3_credential_provider_from_config_file", _fake_provider)
        result = s3.get_fsspec_kwargs()

        assert result["credential_provider"] == "provider"
        assert result["region"] == "us-west-2"

    def test_get_fsspec_kwargs_profile_provider_failure_falls_back(self, monkeypatch):
        s3 = S3()
        monkeypatch.setenv("AWS_PROFILE", "dev-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/config")
        monkeypatch.setattr(
            S3,
            "_build_s3_credential_provider_from_config_file",
            lambda self, aws_profile, aws_config_file, region: (_ for _ in ()).throw(
                ModuleNotFoundError("boto3 missing")
            ),
        )
        result = s3.get_fsspec_kwargs()

        assert "credential_provider" not in result


class TestGCSConfig:
    def test_get_fsspec_kwargs_default(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs()

        assert "client_options" in result
        assert result["client_options"]["timeout"] == "99999s"
        assert result["client_options"]["allow_http"] is True
        assert "retry_config" in result
        assert result["retry_config"]["max_retries"] == 3
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_anonymous(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs(anonymous=True)

        assert "config" in result
        assert result["config"].get("skip_signature") is True
        assert "client_options" in result
        assert "retry_config" in result
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_with_custom_params(self):
        gcs = GCS()
        result = gcs.get_fsspec_kwargs(token="test-token", project="test-project")
        assert result["token"] == "test-token"
        assert result["project"] == "test-project"
        assert "client_options" in result
        assert "retry_config" in result
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_retries_backoff_override(self):
        custom_backoff = datetime.timedelta(seconds=10)
        gcs = GCS(retries=3, backoff=datetime.timedelta(seconds=5))
        result = gcs.get_fsspec_kwargs(retries=5, backoff=custom_backoff)

        assert result["retry_config"]["max_retries"] == 5
        assert result["retry_config"]["backoff"]["init_backoff"] == custom_backoff
        assert "anonymous" not in result

    def test_get_fsspec_kwargs_caller_overrides_not_clobbered(self):
        gcs = GCS()
        custom_retry = {"max_retries": 10, "retry_timeout": datetime.timedelta(minutes=10)}
        custom_client_options = {"timeout": "30s"}
        result = gcs.get_fsspec_kwargs(
            retry_config=custom_retry,
            client_options=custom_client_options,
        )

        # Caller-provided retry_config and client_options must not be overwritten by defaults
        assert result["retry_config"] is custom_retry
        assert result["retry_config"]["max_retries"] == 10
        assert result["client_options"] is custom_client_options
        assert result["client_options"]["timeout"] == "30s"


class TestABFSConfig:
    def test_get_fsspec_kwargs_default(self):
        abfs = ABFS()
        result = abfs.get_fsspec_kwargs()

        assert "config" not in result
        assert "client_options" in result
        assert result["client_options"]["timeout"] == "99999s"
        assert result["client_options"]["allow_http"] is True
        assert "retry_config" in result
        assert result["retry_config"]["max_retries"] == 3
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
