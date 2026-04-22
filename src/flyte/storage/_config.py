from __future__ import annotations

import datetime
import os
import typing
from dataclasses import dataclass
from typing import ClassVar, Literal

from flyte._logging import logger
from flyte.config import set_if_exists


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Storage(object):
    """
    Data storage configuration that applies across any provider.
    """

    retries: int = 3
    backoff: datetime.timedelta = datetime.timedelta(seconds=5)
    enable_debug: bool = False
    attach_execution_metadata: bool = True

    _KEY_ENV_VAR_MAPPING: ClassVar[typing.Dict[str, str]] = {
        "enable_debug": "FLYTE_STORAGE_DEBUG",
        "retries": "FLYTE_STORAGE_RETRIES",
        "backoff": "FLYTE_STORAGE_BACKOFF_SECONDS",
    }

    _KEY_SKIP_SIGNATURE: ClassVar = "skip_signature"

    def get_fsspec_kwargs(self, anonymous: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Returns the configuration as kwargs for constructing an fsspec filesystem.
        """
        retries = kwargs.pop("retries", self.retries)
        backoff = kwargs.pop("backoff", self.backoff)

        if anonymous:
            config = kwargs.get("config", {})
            config[self._KEY_SKIP_SIGNATURE] = True
            kwargs["config"] = config

        kwargs.setdefault(
            "retry_config",
            {
                "max_retries": retries,
                "backoff": {
                    "base": 2,
                    "init_backoff": backoff,
                    "max_backoff": datetime.timedelta(seconds=16),
                },
                "retry_timeout": datetime.timedelta(minutes=3),
            },
        )
        kwargs.setdefault("client_options", {"timeout": "99999s", "allow_http": True})

        return kwargs

    @classmethod
    def _auto_as_kwargs(cls) -> typing.Dict[str, typing.Any]:
        retries = os.getenv(cls._KEY_ENV_VAR_MAPPING["retries"])
        backoff = os.getenv(cls._KEY_ENV_VAR_MAPPING["backoff"])
        enable_debug = os.getenv(cls._KEY_ENV_VAR_MAPPING["enable_debug"])

        kwargs: typing.Dict[str, typing.Any] = {}
        kwargs = set_if_exists(kwargs, "enable_debug", enable_debug)
        kwargs = set_if_exists(kwargs, "retries", retries)
        kwargs = set_if_exists(kwargs, "backoff", backoff)
        return kwargs

    @classmethod
    def auto(cls) -> Storage:
        """
        Construct the config object automatically from environment variables.
        """
        return cls(**cls._auto_as_kwargs())


@dataclass(init=True, repr=True, eq=True, frozen=True)
class S3(Storage):
    """
    S3 specific configuration.

    Authentication resolution used by Flyte + obstore:

    1. If explicit static credentials are provided via Flyte S3 inputs/environment
    (`access_key_id`/`secret_access_key`), those are used.

    2. If static credentials are not provided, and both `AWS_PROFILE` and
    `AWS_CONFIG_FILE` are available, Flyte configures a boto3-backed obstore
    credential provider so profile-based auth can be used. This requires that the `boto3` library
    is installed.

    3. If neither of the above applies, obstore uses the default AWS credential chain
    (for remote runs this commonly resolves via workload identity / IAM attached to
    the service account and then IMDS fallbacks where applicable).
    """

    endpoint: typing.Optional[str] = None
    access_key_id: typing.Optional[str] = None
    secret_access_key: typing.Optional[str] = None
    region: typing.Optional[str] = None
    addressing_style: typing.Optional[str] = None

    _KEY_ENV_VAR_MAPPING: ClassVar[typing.Dict[str, str]] = {
        "endpoint": "FLYTE_AWS_ENDPOINT",
        "access_key_id": "FLYTE_AWS_ACCESS_KEY_ID",
        "secret_access_key": "FLYTE_AWS_SECRET_ACCESS_KEY",
        "addressing_style": "FLYTE_AWS_S3_ADDRESSING_STYLE",
    } | Storage._KEY_ENV_VAR_MAPPING

    # Refer to https://github.com/developmentseed/obstore/blob/33654fc37f19a657689eb93327b621e9f9e01494/obstore/python/obstore/store/_aws.pyi#L11
    # for key and secret
    _CONFIG_KEY_FSSPEC_S3_KEY_ID: ClassVar[Literal["access_key_id"]] = "access_key_id"
    _CONFIG_KEY_FSSPEC_S3_SECRET: ClassVar = "secret_access_key"
    _CONFIG_KEY_ENDPOINT: ClassVar = "endpoint_url"

    @classmethod
    def auto(cls, region: str | None = None) -> S3:
        """
        :return: Config
        """
        endpoint = os.getenv(cls._KEY_ENV_VAR_MAPPING["endpoint"], None)
        access_key_id = os.getenv(cls._KEY_ENV_VAR_MAPPING["access_key_id"], None)
        secret_access_key = os.getenv(cls._KEY_ENV_VAR_MAPPING["secret_access_key"], None)
        addressing_style = os.getenv(cls._KEY_ENV_VAR_MAPPING["addressing_style"], None)

        kwargs = super()._auto_as_kwargs()
        kwargs = set_if_exists(kwargs, "endpoint", endpoint)
        kwargs = set_if_exists(kwargs, "access_key_id", access_key_id)
        kwargs = set_if_exists(kwargs, "secret_access_key", secret_access_key)
        kwargs = set_if_exists(kwargs, "region", region)
        kwargs = set_if_exists(kwargs, "addressing_style", addressing_style)

        return S3(**kwargs)

    @classmethod
    def for_sandbox(cls) -> S3:
        """
        :return:
        """
        kwargs = super()._auto_as_kwargs()
        final_kwargs = kwargs | {
            "endpoint": "http://localhost:4566",
            "access_key_id": "minio",
            "secret_access_key": "miniostorage",
        }
        return S3(**final_kwargs)  # type: ignore[arg-type]

    def _build_s3_credential_provider_from_config_file(
        self,
        aws_profile: str,
        aws_config_file: str,
        region: str | None,
    ) -> typing.Any:
        import boto3
        import boto3.session
        import botocore.session
        from obstore.auth.boto3 import Boto3CredentialProvider

        botocore_session = botocore.session.Session()
        botocore_session.set_config_variable("config_file", aws_config_file)
        boto3_session = boto3.session.Session(
            profile_name=aws_profile, botocore_session=botocore_session, region_name=region
        )
        return Boto3CredentialProvider(session=boto3_session)

    def get_fsspec_kwargs(self, anonymous: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        kwargs = super().get_fsspec_kwargs(anonymous=anonymous, **kwargs)

        config: typing.Dict[str, typing.Any] = kwargs.pop("config", {})
        if self._CONFIG_KEY_FSSPEC_S3_KEY_ID in kwargs or self.access_key_id:
            config[self._CONFIG_KEY_FSSPEC_S3_KEY_ID] = kwargs.pop(
                self._CONFIG_KEY_FSSPEC_S3_KEY_ID, self.access_key_id
            )
        if self._CONFIG_KEY_FSSPEC_S3_SECRET in kwargs or self.secret_access_key:
            config[self._CONFIG_KEY_FSSPEC_S3_SECRET] = kwargs.pop(
                self._CONFIG_KEY_FSSPEC_S3_SECRET, self.secret_access_key
            )
        if self._CONFIG_KEY_ENDPOINT in kwargs or self.endpoint:
            config["endpoint"] = kwargs.pop(self._CONFIG_KEY_ENDPOINT, self.endpoint)

        if self.addressing_style:
            config["virtual_hosted_style_request"] = self.addressing_style == "virtual"

        has_static_credentials = (
            self._CONFIG_KEY_FSSPEC_S3_KEY_ID in config and self._CONFIG_KEY_FSSPEC_S3_SECRET in config
        )

        if not anonymous and not has_static_credentials:
            aws_profile = os.getenv("AWS_PROFILE", None)
            aws_config_file = os.getenv("AWS_CONFIG_FILE", None)
            if aws_profile is not None and aws_config_file is not None:
                try:
                    kwargs["credential_provider"] = self._build_s3_credential_provider_from_config_file(
                        aws_profile=aws_profile,
                        aws_config_file=aws_config_file,
                        region=self.region or os.getenv("AWS_REGION", None),
                    )
                    logger.debug(
                        "Using S3 credentials from AWS config file with profile %s at %s",
                        aws_profile,
                        aws_config_file,
                    )
                except Exception as e:
                    logger.warning(
                        "Unable to initialize S3 profile/config credential provider (%s). "
                        "Falling back to default AWS credential resolution.",
                        e,
                    )

        if config:
            kwargs["config"] = config
        if self.region:
            kwargs["region"] = self.region

        return kwargs


@dataclass(init=True, repr=True, eq=True, frozen=True)
class GCS(Storage):
    """
    Any GCS specific configuration.
    """

    _KEY_ENV_VAR_MAPPING: ClassVar[dict[str, str]] = {} | Storage._KEY_ENV_VAR_MAPPING

    @classmethod
    def auto(cls) -> GCS:
        kwargs = super()._auto_as_kwargs()
        return GCS(**kwargs)

    def get_fsspec_kwargs(self, anonymous: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        return super().get_fsspec_kwargs(anonymous=anonymous, **kwargs)


@dataclass(init=True, repr=True, eq=True, frozen=True)
class ABFS(Storage):
    """
    Any Azure Blob Storage specific configuration.
    """

    account_name: typing.Optional[str] = None
    account_key: typing.Optional[str] = None
    tenant_id: typing.Optional[str] = None
    client_id: typing.Optional[str] = None
    client_secret: typing.Optional[str] = None

    _KEY_ENV_VAR_MAPPING: ClassVar[dict[str, str]] = {
        "account_name": "AZURE_STORAGE_ACCOUNT_NAME",
        "account_key": "AZURE_STORAGE_ACCOUNT_KEY",
        "tenant_id": "AZURE_TENANT_ID",
        "client_id": "AZURE_CLIENT_ID",
        "client_secret": "AZURE_CLIENT_SECRET",
    }

    @classmethod
    def auto(cls) -> ABFS:
        account_name = os.getenv(cls._KEY_ENV_VAR_MAPPING["account_name"], None)
        account_key = os.getenv(cls._KEY_ENV_VAR_MAPPING["account_key"], None)
        tenant_id = os.getenv(cls._KEY_ENV_VAR_MAPPING["tenant_id"], None)
        client_id = os.getenv(cls._KEY_ENV_VAR_MAPPING["client_id"], None)
        client_secret = os.getenv(cls._KEY_ENV_VAR_MAPPING["client_secret"], None)

        kwargs: typing.Dict[str, typing.Any] = {}
        kwargs = set_if_exists(kwargs, "account_name", account_name)
        kwargs = set_if_exists(kwargs, "account_key", account_key)
        kwargs = set_if_exists(kwargs, "tenant_id", tenant_id)
        kwargs = set_if_exists(kwargs, "client_id", client_id)
        kwargs = set_if_exists(kwargs, "client_secret", client_secret)
        return ABFS(**kwargs)

    def get_fsspec_kwargs(self, anonymous: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        kwargs = super().get_fsspec_kwargs(anonymous=anonymous, **kwargs)

        config: typing.Dict[str, typing.Any] = kwargs.pop("config", {})
        if "account_name" in kwargs or self.account_name:
            config["account_name"] = kwargs.get("account_name", self.account_name)
        if "account_key" in kwargs or self.account_key:
            config["account_key"] = kwargs.get("account_key", self.account_key)
        if "client_id" in kwargs or self.client_id:
            config["client_id"] = kwargs.get("client_id", self.client_id)
        if "client_secret" in kwargs or self.client_secret:
            config["client_secret"] = kwargs.get("client_secret", self.client_secret)
        if "tenant_id" in kwargs or self.tenant_id:
            config["tenant_id"] = kwargs.get("tenant_id", self.tenant_id)

        if config:
            kwargs["config"] = config

        return kwargs
