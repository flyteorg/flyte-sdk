import typing
from abc import abstractmethod

import pydantic
from flyteidl2.auth.auth_service_connect import AuthMetadataServiceClient
from flyteidl2.auth.auth_service_pb2 import GetOAuth2MetadataRequest, GetPublicClientConfigRequest


AuthType = typing.Literal["ClientSecret", "Pkce", "ExternalCommand", "DeviceFlow", "Passthrough"]


class ClientConfig(pydantic.BaseModel):
    """
    Client Configuration that is needed by the authenticator
    """

    token_endpoint: str
    authorization_endpoint: str
    redirect_uri: str
    client_id: str
    device_authorization_endpoint: typing.Optional[str] = None
    scopes: typing.Optional[typing.List[str]] = None
    header_key: str = "authorization"
    audience: typing.Optional[str] = None

    def with_override(self, other: "ClientConfig") -> "ClientConfig":
        """
        Returns a new ClientConfig instance with the values from the other instance overriding the current instance.
        """
        return ClientConfig(
            token_endpoint=other.token_endpoint or self.token_endpoint,
            authorization_endpoint=other.authorization_endpoint or self.authorization_endpoint,
            redirect_uri=other.redirect_uri or self.redirect_uri,
            client_id=other.client_id or self.client_id,
            device_authorization_endpoint=other.device_authorization_endpoint or self.device_authorization_endpoint,
            scopes=other.scopes or self.scopes,
            header_key=other.header_key or self.header_key,
            audience=other.audience or self.audience,
        )


class LocalClientConfigOverrides(pydantic.BaseModel):
    """
    Partial public-client configuration read from local config files.
    """

    redirect_uri: typing.Optional[str] = None
    client_id: typing.Optional[str] = None
    scopes: typing.Optional[typing.List[str]] = None
    header_key: typing.Optional[str] = None
    audience: typing.Optional[str] = None

    def has_required_public_client_fields(self) -> bool:
        return bool(self.client_id and self.redirect_uri and self.header_key and self.scopes)


class ClientConfigStore(object):
    """
    Client Config store retrieve client config. this can be done in multiple ways
    """

    @abstractmethod
    async def get_client_config(self) -> ClientConfig: ...


class StaticClientConfigStore(ClientConfigStore):
    def __init__(self, cfg: ClientConfig):
        self._cfg = cfg

    async def get_client_config(self) -> ClientConfig:
        return self._cfg


class RemoteClientConfigStore(ClientConfigStore):
    """
    This class implements the ClientConfigStore that is served by the Flyte Server, that implements AuthMetadataService
    """

    def __init__(
        self,
        endpoint: str,
        http_client=None,
        client_config_overrides: LocalClientConfigOverrides | None = None,
    ):
        self._endpoint = endpoint
        self._client = AuthMetadataServiceClient(address=endpoint, http_client=http_client)
        self._client_config_overrides = client_config_overrides

    async def get_client_config(self) -> ClientConfig:
        """
        Retrieves the ClientConfig from the AuthMetadataService via ConnectRPC.
        """

        oauth2_metadata = await self._client.get_o_auth2_metadata(GetOAuth2MetadataRequest())

        if self._client_config_overrides and self._client_config_overrides.has_required_public_client_fields():
            redirect_uri = self._client_config_overrides.redirect_uri
            client_id = self._client_config_overrides.client_id
            scopes = self._client_config_overrides.scopes
            header_key = self._client_config_overrides.header_key
            assert redirect_uri is not None
            assert client_id is not None
            assert scopes is not None
            assert header_key is not None
            return ClientConfig(
                token_endpoint=oauth2_metadata.token_endpoint,
                authorization_endpoint=oauth2_metadata.authorization_endpoint,
                redirect_uri=redirect_uri,
                client_id=client_id,
                scopes=scopes,
                header_key=header_key,
                device_authorization_endpoint=oauth2_metadata.device_authorization_endpoint,
                audience=self._client_config_overrides.audience,
            )

        public_client_config = await self._client.get_public_client_config(GetPublicClientConfigRequest())

        return ClientConfig(
            token_endpoint=oauth2_metadata.token_endpoint,
            authorization_endpoint=oauth2_metadata.authorization_endpoint,
            redirect_uri=public_client_config.redirect_uri,
            client_id=public_client_config.client_id,
            scopes=public_client_config.scopes,
            header_key=public_client_config.authorization_metadata_key,
            device_authorization_endpoint=oauth2_metadata.device_authorization_endpoint,
            audience=public_client_config.audience,
        )
