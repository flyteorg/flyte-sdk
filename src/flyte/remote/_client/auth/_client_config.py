import asyncio
import typing
from abc import abstractmethod

import grpc.aio
import pydantic
from flyteidl.service.auth_pb2 import OAuth2MetadataRequest, PublicClientAuthConfigRequest
from flyteidl.service.auth_pb2_grpc import AuthMetadataServiceStub

AuthType = typing.Literal["ClientSecret", "Pkce", "ExternalCommand", "DeviceFlow"]


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

    def __init__(self, unauthenticated_channel: grpc.aio.Channel):
        self._unauthenticated_channel = unauthenticated_channel

    async def get_client_config(self) -> ClientConfig:
        """
        Retrieves the ClientConfig from the given grpc.Channel assuming  AuthMetadataService is available
        """
        metadata_service = AuthMetadataServiceStub(self._unauthenticated_channel)
        oauth2_metadata_task = metadata_service.GetOAuth2Metadata(OAuth2MetadataRequest())
        public_client_config_task = metadata_service.GetPublicClientConfig(PublicClientAuthConfigRequest())
        oauth2_metadata, public_client_config = await asyncio.gather(oauth2_metadata_task, public_client_config_task)
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
