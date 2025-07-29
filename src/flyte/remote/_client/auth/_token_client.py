import asyncio
import base64
import enum
import typing
import urllib.parse
from datetime import datetime, timedelta

import httpx
import pydantic

from flyte._logging import logger
from flyte.remote._client.auth.errors import AuthenticationError, AuthenticationPending

utf_8 = "utf-8"

# Errors that Token endpoint will return
error_slow_down = "slow_down"
error_auth_pending = "authorization_pending"


# Grant Types
class GrantType(str, enum.Enum):
    CLIENT_CREDS = "client_credentials"
    DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"
    REFRESH_TOKEN = "refresh_token"


class DeviceCodeResponse(pydantic.BaseModel):
    """
    Response from device auth flow endpoint
    {
        'device_code': 'code',
         'user_code': 'BNDJJFXL',
         'verification_uri': 'url',
         'expires_in': 600,
         'interval': 5
    }

    Attributes:
        device_code (str): The device verification code.
        user_code (str): The user-facing code that should be entered on the verification page.
        verification_uri (str): The URL where the user should enter the user_code.
        expires_in (int): The lifetime in seconds of the device code and user code.
        interval (int): The minimum amount of time in seconds to wait between polling requests.
    """

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int

    @classmethod
    def from_json_response(cls, j: typing.Dict) -> "DeviceCodeResponse":
        """
        Create a DeviceCodeResponse instance from a JSON response dictionary.

        :param j: The JSON response dictionary containing device code information
        :return: A new instance with values from the JSON response
        """
        return cls(
            device_code=j["device_code"],
            user_code=j["user_code"],
            verification_uri=j["verification_uri"],
            expires_in=j["expires_in"],
            interval=j["interval"],
        )


def get_basic_authorization_header(client_id: str, client_secret: str) -> str:
    """
    This function transforms the client id and the client secret into a header that conforms with http basic auth.
    It joins the id and the secret with a : then base64 encodes it, then adds the appropriate text. Secrets are
    first URL encoded to escape illegal characters.

    :param client_id: The client ID for authentication
    :param client_secret: The client secret for authentication
    :rtype: str
    """
    encoded = urllib.parse.quote_plus(client_secret)
    concatenated = "{}:{}".format(client_id, encoded)
    return "Basic {}".format(base64.b64encode(concatenated.encode(utf_8)).decode(utf_8))


async def get_token(
    token_endpoint: str,
    http_session: httpx.AsyncClient,
    scopes: typing.Optional[typing.List[str]] = None,
    authorization_header: typing.Optional[str] = None,
    client_id: typing.Optional[str] = None,
    device_code: typing.Optional[str] = None,
    audience: typing.Optional[str] = None,
    grant_type: GrantType = GrantType.CLIENT_CREDS,
    http_proxy_url: typing.Optional[str] = None,
    verify: typing.Optional[typing.Union[bool, str]] = None,
    refresh_token: typing.Optional[str] = None,
) -> typing.Tuple[str, str | None, int]:
    """
    Retrieves an access token from the specified token endpoint.

    :param token_endpoint: The endpoint URL for token retrieval
    :param http_session: HTTP session to use for requests
    :param scopes: Optional list of scopes to request during authentication
    :param authorization_header: Optional authorization header value
    :param client_id: Optional client ID for authentication
    :param device_code: Optional device code for device flow authentication
    :param audience: Optional audience for the token
    :param grant_type: The grant type to use (default: CLIENT_CREDS)
    :param http_proxy_url: Optional HTTP proxy URL
    :param verify: Whether to verify SSL certificates (bool or path to cert)
    :param refresh_token: Optional refresh token for token refresh
    :return: A tuple of (access_token, refresh_token, expires_in)

    :param token_endpoint: The URL of the token endpoint
    :param scopes: List of scopes to request
    :param authorization_header: Authorization header value if using client credentials
    :param client_id: The client ID to use for authentication
    :param device_code: The device code when using device code flow
    :param audience: The audience value to request
    :param grant_type: The OAuth grant type to use
    :param http_proxy_url: HTTP proxy URL if needed
    :param verify: SSL verification mode
    :param http_session: An existing HTTP client session
    :param refresh_token: Refresh token for refresh token flow
    :return: A tuple containing (access_token, refresh_token, expires_in)

    Raises:
        AuthenticationPending: When authentication is still pending (for device code flow).
        AuthenticationError: When authentication fails for any reason.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    if authorization_header:
        headers["Authorization"] = authorization_header
    body = {
        "grant_type": grant_type.value,
    }
    if client_id:
        body["client_id"] = client_id
    if device_code:
        body["device_code"] = device_code
    if scopes is not None:
        body["scope"] = " ".join(s.strip("' ") for s in scopes).strip("[]'")
    if audience:
        body["audience"] = audience
    if refresh_token:
        body["refresh_token"] = refresh_token

    response = await http_session.post(token_endpoint, data=body, headers=headers)

    if not response.is_success:
        j = response.json()
        if "error" in j:
            err = j["error"]
            if err == error_auth_pending or err == error_slow_down:
                raise AuthenticationPending(f"Token not yet available, try again in some time {err}")
        logger.error("Status Code ({}) received from IDP: {}".format(response.status_code, response.text))
        raise AuthenticationError("Status Code ({}) received from IDP: {}".format(response.status_code, response.text))

    j = response.json()
    new_refresh_token = None
    if "refresh_token" in j:
        new_refresh_token = j["refresh_token"]
    else:
        logger.info("No refresh token received, this is expected for client credentials flow")

    return j["access_token"], new_refresh_token, j["expires_in"]


async def get_device_code(
    device_auth_endpoint: str,
    client_id: str,
    http_session: httpx.AsyncClient,
    *,
    audience: typing.Optional[str] = None,
    scopes: typing.Optional[typing.List[str]] = None,
) -> DeviceCodeResponse:
    """
    Retrieves the device authentication code that can be used to authenticate the request using a browser on a
    separate device.

    :param device_auth_endpoint: The URL of the device authorization endpoint
    :param client_id: The client ID to use for authentication
    :param audience: The audience value to request
    :param scopes: List of scopes to request
    :param http_proxy_url: HTTP proxy URL if needed
    :param verify: SSL verification mode
    :param http_session: An existing HTTP client session
    :return: An object containing the device code and related information
    :raises AuthenticationError: When device code retrieval fails
    """
    _scope = " ".join(s.strip("' ") for s in scopes).strip("[]'") if scopes is not None else ""
    payload = {"client_id": client_id, "scope": _scope, "audience": audience}
    resp = await http_session.post(device_auth_endpoint, data=payload)
    if not resp.is_success:
        raise AuthenticationError(
            f"Unable to retrieve Device Authentication Code for {payload},"
            f" Status Code {resp.status_code} Reason {resp.json()}"
        )
    return DeviceCodeResponse.from_json_response(resp.json())


async def poll_token_endpoint(
    resp: DeviceCodeResponse,
    *,
    token_endpoint: str,
    client_id: str,
    http_session: httpx.AsyncClient,
    audience: typing.Optional[str] = None,
    scopes: typing.Optional[typing.List[str]] = None,
    http_proxy_url: typing.Optional[str] = None,
    verify: typing.Optional[typing.Union[bool, str]] = None,
) -> typing.Tuple[str, str | None, int]:
    """
    Polls the token endpoint until authentication is complete or times out.

    This function repeatedly calls the token endpoint at the specified interval until either:
    1. Authentication is successful and a token is returned
    2. The device code expires (as specified in the DeviceCodeResponse)

    :param resp: The device code response from a previous call to get_device_code
    :param token_endpoint: The URL of the token endpoint
    :param client_id: The client ID to use for authentication
    :param audience: The audience value to request
    :param scopes: Space-separated list of scopes to request
    :param http_proxy_url: HTTP proxy URL if needed
    :param verify: SSL verification mode
    :return: A tuple containing (access_token, refresh_token, expires_in)
    :raises AuthenticationError: When authentication fails or times out
    """
    tick = datetime.now()
    interval = timedelta(seconds=resp.interval)
    end_time = tick + timedelta(seconds=resp.expires_in)
    while tick < end_time:
        try:
            access_token, refresh_token, expires_in = await get_token(
                token_endpoint,
                grant_type=GrantType.DEVICE_CODE,
                client_id=client_id,
                audience=audience,
                scopes=scopes,
                device_code=resp.device_code,
                http_proxy_url=http_proxy_url,
                verify=verify,
                http_session=http_session,
            )
            logger.debug(f"Authentication successful, access token received, expires in {expires_in} seconds")
            return access_token, refresh_token, expires_in
        except AuthenticationPending:
            ...
        except Exception as e:
            logger.warning(f"Authentication failed, reason {e}")
            raise e
        logger.debug(f"Authentication pending, ..., waiting for {resp.interval} seconds")
        await asyncio.sleep(interval.total_seconds())
        tick = tick + interval
    raise AuthenticationError("Authentication failed!")
