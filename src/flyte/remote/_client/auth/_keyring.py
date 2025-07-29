import hashlib  # Added import for hashing
import typing
from urllib.parse import urlparse  # Added import

import keyring
import pydantic
from keyring.errors import NoKeyringError, PasswordDeleteError

from flyte._logging import logger


def strip_scheme(url: str) -> str:
    """
    Strips the scheme from a URL.
    Handles cases like:
    - dns:///foo.com -> foo.com
    - https://foo.com -> foo.com
    - https://foo.com/blah -> foo.com/blah
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme == "dns":
        return parsed_url.path.lstrip("/")
    return f"{parsed_url.netloc}{parsed_url.path}" if parsed_url.netloc else url


class Credentials(pydantic.BaseModel):
    """
    Stores the credentials together
    """

    access_token: str
    for_endpoint: str = "flyte-default"
    id: str = ""
    refresh_token: str | None = None
    expires_in: int | None = None

    @pydantic.field_validator("for_endpoint", mode="after")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        return strip_scheme(v)

    @pydantic.model_validator(mode="after")
    def compute_id(self) -> "Credentials":
        """Computes the id field as a hash of the access_token."""
        if self.access_token:
            self.id = hashlib.md5(self.access_token.encode()).hexdigest()
        return self


class KeyringStore:
    """
    Methods to access Keyring Store.
    """

    _access_token_key = "access_token"
    _refresh_token_key = "refresh_token"

    @staticmethod
    def store(credentials: Credentials) -> Credentials:
        """
        Stores the provided credentials in the system keyring.

        This method stores the access token, refresh token (if available), and ID token (if available)
        in the system keyring, using the endpoint as the service name and specific key names for each token type.

        :param credentials: The credentials object containing tokens to store
        :return: The same credentials object that was passed in
        :raises: Logs but does not raise NoKeyringError if the system keyring is not available
        """
        try:
            if credentials.refresh_token:
                keyring.set_password(
                    credentials.for_endpoint,
                    KeyringStore._refresh_token_key,
                    credentials.refresh_token,
                )
            keyring.set_password(
                credentials.for_endpoint,
                KeyringStore._access_token_key,
                credentials.access_token,
            )
        except NoKeyringError as e:
            logger.debug(f"KeyRing not available, tokens will not be cached. Error: {e}")
        except Exception as e:
            logger.debug(f"Failed to store tokens in keyring. Error: {e}")
        return credentials

    @staticmethod
    def retrieve(for_endpoint: str) -> typing.Optional[Credentials]:
        """
        Retrieves stored credentials from the system keyring for the specified endpoint.

        This method attempts to retrieve the access token, refresh token, and ID token from the system keyring
        using the endpoint as the service name. The endpoint URL scheme is stripped before lookup.

        :param for_endpoint: The endpoint URL to retrieve credentials for
        :return: A Credentials object containing the retrieved tokens, or None if no tokens were found
                 or if the system keyring is not available
        """
        for_endpoint = strip_scheme(for_endpoint)
        try:
            refresh_token = keyring.get_password(for_endpoint, KeyringStore._refresh_token_key)
            access_token = keyring.get_password(for_endpoint, KeyringStore._access_token_key)
        except NoKeyringError as e:
            logger.debug(f"KeyRing not available, tokens will not be cached. Error: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve tokens from keyring. Error: {e}")
            return None

        if not access_token:
            logger.debug("No access token found in keyring.")
            return None

        return Credentials(
            access_token=access_token,
            refresh_token=refresh_token,
            for_endpoint=for_endpoint,
            expires_in=None,
        )

    @staticmethod
    def delete(for_endpoint: str):
        """
        Deletes all stored credentials for the specified endpoint from the system keyring.

        This method attempts to delete the access token, refresh token, and ID token from the system keyring
        using the endpoint as the service name. The endpoint URL scheme is stripped before lookup.

        :param for_endpoint: The endpoint URL to delete credentials for
        """
        for_endpoint = strip_scheme(for_endpoint)

        def _delete_key(key):
            """
            Helper function to delete a specific key from the keyring.

            :param key: The key name to delete
            """
            try:
                keyring.delete_password(for_endpoint, key)
            except PasswordDeleteError as e:
                logger.debug(f"Key {key} not found in key store, Ignoring. Error: {e}")
            except NoKeyringError as e:
                logger.debug(f"KeyRing not available, Key {key} deletion failed. Error: {e}")
            except NotImplementedError as e:
                logger.debug(f"Key {key} deletion not implemented in keyring backend. Error: {e}")
            except Exception as e:
                logger.debug(f"Failed to delete key {key} from keyring. Error: {e}")

        _delete_key(KeyringStore._access_token_key)
        _delete_key(KeyringStore._refresh_token_key)
