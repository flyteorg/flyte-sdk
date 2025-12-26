from base64 import decodebytes, encodebytes
from configparser import ConfigParser, NoOptionError, NoSectionError
from pathlib import Path
from typing import Optional

from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

_FLYTE_KEYRING_PATH: Path = Path.home() / ".flyte" / "keyring.cfg"


class SimplePlainTextKeyring(KeyringBackend):
    """
    Simple plain text keyring for remote notebook environments.

    This backend is only active when running in IPython/Jupyter notebooks.
    For local development, the system keyring is used instead.
    """

    @property
    def priority(self):
        """
        Return priority based on whether we're in a notebook environment.
        Negative priority means this backend will be skipped by keyring.
        """
        from flyte._tools import ipython_check

        if ipython_check():
            # In IPython/Jupyter - use this backend
            return 0.5
        else:
            # Not in IPython - skip this backend, let system keyring handle it
            return -1

    def get_password(self, service: str, username: str) -> Optional[str]:
        """Get password."""
        if not self.file_path.exists():
            return None

        config = ConfigParser(interpolation=None)
        config.read(self.file_path, encoding="utf-8")

        try:
            password_base64 = config.get(service, username).encode("utf-8")
            return decodebytes(password_base64).decode("utf-8")
        except (NoOptionError, NoSectionError):
            return None

    def delete_password(self, service: str, username: str) -> None:
        """Delete password."""
        if not self.file_path.exists():
            raise PasswordDeleteError("Config file does not exist")

        config = ConfigParser(interpolation=None)
        config.read(self.file_path, encoding="utf-8")

        try:
            if not config.remove_option(service, username):
                raise PasswordDeleteError("Password not found")
        except NoSectionError:
            raise PasswordDeleteError("Password not found")

        with self.file_path.open("w", encoding="utf-8") as config_file:
            config.write(config_file)

    def set_password(self, service: str, username: str, password: str) -> None:
        """Set password."""
        if not username:
            raise ValueError("Username must be provided")

        file_path = self._ensure_file_path()
        value = encodebytes(password.encode("utf-8")).decode("utf-8")

        config = ConfigParser(interpolation=None)
        config.read(file_path, encoding="utf-8")

        if not config.has_section(service):
            config.add_section(service)

        config.set(service, username, value)

        with file_path.open("w", encoding="utf-8") as config_file:
            config.write(config_file)

    def _ensure_file_path(self):
        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.file_path.is_file():
            self.file_path.touch(0o600)
        return self.file_path

    @property
    def file_path(self) -> Path:
        from flyte._initialize import get_init_config, is_initialized
        from flyte._logging import logger

        # Only try to use source_config_path if flyte.init() has been called
        if is_initialized():
            try:
                config = get_init_config()
                config_path = config.source_config_path
                if config_path and str(config_path.parent.name) == ".flyte":
                    # if the config is in a .flyte directory, use that as the path
                    return config_path.parent / "keyring.cfg"
            except Exception as e:
                # If anything fails, fall back to default path
                logger.debug(f"Skipping config-based keyring path due to error: {e}")
        else:
            # flyte.init() hasn't been called, use default path
            logger.debug("flyte.init() not called, using default keyring path")

        # Default path
        return _FLYTE_KEYRING_PATH

    def __repr__(self):
        return f"<{self.__class__.__name__}> at {self.file_path}>"
