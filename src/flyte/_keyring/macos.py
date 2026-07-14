import platform
import subprocess
from typing import Optional

from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

_SECURITY = "/usr/bin/security"
# errSecItemNotFound exit code of /usr/bin/security
_NOT_FOUND = 44


def _quote(value: str) -> str:
    """Quote a token for `security -i` (double quotes, backslash escapes)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class SecurityCliKeyring(KeyringBackend):
    """
    macOS login keychain accessed through /usr/bin/security instead of the
    Security-framework bindings used by keyring's native macOS backend.

    The keychain authorizes clients by code signature. Python interpreters
    from uv / python-build-standalone are ad-hoc signed with no identity, so
    every venv is a "different app": items written by one interpreter trigger
    a password prompt when read by another, and "Always Allow" grants never
    stick. /usr/bin/security is Apple-signed and identical for every venv, so
    routing all keychain access through it makes the prompts disappear while
    tokens stay in the encrypted keychain.
    """

    @property
    def priority(self):
        if platform.system() != "Darwin":
            return -1
        # Outrank keyring's native macOS backend (priority 5).
        return 6

    def get_password(self, service: str, username: str) -> Optional[str]:
        result = subprocess.run(
            [_SECURITY, "find-generic-password", "-a", username, "-s", service, "-w"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        # -w appends exactly one newline to the secret.
        return result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout

    def set_password(self, service: str, username: str, password: str) -> None:
        # Interactive mode keeps the secret out of the process argv.
        # -U updates in place; the item stays owned by /usr/bin/security.
        command = (
            f"add-generic-password -U -a {_quote(username)} -s {_quote(service)} -w {_quote(password)}\n"
        )
        result = subprocess.run(
            [_SECURITY, "-i"],
            input=command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"security add-generic-password failed: {result.stderr.strip()}")

    def delete_password(self, service: str, username: str) -> None:
        result = subprocess.run(
            [_SECURITY, "delete-generic-password", "-a", username, "-s", service],
            capture_output=True,
            text=True,
        )
        if result.returncode == _NOT_FOUND:
            raise PasswordDeleteError("Password not found")
        if result.returncode != 0:
            raise PasswordDeleteError(f"security delete-generic-password failed: {result.stderr.strip()}")

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
