import subprocess
from typing import Optional

from keyring.errors import KeyringError, PasswordDeleteError, PasswordSetError

# macOS's built-in keychain CLI. Absolute path so a malicious `security` earlier
# in $PATH can't intercept tokens; SIP guarantees this one is Apple's. Its stable
# code signature is the whole point of this class: the keychain authorizes
# clients per app signature, and this binary is the same "app" from every venv.
_SECURITY = "/usr/bin/security"
# errSecItemNotFound exit code of /usr/bin/security
_NOT_FOUND = 44
# Generous ceiling for a keychain-unlock dialog; prevents the auth path from
# hanging forever on a wedged SecurityAgent.
_TIMEOUT_S = 60


class SecurityCliKeyring:
    """
    macOS default keychain accessed through /usr/bin/security instead of the
    Security-framework bindings used by the `keyring` package's native backend.

    Why: the keychain authorizes clients by code signature. Python interpreters
    from uv / python-build-standalone are ad-hoc signed with no identity, so
    every venv is a "different app": items written by one interpreter trigger a
    password prompt when read from another, and "Always Allow" grants never
    stick. /usr/bin/security is Apple-signed and identical for every venv, so
    items it creates are read back with zero prompts.

    Access-control tradeoff (deliberate): items created this way trust
    /usr/bin/security, so any process running as the same macOS user can read
    them through the same CLI without a prompt. That is the same exposure as
    the file-based credential stores of gcloud/aws/kubectl, with the added
    benefit that the keychain is encrypted at rest and locks with the session.

    This class is used directly by flyte's KeyringStore on Darwin. It is
    deliberately NOT registered as a `keyring` entry-point backend: entry
    points are process-global, and installing flyte must not change credential
    storage for unrelated packages.
    """

    def get_password(self, service: str, username: str) -> Optional[str]:
        result = subprocess.run(
            [_SECURITY, "find-generic-password", "-a", username, "-s", service, "-w"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
        if result.returncode == _NOT_FOUND:
            return None
        if result.returncode != 0:
            # A locked keychain or denied access is not "no password stored";
            # surface it instead of silently triggering a fresh login.
            raise KeyringError(
                f"security find-generic-password failed (code {result.returncode}): {result.stderr.strip()}"
            )
        # -w appends exactly one newline to the secret.
        return result.stdout.removesuffix("\n")

    def set_password(self, service: str, username: str, password: str) -> None:
        # Recreate instead of updating (-U): an update preserves the existing
        # item's ACL, so a legacy item created by a python interpreter would
        # stay owned by that interpreter and keep prompting. Delete + add makes
        # the item owned by /usr/bin/security.
        try:
            self.delete_password(service, username)
        except PasswordDeleteError:
            pass
        # The secret rides in argv: execve passes it verbatim — no command
        # interpreter, no quoting, no length limit. (`security -i` is unsafe as
        # a secret transport: its line parser executes embedded newlines as
        # further commands and truncates at 4096 bytes.) argv is briefly
        # visible to same-user `ps`, which adds no exposure here: the stored
        # item is readable by any same-user process through this same CLI
        # anyway (see class docstring).
        result = subprocess.run(
            [_SECURITY, "add-generic-password", "-a", username, "-s", service, "-w", password],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise PasswordSetError(
                f"security add-generic-password failed (code {result.returncode}): {result.stderr.strip()}"
            )

    def delete_password(self, service: str, username: str) -> None:
        result = subprocess.run(
            [_SECURITY, "delete-generic-password", "-a", username, "-s", service],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
        if result.returncode == _NOT_FOUND:
            raise PasswordDeleteError("Password not found")
        if result.returncode != 0:
            raise PasswordDeleteError(
                f"security delete-generic-password failed (code {result.returncode}): {result.stderr.strip()}"
            )

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
