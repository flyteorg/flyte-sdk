import pathlib
import re
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Secret:
    """
    Secrets are used to inject sensitive information into tasks or image build context.
    Secrets can be mounted as environment variables or files.
     The secret key is the name of the secret in the secret store. The group is optional and maybe used with some
    secret stores to organize secrets. The secret_mount is used to specify how the secret should be mounted. If the
    secret_mount is set to "env" the secret will be mounted as an environment variable. If the secret_mount is set to
    "file" the secret will be mounted as a file. The as_env_var is an optional parameter that can be used to specify the
    name of the environment variable that the secret should be mounted as.

    Example:
    ```python
    @task(secrets="my-secret")
    async def my_task():
        # This will be set to the value of the secret. Note: The env var is always uppercase, and - is replaced with _.
        os.environ["MY_SECRET"]

    @task(secrets=Secret("my-openai-api-key", as_env_var="OPENAI_API_KEY"))
    async def my_task2():
        os.environ["OPENAI_API_KEY"]
    ```

    TODO: Add support for secret versioning (some stores) and secret groups (some stores) and mounting as files.

    :param key: The name of the secret in the secret store.
    :param group: The group of the secret in the secret store.
    :param mount: Use this to specify the path where the secret should be mounted.
    TODO: support arbitrary mount paths. Today only "/etc/flyte/secrets" is supported
    :param as_env_var: The name of the environment variable that the secret should be mounted as.
    """

    key: str
    group: Optional[str] = None
    mount: pathlib.Path | None = None
    as_env_var: Optional[str] = None

    def __post_init__(self):
        if not self.mount and not self.as_env_var:
            self.as_env_var = f"{self.group}_{self.key}" if self.group else self.key
            self.as_env_var = self.as_env_var.replace("-", "_").upper()
        if self.as_env_var is not None:
            pattern = r"^[A-Z_][A-Z0-9_]*$"
            if not re.match(pattern, self.as_env_var):
                raise ValueError(f"Invalid environment variable name: {self.as_env_var}, must match {pattern}")

    def stable_hash(self) -> str:
        """
        Deterministic, process-independent hash (as hex string).
        """
        import hashlib

        data = (
            self.key,
            self.group or "",
            str(self.mount) if self.mount else "",
            self.as_env_var or "",
        )
        joined = "|".join(data)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        """
        Deterministic hash function for the Secret class.
        """
        return int(self.stable_hash()[:16], 16)


SecretRequest = Union[str, Secret, List[str | Secret]]


def secrets_from_request(secrets: SecretRequest) -> List[Secret]:
    """
    Converts a secret request into a list of secrets.
    """
    if isinstance(secrets, str):
        return [Secret(key=secrets)]
    elif isinstance(secrets, Secret):
        return [secrets]
    else:
        return [Secret(key=s) if isinstance(s, str) else s for s in secrets]


if __name__ == "__main__":
    # Example usage
    secret1 = Secret(key="MY_SECRET", mount=pathlib.Path("/path/to/secret"), as_env_var="MY_SECRET_ENV")
    secret2 = Secret(
        key="ANOTHER_SECRET",
    )
    print(hash(secret1), hash(secret2))
