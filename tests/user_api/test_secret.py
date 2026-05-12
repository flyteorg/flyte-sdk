import pathlib

import pytest

import flyte
from flyte._secret import Secret, SecretRequest, secrets_from_request


def test_secret_basic():
    secret = Secret(key="my-secret")
    assert secret.key == "my-secret"
    assert secret.group is None
    assert secret.mount is None
    assert secret.as_env_var == "MY_SECRET"


def test_secret_auto_env_var_from_key():
    secret = Secret(key="my-api-key")
    assert secret.as_env_var == "MY_API_KEY"


def test_secret_auto_env_var_from_key_and_group():
    secret = Secret(key="api-key", group="openai")
    assert secret.as_env_var == "OPENAI_API_KEY"


def test_secret_explicit_env_var():
    secret = Secret(key="my-secret", as_env_var="OPENAI_API_KEY")
    assert secret.as_env_var == "OPENAI_API_KEY"


def test_secret_invalid_env_var():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        Secret(key="my-secret", as_env_var="invalid-name")


def test_secret_invalid_env_var_lowercase():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        Secret(key="my-secret", as_env_var="lowercase")


def test_secret_mount_valid():
    secret = Secret(key="my-secret", mount=pathlib.Path("/etc/flyte/secrets"))
    assert secret.mount == pathlib.Path("/etc/flyte/secrets")


def test_secret_mount_invalid():
    with pytest.raises(ValueError, match="Only /etc/flyte/secrets is supported"):
        Secret(key="my-secret", mount=pathlib.Path("/tmp/secrets"))


def test_secret_stable_hash_deterministic():
    s1 = Secret(key="test-key", group="test-group")
    s2 = Secret(key="test-key", group="test-group")
    assert s1.stable_hash() == s2.stable_hash()


def test_secret_stable_hash_different_keys():
    s1 = Secret(key="key1")
    s2 = Secret(key="key2")
    assert s1.stable_hash() != s2.stable_hash()


def test_secret_hash_works_in_set():
    s1 = Secret(key="test-key")
    s2 = Secret(key="test-key")
    assert hash(s1) == hash(s2)
    # dataclass eq means equal instances collapse in a set
    assert len({s1, s2}) == 1

    s3 = Secret(key="other-key")
    assert len({s1, s3}) == 2


def test_secrets_from_request_string():
    result = secrets_from_request("my-secret")
    assert len(result) == 1
    assert result[0].key == "my-secret"
    assert isinstance(result[0], Secret)


def test_secrets_from_request_secret_object():
    secret = Secret(key="my-secret", as_env_var="MY_SECRET")
    result = secrets_from_request(secret)
    assert len(result) == 1
    assert result[0] is secret


def test_secrets_from_request_list_of_strings():
    result = secrets_from_request(["secret1", "secret2"])
    assert len(result) == 2
    assert result[0].key == "secret1"
    assert result[1].key == "secret2"


def test_secrets_from_request_mixed_list():
    secret = Secret(key="explicit", as_env_var="EXPLICIT")
    result = secrets_from_request(["string-secret", secret])
    assert len(result) == 2
    assert result[0].key == "string-secret"
    assert result[1] is secret


def test_flyte_secret_importable():
    assert flyte.Secret is Secret
    assert flyte.SecretRequest is SecretRequest
