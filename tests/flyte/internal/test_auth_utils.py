import base64

import pytest

from flyte.errors import InitializationError
from flyte.remote._client.auth._auth_utils import decode_api_key


@pytest.mark.skip("debugging only")
def test_decode():
    endpoint, client_id, _, org = decode_api_key("encoded-key==")
    assert endpoint == "dogfood-gcp.cloud-staging.union.ai"
    assert org == "None"
    assert client_id == "dogfood-gcp-EAGER_API_KEY"


def test_decode_valid_api_key():
    raw = "example.union.ai:my-client-id:my-secret:my-org"
    encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
    endpoint, client_id, client_secret, org = decode_api_key(encoded)
    assert endpoint == "example.union.ai"
    assert client_id == "my-client-id"
    assert client_secret == "my-secret"
    assert org == "my-org"


def test_decode_malformed_base64_raises_initialization_error():
    """A garbled API key (invalid base64) must raise a typed user error, not leak a raw
    binascii error as a crash report (FLYTE-SDK-5C)."""
    # 5 data characters -> count is 1 more than a multiple of 4 -> binascii.Error
    with pytest.raises(InitializationError) as exc_info:
        decode_api_key("AAAAA")
    assert exc_info.value.kind == "user"
    assert exc_info.value.code == "InvalidApiKey"


def test_decode_wrong_part_count_raises_initialization_error():
    """A well-formed base64 value that does not contain 4 ':'-separated parts is also user
    input error, surfaced as the same typed user error."""
    encoded = base64.b64encode(b"only-one-part").decode("utf-8")
    with pytest.raises(InitializationError) as exc_info:
        decode_api_key(encoded)
    assert exc_info.value.kind == "user"
    assert exc_info.value.code == "InvalidApiKey"
