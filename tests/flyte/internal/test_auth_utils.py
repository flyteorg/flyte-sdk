import pytest

from flyte.remote._client.auth._auth_utils import decode_api_key


@pytest.mark.skip("debugging only")
def test_decode():
    endpoint, client_id, _, org = decode_api_key("encoded-key==")
    assert endpoint == "dogfood-gcp.cloud-staging.union.ai"
    assert org == "None"
    assert client_id == "dogfood-gcp-EAGER_API_KEY"
