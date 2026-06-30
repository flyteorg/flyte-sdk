from flyte._utils.helpers import base36_encode


def test_encode_max():
    # Test that the max value coming from md5 hash can be correctly encoded.
    max_128bit_value = 2**128 - 1
    max_byte_value = int.to_bytes(max_128bit_value, 16, "big")
    max_hash = base36_encode(max_byte_value)
    assert max_hash == "f5lxx1zz5pnorynqglhzmsp33"
    assert len(max_hash) <= 30
