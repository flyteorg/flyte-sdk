import os
import string
import typing
from contextlib import contextmanager
from pathlib import Path


def load_proto_from_file(pb2_type, path):
    with open(path, "rb") as reader:
        out = pb2_type()
        out.ParseFromString(reader.read())
        return out


def write_proto_to_file(proto, path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as writer:
        writer.write(proto.SerializeToString())


def str2bool(value: typing.Optional[str]) -> bool:
    """
    Convert a string to a boolean. This is useful for parsing environment variables.
    :param value: The string to convert to a boolean
    :return: the boolean value
    """
    if value is None:
        return False
    return value.lower() in ("true", "t", "1")


BASE36_ALPHABET = string.digits + string.ascii_lowercase  # 0-9 + a-z (36 characters)


def base36_encode(byte_data: bytes) -> str:
    """
    This function expects to encode bytes coming from an hd5 hash function into a base36 encoded string.
    md5 shas are limited to 128 bits, so the maximum byte value should easily fit into a 30 character long string.
    If the input is too large howeer
    """
    # Convert bytes to a big integer
    num = int.from_bytes(byte_data, byteorder="big")

    # Convert integer to base36 string
    if num == 0:
        return BASE36_ALPHABET[0]

    base36 = []
    while num:
        num, rem = divmod(num, 36)
        base36.append(BASE36_ALPHABET[rem])
    return "".join(reversed(base36))


@contextmanager
def _selector_policy():
    import asyncio

    original_policy = asyncio.get_event_loop_policy()
    try:
        if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        yield
    finally:
        asyncio.set_event_loop_policy(original_policy)
