import os
import string
import sys
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
def original_std_streams():
    """
    Temporarily rebind ``sys.stdout`` / ``sys.stderr`` to the interpreter's original streams.

    cloudpickle only pickles the standard streams by reference when the object it encounters
    *is* the current ``sys.stdout`` / ``sys.stderr``; any other write-mode file raises
    ``PicklingError``. UI layers such as rich's Live/status spinner rebind those names to
    proxies for their duration, so cloudpickling an object graph that holds the real stderr
    (e.g. loguru's default sink, captured at import time) fails while a spinner is active.
    Wrap ``cloudpickle.dumps`` of user-supplied object graphs in this context so the identity
    check sees the original streams again.
    """
    captured_stdout, captured_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.__stdout__ if sys.__stdout__ is not None else captured_stdout
    sys.stderr = sys.__stderr__ if sys.__stderr__ is not None else captured_stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = captured_stdout, captured_stderr


@contextmanager
def _selector_policy():
    import asyncio

    original_policy = asyncio.get_event_loop_policy()  # ty: ignore[deprecated]  # kept until 3.16 drops the API
    try:
        if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        yield
    finally:
        asyncio.set_event_loop_policy(original_policy)  # ty: ignore[deprecated]  # kept until 3.16 drops the API
