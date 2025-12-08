from __future__ import annotations

import traceback
from typing import AsyncGenerator, AsyncIterator, Iterator, Union

import pytest

from flyte.syncify import syncify


class DummyAsyncIterator(AsyncIterator[str]):
    """
    A dummy async iterator for testing purposes.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.current = 0

    async def __anext__(self) -> str:
        if self.current >= self.limit:
            raise StopAsyncIteration
        item = f"Item {self.current + 1}"
        self.current += 1
        return item

    def __aiter__(self):
        return self


@syncify
async def async_iterator(limit: int = 100) -> AsyncIterator[str] | Iterator[str]:
    """
    An async generator that yields items from 1 to limit.
    """
    return DummyAsyncIterator(limit)


@syncify
async def async_hello(x: str) -> str:
    return f"Hello, Async World {x}!"


@syncify
async def listall(limit: int = 100) -> Union[AsyncIterator[str], Iterator[str]]:
    """
    An async generator that yields items from 1 to limit.
    """
    for i in range(limit):
        yield f"Item {i + 1}"


@syncify
async def listall_proxy(limit: int = 100) -> Union[AsyncIterator[str], Iterator[str]]:
    """
    An async generator that proxies the listall function.
    """
    async for x in listall.aio(limit=limit):
        yield x


async def listall_proxy_separate_loop(limit: int = 100) -> AsyncIterator[str]:
    """
    This method is not decorated with syncify, so it should not be converted to a synchronous method.
    """
    async for x in listall.aio(limit=limit):
        yield x


class MyClass:
    def __init__(self, name: str):
        self.name = name

    @syncify
    async def async_method(self) -> str:
        return f"Hello, {self.name} from async method!"

    @syncify
    async def async_method_proxy(self) -> str:
        return await self.async_method.aio()

    async def async_method_proxy_separate_loop(self) -> str:
        """
        This method is not decorated with syncify, so it should not be converted to a synchronous method.
        """
        return await self.async_method.aio()

    @syncify
    @classmethod
    async def class_method(cls, x: str) -> MyClass:
        return cls(name=x)

    @syncify
    @staticmethod
    async def static_method(x: str) -> str:
        return f"Hello from static method with {x}!"

    @syncify
    @classmethod
    async def listall(cls, limit: int = 100) -> Union[AsyncIterator[str], Iterator[str]]:
        for i in range(limit):
            yield f"Item {i + 1}"

    @syncify
    @classmethod
    async def listall_proxy(cls, limit: int = 100) -> Union[AsyncIterator[str], Iterator[str]]:
        async for x in cls.listall.aio(limit=limit):
            yield x

    @classmethod
    async def listall_proxy_separate_loop(cls, limit: int = 100) -> AsyncIterator[str]:
        """
        This method is not decorated with syncify, so it should not be converted to a synchronous method.
        """
        async for x in cls.listall.aio(limit=limit):
            yield x


def test_async_hello_sync():
    assert async_hello(x="Async World") == "Hello, Async World Async World!"


def test_listall_sync():
    items = list(listall(limit=5))
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


@pytest.mark.asyncio
async def test_listall():
    items = [item async for item in listall.aio(limit=5)]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


def test_listall_proxy_sync():
    items = list(listall_proxy(limit=5))
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


@pytest.mark.asyncio
async def test_listall_proxy_separate_loop_sync():
    items = [item async for item in listall_proxy_separate_loop(limit=5)]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


def test_async_method_sync():
    my_instance = MyClass("Test")
    assert my_instance.async_method() == "Hello, Test from async method!"


def test_class_method_sync():
    my_instance = MyClass.class_method(x="Async Class Method")
    assert my_instance.name == "Async Class Method"


def test_static_method_sync():
    result = MyClass.static_method(x="Async Static Method")
    assert result == "Hello from static method with Async Static Method!"


def test_listall_instance_sync():
    items = list(MyClass.listall(limit=5))
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


def test_run_listall_in_bg_loop():
    items = list(MyClass.listall_proxy(limit=5))
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


@pytest.mark.asyncio
async def test_async_hello_async():
    result = await async_hello.aio(x="World")
    assert result == "Hello, Async World World!"


@pytest.mark.asyncio
async def test_async_method_async():
    my_instance = MyClass("Test")
    method_result = await my_instance.async_method.aio()
    assert method_result == "Hello, Test from async method!"


@pytest.mark.asyncio
async def test_async_method_proxy():
    my_instance = MyClass("Test")
    method_result = await my_instance.async_method_proxy.aio()
    assert method_result == "Hello, Test from async method!"


@pytest.mark.asyncio
async def test_async_method_proxy_separate_loop():
    my_instance = MyClass("Test")
    method_result = await my_instance.async_method_proxy_separate_loop()
    assert method_result == "Hello, Test from async method!"


@pytest.mark.asyncio
async def test_class_method_async():
    class_method_result = await MyClass.class_method.aio("Async Class Method")
    assert class_method_result.name == "Async Class Method"


@pytest.mark.asyncio
async def test_static_method_async():
    static_method_result = await MyClass.static_method.aio("Async Static Method")
    assert static_method_result == "Hello from static method with Async Static Method!"


@pytest.mark.asyncio
async def test_listall_async():
    c = MyClass.listall.aio(limit=5)
    items = [item async for item in c]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


@pytest.mark.asyncio
async def test_listall_proxy_separate_loop():
    items = [item async for item in MyClass.listall_proxy_separate_loop(limit=5)]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


@pytest.mark.asyncio
async def test_listall_proxy_async():
    items = [item async for item in MyClass.listall_proxy.aio(limit=5)]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items


def test_illegal_syncify_sync_methods():
    with pytest.raises(
        TypeError,
        match=r"Syncify can only be applied to async functions, async generators, async classmethods or staticmethods.",
    ):

        @syncify
        def sync_function(x: str) -> str:
            return f"Hello, {x}!"  # This should raise an error since it's not async


def test_illegal_syncify_sync_methods_class():
    with pytest.raises(
        TypeError,
        match=r"Syncify can only be applied to async functions, async generators, async classmethods or staticmethods.",
    ):

        class MySyncClass:
            @syncify
            def sync_method(self) -> str:
                return "This should not work"  # This should raise an error since it's not async


def test_illegal_syncify_classes():
    with pytest.raises(
        TypeError,
        match=r"Syncify can only be applied to async functions, async generators, async classmethods or staticmethods.",
    ):

        @syncify
        class MySyncClass:
            def sync_class_method(cls) -> str:
                pass


def test_context_propagation():
    """
    Test that context is propagated correctly when using syncify.
    """
    import contextvars

    my_var = contextvars.ContextVar("my_var")
    my_var.set("test_value")

    @syncify
    async def context_test() -> str:
        v = my_var.get()
        my_var.set("test_value-updated")
        return v

    result = context_test()
    assert my_var.get() == "test_value", "Context variable should remain unchanged in the sync function."
    assert result == "test_value", "Context variable should be propagated correctly."


@pytest.mark.asyncio
async def test_context_propagation_async():
    """
    Test that context is propagated correctly when using syncify with async functions.
    """
    import contextvars

    my_var = contextvars.ContextVar("my_var")
    my_var.set("test_value_async")

    @syncify
    async def context_test_async() -> str:
        v = my_var.get()
        my_var.set("test_value_async-updated")  # Update the context variable
        return v

    result = await context_test_async.aio()
    assert my_var.get() == "test_value_async", "Context variable should remain unchanged in the async function."
    assert result == "test_value_async", "Context variable should be propagated correctly in async function."


@pytest.mark.asyncio
async def test_syncify_async_iterator():
    """
    Test that syncify works with async iterators.
    """
    items = [item async for item in await async_iterator.aio(limit=5)]
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items, "Async iterator should yield correct items when using syncify."


def test_syncify_async_iterator_sync():
    """
    Test that syncify works with async iterators in sync context.
    """
    items = list(async_iterator(limit=5))
    expected_items = [f"Item {i + 1}" for i in range(5)]
    assert items == expected_items, "Async iterator should yield correct items when using syncify in sync context."


@syncify
async def deadlock_function():
    """
    Call syncify again to ensure it works correctly in nested calls.
    """
    _ = async_hello("Nested Call")


@syncify
async def nested_syncify() -> str:
    return await async_hello.aio("Nested Call")


def test_syncify_nested_calls():
    """
    Test that syncify can be called again within a syncified function.
    """
    result = nested_syncify()
    assert result == "Hello, Async World Nested Call!", "Syncify should work correctly in nested calls."


def test_exception_when_deadlock():
    """
    Test that syncify raises an exception when it detects a deadlock.
    """
    with pytest.raises(AssertionError, match="Deadlock detected:"):
        deadlock_function()


@syncify
async def nested_syncify_exception() -> str:
    """
    This function is intentionally designed to raise an exception to test error handling in syncify.
    """
    raise ValueError("This is a test exception from nested_syncify_exception.")


@syncify
async def syncify_generator_exception() -> AsyncGenerator[str, None]:
    yield "hello"
    raise ValueError("This is a test exception from syncify_generator_exception.")


def test_syncify_nested_exception_sync():
    try:
        nested_syncify_exception()
    except ValueError as e:
        assert str(e) == "This is a test exception from nested_syncify_exception."
        tb_list = traceback.extract_tb(e.__traceback__)
        assert len(tb_list) == 3, (
            "The Traceback should contain two frames: one for the exception and one for the function."
        )


@pytest.mark.asyncio
async def test_syncify_nested_exception():
    try:
        nested_syncify_exception.aio()
    except ValueError as e:
        assert str(e) == "This is a test exception from nested_syncify_exception."
        tb_list = traceback.extract_tb(e.__traceback__)
        assert len(tb_list) == 3, (
            "The Traceback should contain two frames: one for the exception and one for the function."
        )


def test_syncify_generator_exception_sync():
    try:
        list(syncify_generator_exception())
    except ValueError as e:
        assert str(e) == "This is a test exception from syncify_generator_exception."
        tb_list = traceback.extract_tb(e.__traceback__)
        assert len(tb_list) == 3, (
            f"traceback should contain two frames: one for the exception and one for the function, got {tb_list}"
        )


@pytest.mark.asyncio
async def test_syncify_generator_exception():
    try:
        async for x in syncify_generator_exception.aio():
            pass
    except ValueError as e:
        assert str(e) == "This is a test exception from syncify_generator_exception."
        tb_list = traceback.extract_tb(e.__traceback__)
        assert len(tb_list) == 3, (
            f"traceback should contain two frames: one for the exception and one for the function, got {tb_list}"
        )
