"""
# Syncify Module
This module provides the `syncify` decorator and the `Syncify` class.
The decorator can be used to convert asynchronous functions or methods into synchronous ones.
This is useful for integrating async code into synchronous contexts.

Every asynchronous function or method wrapped with `syncify` can be called synchronously using the
parenthesis `()` operator, or asynchronously using the `.aio()` method.

Example::

```python
from flyte.syncify import syncify

@syncify
async def async_function(x: str) -> str:
    return f"Hello, Async World {x}!"


# now you can call it synchronously
result = async_function("Async World")   # Note: no .aio() needed for sync calls
print(result)
# Output: Hello, Async World Async World!

# or call it asynchronously
async def main():
    result = await async_function.aio("World")    # Note the use of .aio() for async calls
    print(result)
```

## Creating a Syncify Instance
```python
from flyte.syncify. import Syncify

syncer = Syncify("my_syncer")

# Now you can use `syncer` to decorate your async functions or methods

```

## How does it work?
The Syncify class wraps asynchronous functions, classmethods, instance methods, and static methods to
 provide a synchronous interface. The wrapped methods are always executed in the context of a background loop,
 whether they are called synchronously or asynchronously. This allows for seamless integration of async code, as
 certain async libraries capture the event loop. An example is grpc.aio, which captures the event loop.
 In such a case, the Syncify class ensures that the async function is executed in the context of the background loop.

To use it correctly with grpc.aio, you should wrap every grpc.aio channel creation, and client invocation
with the same `Syncify` instance. This ensures that the async code runs in the correct event loop context.
"""

from flyte.syncify._api import Syncify

syncify = Syncify()

__all__ = ["Syncify", "syncify"]
