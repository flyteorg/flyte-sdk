"""
Pure Python in a Sandbox
========================

Sandboxed tasks execute pure Python inside Monty — Pydantic's Rust-based
Python interpreter. They have **no** filesystem, network, or OS access,
start in microseconds, and many can safely share a single container.

This example shows the basics: defining sandboxed tasks with the
``@sandboxed.task`` decorator and calling them locally with ``forward()``.

Install the optional dependency first::

    pip install 'flyte[sandboxed]'
"""

from typing import Dict, List, Optional

import flyte
import flyte.sandboxed

# --- Basic tasks -------------------------------------------------------------
# Use ``@flyte.sandboxed.task`` exactly like ``@env.task``, but the body
# runs inside Monty instead of a full container.


@flyte.sandboxed.task
def add(x: int, y: int) -> int:
    return x + y


@flyte.sandboxed.task
def greet(name: str) -> str:
    return "Hello, " + name + "!"


# --- Multiple return styles --------------------------------------------------
# Monty supports the same return conventions as regular Python.


@flyte.sandboxed.task
def early_return(x: int) -> str:
    if x < 0:
        return "negative"
    if x == 0:
        return "zero"
    return "positive"


# --- Collection types --------------------------------------------------------
# list, dict, tuple, set, and their generic forms are all supported.


@flyte.sandboxed.task
def sum_list(numbers: List[int]) -> int:
    total = 0
    for n in numbers:
        total = total + n
    return total


@flyte.sandboxed.task
def word_lengths(words: List[str]) -> Dict[str, int]:
    result = {}
    for w in words:
        result[w] = len(w)
    return result


@flyte.sandboxed.task
def first_and_last(items: List[int]) -> tuple:
    return (items[0], items[len(items) - 1])


# --- Optional types ----------------------------------------------------------


@flyte.sandboxed.task
def maybe_double(x: int, flag: Optional[bool] = None) -> Optional[int]:
    if flag:
        return x * 2
    return None


# --- Loops and conditionals --------------------------------------------------


@flyte.sandboxed.task
def fizzbuzz(n: int) -> List[str]:
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


# --- Attach to an environment for ``flyte run`` -----------------------------
# ``@flyte.sandboxed.task`` creates standalone templates. ``flyte run``
# requires every task to belong to a TaskEnvironment.

sandbox_env = flyte.TaskEnvironment.from_task(
    "pure-python-demo",
    add,
    greet,
    early_return,
    sum_list,
    word_lengths,
    first_and_last,
    maybe_double,
    fizzbuzz,
)


if __name__ == "__main__":
    # forward() bypasses Monty and calls the Python function directly —
    # useful for local testing and debugging.
    print("add(2, 3) =", add.forward(2, 3))
    print("greet('world') =", greet.forward("world"))
    print("early_return(-1) =", early_return.forward(-1))
    print("early_return(0) =", early_return.forward(0))
    print("early_return(5) =", early_return.forward(5))
    print("sum_list([1,2,3,4]) =", sum_list.forward([1, 2, 3, 4]))
    print("word_lengths(['hi','bye']) =", word_lengths.forward(["hi", "bye"]))
    print("first_and_last([10,20,30]) =", first_and_last.forward([10, 20, 30]))
    print("maybe_double(5, True) =", maybe_double.forward(5, True))
    print("maybe_double(5) =", maybe_double.forward(5))
    print("fizzbuzz(15) =", fizzbuzz.forward(15))
