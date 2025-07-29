# /// script
# requires-python = "==3.12"
# dependencies = [
#    "click",
# ]
# ///
import click


@click.command()
@click.argument("a", type=int)
@click.argument("b", type=int)
def add(a: int, b: int):
    """Adds two numbers and prints the result."""
    result = a + b
    print(f"{result}")


if __name__ == "__main__":
    add()
