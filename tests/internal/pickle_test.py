import click
import cloudpickle
import cloudpickle as cp


def pickle_level3(x: str):
    print(f"Pickle level 3 -> {x}")


def pickle_level2(x: str):
    print("Pickle level 2")
    with open(x, "wb") as f:
        cloudpickle.dump(pickle_level3, f)


def pickle_level1(x: str):
    print("Pickle level 1")
    with open(x, "wb") as f:
        cloudpickle.dump(pickle_level2, f)


def pickle_level0(x: str):
    print("Pickle level 0")
    with open(x, "wb") as f:
        cloudpickle.dump(pickle_level1, f)


@click.command()
@click.argument("filename")
@click.argument("new_filename")
def pickle_now(filename, new_filename):
    """
    Pickle the given filename.
    """
    if filename == "level0":
        pickle_level0(new_filename)
        return
    with open(filename, "rb") as f:
        data = cp.load(f)
        data(new_filename)


if __name__ == "__main__":
    pickle_now()
