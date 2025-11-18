import pathlib

from dep import foo

import flyte

env = flyte.TaskEnvironment(
    name="full_build",
    # The image is built and it copies the contents to the working directory of the container including the folder by
    # default. If you want to copy only the contents of the folder, use `copy_contents_only=True`. This is useful when
    # you want to avoid nested folders - for example all your code is in the root of the repo.
    image=flyte.Image.from_debian_base().with_source_folder(pathlib.Path(__file__).parent, copy_contents_only=True),
)


@env.task
def square(x) -> int:
    return x ** foo()


@env.task
def main(n: int) -> list[int]:
    return list(flyte.map(square, range(n)))


if __name__ == "__main__":

    # Another important trick is to set the root_dir correctly. if copy_contents_only is False, then we can set the
    # root_dir to the parent.parent because the folder is copied. If copy_contents_only is True, then we need to set
    # the root_dir to parent because the contents are copied
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.with_runcontext(copy_style="none", version="x").run(main, n=10)
    print(run.url)
