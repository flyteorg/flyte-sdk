from dep import foo

import flyte

env = flyte.TaskEnvironment(
    name="full_build",
    # with_code_bundle() automatically copies source code from root_dir into the image
    # when copy_style="none" is set in with_runcontext() or flyte deploy. When copy_style
    # is "loaded_modules" or "all", it's a no-op since the code is bundled separately.
    image=flyte.Image.from_debian_base().with_code_bundle(),
)


@env.task
def square(x) -> int:
    return x ** foo()


@env.task
def main(n: int) -> list[int]:
    return list(flyte.map(square, range(n)))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(copy_style="none", version="x").run(main, n=10)
    print(run.url)
