import flyte

env = flyte.TaskEnvironment(name="debug_example")


@env.task
def say_hello(name: str) -> str:
    greeting = f"Hello, {name}!"
    print(greeting)
    return greeting


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(debug=True).run(say_hello, name="World")
    print(run.name)
    print("Run url", run.url)
    print("Waiting for debug url...")
    print("Debug url", run.get_debug_url())
