import flyte

# Create a task environment
env = flyte.TaskEnvironment(
    name="hello_world_env",
    image="auto",  # Uses default Python image with Flyte installed
)

@env.task
def hello_world(name: str = "World") -> str:
    """A simple hello world task that takes a name and returns a greeting."""
    greeting = f"Hello, {name}!"
    print(greeting)
    return greeting

if __name__ == "__main__":

    import flyte, logging 

    flyte.init_from_config(log_level=logging.DEBUG)
    run = flyte.with_runcontext(mode="remote", log_level=logging.DEBUG).run(hello_world)
    print(run.name)
    print(run.url)